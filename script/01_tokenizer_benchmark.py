#!/usr/bin/env python3
"""
Tokenizer CPU Micro-Benchmark
==============================
Measures encode/decode throughput for the Qwen3-30B-A3B tokenizer
at various token lengths to validate linear scaling and ~500k tok/s throughput.

Reference: BLOG.md Section 3.4 "Tokenization Deep-Dive"
Target:   ~500k tok/s encode, ~4M tok/s decode, linear scaling with input length

Usage:
    python script/01_tokenizer_benchmark.py [--model MODEL_ID] [--output OUTPUT_PATH]

Output:
    result/01_tokenizer_benchmark.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Tokenizer CPU micro-benchmark")
parser.add_argument(
    "--model", default="Qwen/Qwen3-30B-A3B",
    help="HuggingFace model ID (default: Qwen/Qwen3-30B-A3B)"
)
parser.add_argument(
    "--output", default=None,
    help="Output JSON path (default: result/01_tokenizer_benchmark.json)"
)
parser.add_argument(
    "--token-lengths", nargs="+", type=int,
    default=[679, 2711, 5423, 10840, 21679, 43359, 67745, 101615],
    help="Token lengths to benchmark (default: matching blog matrix)"
)
parser.add_argument(
    "--warmup-iters", type=int, default=3,
    help="Warmup iterations per length"
)
parser.add_argument(
    "--measure-iters", type=int, default=10,
    help="Measurement iterations per length"
)
args = parser.parse_args()

# Output path
if args.output:
    out_path = Path(args.output)
else:
    out_path = Path(__file__).resolve().parents[1] / "result" / "01_tokenizer_benchmark.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------------
print(f"[01_tokenizer] Loading tokenizer: {args.model}")
t0 = time.perf_counter()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
load_time = time.perf_counter() - t0
print(f"[01_tokenizer] Tokenizer loaded in {load_time:.2f}s")

# ---------------------------------------------------------------------------
# Generate synthetic text of target token lengths
# ---------------------------------------------------------------------------
# We use repeated padding text so that lengths are reproducible.
# Start with a long seed text, tokenize, then slice/expand.
SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 500
    + "System: You are a helpful AI assistant. You have access to tools for "
      "reading files, searching code, and executing shell commands. "
      "Always respond with accurate and helpful information. " * 300
    + "User: Please analyze the following codebase and identify potential "
      "performance bottlenecks in the scheduling algorithm. " * 200
)

# Pre-tokenize the seed
seed_ids = tokenizer.encode(SEED_TEXT, add_special_tokens=False)
print(f"[01_tokenizer] Seed text has {len(seed_ids)} tokens")

def text_at_length(target_tokens: int) -> str:
    """Return a text string of approximately `target_tokens` tokens."""
    if target_tokens <= len(seed_ids):
        ids = seed_ids[:target_tokens]
    else:
        # Repeat seed enough times
        repeats = (target_tokens // len(seed_ids)) + 1
        ids = (seed_ids * repeats)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

# Pre-build texts for all lengths
texts = {}
for n in args.token_lengths:
    texts[n] = text_at_length(n)
    # Verify
    verify_ids = tokenizer.encode(texts[n], add_special_tokens=False)
    print(f"[01_tokenizer] Target={n:>7,}  Actual={len(verify_ids):>7,}  "
          f"Error={abs(len(verify_ids)-n):>5,} tokens")

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
results = []
incremental_decode_tokens = 128  # Match blog: 128 output tokens

for target_len in args.token_lengths:
    text = texts[target_len]

    # -- Encode -----------------------------------------------------------------
    encode_times = []
    for _ in range(args.warmup_iters):
        tokenizer.encode(text, add_special_tokens=False)
    for _ in range(args.measure_iters):
        t0 = time.perf_counter()
        tok_ids = tokenizer.encode(text, add_special_tokens=False)
        t1 = time.perf_counter()
        encode_times.append((t1 - t0) * 1000)  # ms

    avg_encode_ms = sum(encode_times) / len(encode_times)
    actual_tokens = len(tok_ids)
    encode_tok_per_s = actual_tokens / (avg_encode_ms / 1000)

    # -- Full Decode ------------------------------------------------------------
    decode_times = []
    for _ in range(args.warmup_iters):
        tokenizer.decode(tok_ids, skip_special_tokens=True)
    for _ in range(args.measure_iters):
        t0 = time.perf_counter()
        tokenizer.decode(tok_ids, skip_special_tokens=True)
        t1 = time.perf_counter()
        decode_times.append((t1 - t0) * 1000)

    avg_decode_ms = sum(decode_times) / len(decode_times)
    decode_tok_per_s = actual_tokens / (avg_decode_ms / 1000)

    # -- Incremental Decode (streaming, last 128 tokens) -----------------------
    incr_ids = tok_ids[-incremental_decode_tokens:] if len(tok_ids) >= incremental_decode_tokens else tok_ids
    incr_times = []
    for _ in range(args.warmup_iters):
        tokenizer.decode(incr_ids, skip_special_tokens=True)
    for _ in range(args.measure_iters):
        t0 = time.perf_counter()
        tokenizer.decode(incr_ids, skip_special_tokens=True)
        t1 = time.perf_counter()
        incr_times.append((t1 - t0) * 1000)

    avg_incr_ms = sum(incr_times) / len(incr_times)

    entry = {
        "target_tokens": target_len,
        "actual_tokens": actual_tokens,
        "encode_ms": round(avg_encode_ms, 3),
        "encode_tok_per_s": round(encode_tok_per_s, 0),
        "decode_ms": round(avg_decode_ms, 3),
        "decode_tok_per_s": round(decode_tok_per_s, 0),
        "incremental_decode_128tokens_ms": round(avg_incr_ms, 3),
    }
    results.append(entry)
    print(f"[01_tokenizer] {target_len:>7,} tokens | "
          f"encode: {avg_encode_ms:>8.2f}ms ({encode_tok_per_s:>10,.0f} tok/s) | "
          f"decode: {avg_decode_ms:>8.2f}ms ({decode_tok_per_s:>10,.0f} tok/s) | "
          f"incr(128): {avg_incr_ms:.3f}ms")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
output = {
    "benchmark": "tokenizer_cpu",
    "model": args.model,
    "tokenizer_load_time_s": round(load_time, 3),
    "warmup_iterations": args.warmup_iters,
    "measure_iterations": args.measure_iters,
    "incremental_decode_tokens": incremental_decode_tokens,
    "results": results,
}

out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
print(f"\n[01_tokenizer] Results saved to {out_path}")

# Print summary table
print("\n" + "=" * 85)
print("SUMMARY: Tokenizer CPU Throughput")
print("=" * 85)
print(f"{'Tokens':>8s}  {'Encode ms':>10s}  {'Encode tok/s':>14s}  "
      f"{'Decode ms':>10s}  {'Decode tok/s':>14s}  {'Incr(128) ms':>14s}")
print("-" * 85)
for r in results:
    print(f"{r['target_tokens']:>8,}  {r['encode_ms']:>10.2f}  {r['encode_tok_per_s']:>14,.0f}  "
          f"{r['decode_ms']:>10.2f}  {r['decode_tok_per_s']:>14,.0f}  {r['incremental_decode_128tokens_ms']:>14.3f}")
print("=" * 85)
