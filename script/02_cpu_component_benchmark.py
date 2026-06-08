#!/usr/bin/env python3
"""
CPU Component Micro-Benchmark
==============================
Isolate and measure each CPU-side operation in the inference pipeline:
  - JSON serialization   (request payload construction)
  - SHA256 hash          (KV cache key computation)
  - SSE chunk parsing    (streaming response processing)
  - Detokenization       (token ID → text for streaming output)

Reference: BLOG.md Section 3.3 "Where Does CPU Time Actually Go?"
Targets:
  - JSON serialize at 100k tokens: < 1 ms
  - SHA256 hash at 100k tokens:    < 1 ms
  - SSE chunk parse:               ~2 µs per chunk
  - Detokenization (128 tokens):   < 0.3 ms

Usage:
    python script/02_cpu_component_benchmark.py [--model MODEL_ID] [--output OUTPUT_PATH]

Output:
    result/02_cpu_component_benchmark.json
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CPU component micro-benchmark")
parser.add_argument(
    "--model", default="Qwen/Qwen3-30B-A3B",
    help="HuggingFace model ID for tokenizer (detokenization benchmark)"
)
parser.add_argument(
    "--output", default=None,
    help="Output JSON path (default: result/02_cpu_component_benchmark.json)"
)
parser.add_argument(
    "--token-lengths", nargs="+", type=int,
    default=[1000, 8000, 32000, 100000],
    help="Token lengths to simulate for serialization/hashing"
)
parser.add_argument(
    "--iterations", type=int, default=100,
    help="Measurement iterations per component"
)
args = parser.parse_args()

if args.output:
    out_path = Path(args.output)
else:
    out_path = Path(__file__).resolve().parents[1] / "result" / "02_cpu_component_benchmark.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load tokenizer (for detokenization benchmark only)
# ---------------------------------------------------------------------------
print(f"[02_cpu] Loading tokenizer: {args.model}")
t0 = time.perf_counter()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
load_time = time.perf_counter() - t0
print(f"[02_cpu] Tokenizer loaded in {load_time:.2f}s")

# ---------------------------------------------------------------------------
# Generate simulated data at target token lengths
# ---------------------------------------------------------------------------
# For JSON serialization and hashing, we need text of ~N tokens.
# Approximate: 1 token ≈ 4 characters for English text.
def generate_text_for_tokens(target_tokens: int) -> tuple:
    """Generate a text blob of approximately `target_tokens` tokens.
    Returns (text, actual_tokens)."""
    chars_needed = target_tokens * 5  # conservative over-estimate
    base = (
        '{"role": "user", "content": "This is a simulated agentic conversation '
        'turn for benchmarking CPU overhead in LLM inference pipelines. '
        'The quick brown fox jumps over the lazy dog. '
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. "}'
    )
    repeats = (chars_needed // len(base)) + 1
    text = (base * repeats)[:chars_needed]
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, actual

# Pre-build texts
texts = {}
for n in args.token_lengths:
    text, actual = generate_text_for_tokens(n)
    texts[n] = text
    print(f"[02_cpu] Generated ~{n:,} token text → actual {actual:,} tokens")

# ---------------------------------------------------------------------------
# 1. JSON Serialization Benchmark
# ---------------------------------------------------------------------------
print("\n[02_cpu] === JSON Serialization ===")
json_results = []

for target_len in args.token_lengths:
    text = texts[target_len]
    # Build a realistic OpenAI-compatible chat request payload
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to file reading, code search, and shell execution tools."},
            {"role": "user", "content": text},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
        "stream": True,
    }

    # Warmup
    for _ in range(10):
        json.dumps(payload, ensure_ascii=False)
    # Measure
    times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        s = json.dumps(payload, ensure_ascii=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)
    json_results.append({
        "target_tokens": target_len,
        "payload_size_bytes": len(s),
        "serialize_ms": round(avg_ms, 4),
    })
    print(f"  {target_len:>7,} tokens → {len(s):>10,} bytes → {avg_ms:>8.4f} ms avg")

# ---------------------------------------------------------------------------
# 2. SHA256 Hash Benchmark
# ---------------------------------------------------------------------------
print("\n[02_cpu] === SHA256 Hash ===")
hash_results = []

for target_len in args.token_lengths:
    text = texts[target_len]
    data = text.encode("utf-8")

    # Warmup
    for _ in range(10):
        hashlib.sha256(data).hexdigest()
    # Measure
    times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        h = hashlib.sha256(data).hexdigest()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)
    throughput_mbps = (len(data) / (1024 * 1024)) / (avg_ms / 1000)
    hash_results.append({
        "target_tokens": target_len,
        "data_size_bytes": len(data),
        "hash_ms": round(avg_ms, 4),
        "throughput_mbps": round(throughput_mbps, 1),
    })
    print(f"  {target_len:>7,} tokens → {len(data):>10,} bytes → "
          f"{avg_ms:>8.4f} ms  ({throughput_mbps:>8,.1f} MB/s)")

# ---------------------------------------------------------------------------
# 3. SSE Chunk Parse Benchmark
# ---------------------------------------------------------------------------
print("\n[02_cpu] === SSE Chunk Parsing ===")

# Simulate realistic vLLM SSE chunks
# vLLM streaming chunks look like:
#   data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

SSE_CHUNKS = [
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":" quick"},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":" brown"},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":" fox"},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":" jumps"},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":" over"},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":" lazy"},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":" dog"},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"content":"."},"finish_reason":null}]}\n\n',
    'data: {"id":"cmpl-abc123","object":"chat.completion.chunk","created":1715000000,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"function":{"name":"read_file","arguments":"{\\"path\\":\\"/src/main.py\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
    'data: [DONE]\n\n',
]

NUM_CHUNKS_TO_TEST = [1, 10, 100, 1000, 10000]

sse_results = []
for n_chunks in NUM_CHUNKS_TO_TEST:
    # Build a stream of n_chunks (cycle the templates)
    cycle = SSE_CHUNKS * ((n_chunks // len(SSE_CHUNKS)) + 1)
    chunks = cycle[:n_chunks]

    # Warmup
    for _ in range(5):
        for chunk in chunks[:10]:
            line = chunk.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                json.loads(line[6:])

    # Measure
    t0 = time.perf_counter()
    parsed = 0
    for chunk in chunks:
        line = chunk.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                json.loads(line[6:])
                parsed += 1
            except json.JSONDecodeError:
                pass
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000
    per_chunk_us = (total_ms / n_chunks) * 1000 if n_chunks > 0 else 0

    sse_results.append({
        "num_chunks": n_chunks,
        "parsed_successfully": parsed,
        "total_ms": round(total_ms, 4),
        "per_chunk_us": round(per_chunk_us, 3),
    })
    print(f"  {n_chunks:>6,} chunks → {total_ms:>10.4f}ms total → {per_chunk_us:>8.3f} µs/chunk")

# ---------------------------------------------------------------------------
# 4. Detokenization Benchmark
# ---------------------------------------------------------------------------
print("\n[02_cpu] === Detokenization ===")
detok_results = []

# Generate random token IDs (simulating model output)
import random
random.seed(42)

for n_tokens in [1, 16, 64, 128, 256, 512]:
    # Use real token IDs from the tokenizer's vocabulary range
    vocab_size = tokenizer.vocab_size
    token_ids = [random.randint(0, min(vocab_size - 1, 150000)) for _ in range(n_tokens)]

    # Warmup
    for _ in range(10):
        tokenizer.decode(token_ids, skip_special_tokens=True)
    # Measure
    times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        tokenizer.decode(token_ids, skip_special_tokens=True)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)
    detok_results.append({
        "num_tokens": n_tokens,
        "decode_ms": round(avg_ms, 4),
    })
    print(f"  {n_tokens:>4} tokens → {avg_ms:>8.4f} ms")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output = {
    "benchmark": "cpu_component_micro",
    "model": args.model,
    "iterations": args.iterations,
    "tokenizer_load_time_s": round(load_time, 3),
    "json_serialization": json_results,
    "sha256_hash": hash_results,
    "sse_chunk_parse": sse_results,
    "detokenization": detok_results,
}

out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
print(f"\n[02_cpu] Results saved to {out_path}")

# Print comparison table
print("\n" + "=" * 80)
print("SUMMARY: CPU Component Overhead (matching blog Section 3.3)")
print("=" * 80)
print(f"{'Component':<30s} {'Size':>12s} {'Time':>12s} {'Blog Target':>15s}")
print("-" * 80)
# JSON at 100k
j100k = next((j for j in json_results if j["target_tokens"] == 100000), None)
if j100k:
    j_val = f"{j100k['serialize_ms']:.2f} ms"
    print(f"{'JSON serialize':<30s} {'100k tokens':>12s} {j_val:>12s} {'< 0.82 ms':>15s}")
# Hash at 100k
h100k = next((h for h in hash_results if h["target_tokens"] == 100000), None)
if h100k:
    h_val = f"{h100k['hash_ms']:.2f} ms"
    print(f"{'SHA256 hash':<30s} {'100k tokens':>12s} {h_val:>12s} {'< 0.62 ms':>15s}")
# SSE per chunk
sse10k = next((s for s in sse_results if s["num_chunks"] == 10000), None)
if sse10k:
    s_val = f"{sse10k['per_chunk_us']:.2f} µs"
    print(f"{'SSE parse/chunk':<30s} {'per chunk':>12s} {s_val:>12s} {'~1.9 µs':>15s}")
# Detok 128
d128 = next((d for d in detok_results if d["num_tokens"] == 128), None)
if d128:
    d_val = f"{d128['decode_ms']:.3f} ms"
    print(f"{'Detokenization':<30s} {'128 tokens':>12s} {d_val:>12s} {'< 0.27 ms':>15s}")
print("=" * 80)
