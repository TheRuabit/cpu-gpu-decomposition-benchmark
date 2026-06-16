#!/usr/bin/env python3
"""
E2E Request Time Decomposition (Client-Side)
=============================================
Instruments the full request lifecycle from the client perspective:
  t_serialize       — JSON request body construction
  t_network_to_first_byte — HTTP POST until first SSE byte (≈ t_http_overhead on localhost)
  t_ttft            — Time to first token (includes prefill + decode of 1st token)
  t_decode          — Streaming token generation duration
  t_response_parse  — SSE chunk parsing + tool call extraction
  t_e2e             — Total wall-clock from serialize start to last byte parsed

When running against a local vLLM server (no network latency), we can decompose:
  t_http_overhead   ≈ t_network_to_first_byte  (scheduling + tokenization + queue wait)
  t_prefill         ≈ t_ttft - t_network_to_first_byte
  t_decode          = time from first token byte to last

Reference: BLOG.md Section 2.2 "What We Measured"

Usage:
    # Single request with 1k context
    python script/03_request_profiler.py --url http://localhost:8000 --context 1000

    # 4 concurrent requests with 8k context
    python script/03_request_profiler.py --url http://localhost:8000 --context 8000 --concurrency 4

    # Full matrix (used by run_all.sh)
    python script/03_request_profiler.py --url http://localhost:8000 --matrix

Output:
    result/03_request_profiler.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="E2E request time decomposition (client-side)")
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM OpenAI-compatible server base URL")
parser.add_argument("--model", default="./models/Qwen3-30B-A3B",
                    help="Model name to pass in API requests")
parser.add_argument("--context", type=int, default=1000,
                    help="Target context length in tokens")
parser.add_argument("--concurrency", type=int, default=1,
                    help="Number of concurrent requests")
parser.add_argument("--output-tokens", type=int, default=512,
                    help="Max output tokens per request")
parser.add_argument("--num-batches", type=int, default=3,
                    help="Number of batches to run (for averaging)")
parser.add_argument("--requests-per-batch", type=int, default=None,
                    help="Requests per batch (default: same as concurrency)")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
parser.add_argument("--matrix", action="store_true",
                    help="Run full test matrix (overrides --context and --concurrency)")
parser.add_argument("--warmup", type=int, default=1,
                    help="Warmup batches before measurement")
args = parser.parse_args()

if args.output:
    out_path = Path(args.output)
else:
    out_path = Path(__file__).resolve().parents[1] / "result" / "03_request_profiler.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Test matrix (matching blog Table in Section 2.3)
# ---------------------------------------------------------------------------
TEST_MATRIX = [
    # (scenario_name, concurrency, context_length)
    ("single_1k",    1,  1000),
    ("single_8k",    1,  8000),
    ("single_32k",   1,  32000),
    ("single_100k",  1,  100000),
    ("conc4_8k",     4,  8000),
    ("conc16_32k",  16,  32000),
    ("conc32_32k",  32,  32000),
    ("conc32_50k", 32,  50000),
]

# ---------------------------------------------------------------------------
# Request timing data class
# ---------------------------------------------------------------------------
@dataclass
class RequestTiming:
    scenario: str = ""
    batch: int = 0
    request_idx: int = 0
    context_tokens: int = 0
    output_tokens: int = 0
    t_serialize_ms: float = 0.0
    t_first_byte_ms: float = 0.0   # HTTP POST → first SSE byte (≈ t_http_overhead)
    t_ttft_ms: float = 0.0         # HTTP POST → first token content byte
    t_decode_ms: float = 0.0       # First token → last token
    t_response_parse_ms: float = 0.0
    t_e2e_ms: float = 0.0
    first_token_id: int = 0
    num_output_tokens: int = 0
    success: bool = True
    error: str = ""

# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------
import aiohttp

# Pre-build context texts (lazily cached)
_CONTEXT_CACHE: dict[int, str] = {}

def _build_seed_text() -> str:
    """Build a large seed text that will serve as the basis for context generation."""
    return (
        # System prompt (typical agentic system prompt ~1-2k tokens)
        "System: You are an AI coding assistant with access to tools for reading files, "
        "searching codebases, executing shell commands, and editing code. Always respond "
        "accurately and use tools when appropriate.\n\n"
        # Conversation turns (to simulate agentic workload)
        "User: Please analyze the performance characteristics of the following code:\n\n"
        + ("```python\n" + "def process_data(items):\n    results = []\n"
           "    for item in items:\n        "
           "results.append(transform(item))\n    return results\n```\n\n" * 50)
        + "Assistant: Let me analyze this code step by step.\n\n"
        + "the project structure is well organized with clear separation of concerns. " * 100
    )

def get_context_text(target_tokens: int) -> str:
    """Get or generate text of approximately `target_tokens` tokens."""
    if target_tokens in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[target_tokens]

    seed = _build_seed_text()
    # Use a simple character-based approximation: ~4 chars/token for English
    chars_per_token = 4
    chars_needed = target_tokens * chars_per_token

    if chars_needed <= len(seed):
        text = seed[:chars_needed]
    else:
        repeats = (chars_needed // len(seed)) + 1
        text = (seed * repeats)[:chars_needed]

    _CONTEXT_CACHE[target_tokens] = text
    return text

def build_payload(context_text: str, model: str, max_tokens: int) -> dict:
    """Build an OpenAI-compatible chat completions request."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": context_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------
def compute_stats(values: list[float], ndigits: int = 20) -> dict:
    """Compute mean, p50, p95, min, max for a list of values."""
    if not values:
        return {"mean": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
    n = len(values)
    s = sorted(values)
    return {
        "mean": round(sum(values) / n, ndigits),
        "p50": round(s[n // 2], ndigits),
        "p95": round(s[int(n * 0.95)], ndigits),
        "min": round(s[0], ndigits),
        "max": round(s[-1], ndigits),
    }
async def profile_single_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    scenario: str,
    batch: int,
    request_idx: int,
    context_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestTiming:
    """Send one streaming request and time each phase."""
    t = RequestTiming(scenario=scenario, batch=batch,
                      request_idx=request_idx, context_tokens=context_tokens)

    t_e2e_start = time.perf_counter()

    # ----- t_serialize -----
    t0 = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False)
    t.t_serialize_ms = (time.perf_counter() - t0) * 1000

    async with semaphore:
        try:
            # ----- HTTP POST → first byte -----
            t_post = time.perf_counter()
            async with session.post(
                f"{url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    t.success = False
                    t.error = f"HTTP {resp.status}: {error_body[:500]}"
                    t.t_e2e_ms = (time.perf_counter() - t_e2e_start) * 1000
                    return t

                first_byte_received = False
                first_token_received = False
                t_first_byte = None
                t_first_token = None
                t_last_token = None
                output_content = ""
                parse_start = None

                # Iterate over SSE chunks
                async for line in resp.content:
                    if not first_byte_received:
                        t_first_byte = time.perf_counter()
                        t.t_first_byte_ms = (t_first_byte - t_post) * 1000
                        first_byte_received = True
                        parse_start = time.perf_counter()

                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        try:
                            chunk = json.loads(line_str[6:])
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if not first_token_received:
                                        t_first_token = time.perf_counter()
                                        t.t_ttft_ms = (t_first_token - t_post) * 1000
                                        first_token_received = True
                                    t_last_token = time.perf_counter()
                                    output_content += content
                        except json.JSONDecodeError:
                            pass

                # ----- t_response_parse -----
                t_parse_end = time.perf_counter()
                if parse_start:
                    t.t_response_parse_ms = (t_parse_end - parse_start) * 1000

                # ----- t_decode -----
                if first_token_received and t_last_token:
                    t.t_decode_ms = (t_last_token - t_first_token) * 1000

                t.num_output_tokens = len(output_content.split())  # rough estimate
                t.output_tokens = t.num_output_tokens

        except asyncio.TimeoutError:
            t.success = False
            t.error = "Timeout"
        except Exception as e:
            t.success = False
            t.error = str(e)[:500]

    t.t_e2e_ms = (time.perf_counter() - t_e2e_start) * 1000
    return t

# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
async def run_batch(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    scenario: str,
    batch: int,
    concurrency: int,
    context_tokens: int,
    output_tokens: int,
    requests_per_batch: int,
) -> list[RequestTiming]:
    """Run a batch of concurrent requests."""
    context_text = get_context_text(context_tokens)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = []
    for i in range(requests_per_batch):
        payload = build_payload(context_text, model, output_tokens)
        task = profile_single_request(
            session, url, payload, scenario, batch, i,
            context_tokens, semaphore
        )
        tasks.append(task)

    print(f"  [{scenario}] batch={batch}: Dispatching {len(tasks)} requests "
          f"(concurrency={concurrency}, ctx={context_tokens})...", end=" ", flush=True)
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    ok = sum(1 for r in results if r.success)
    print(f"Done in {elapsed:.1f}s ({ok}/{len(results)} OK)")
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    # Health check
    url = args.url.rstrip("/")
    print(f"[03_profiler] Testing connection to {url} ...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                print(f"[03_profiler] Server health: HTTP {resp.status}")
    except Exception as e:
        print(f"[03_profiler] WARNING: Could not reach server at {url}: {e}")
        print("[03_profiler] Continuing anyway — will attempt API calls...")

    # Build run list
    if args.matrix:
        runs = TEST_MATRIX
        print(f"[03_profiler] Running full test matrix: {len(runs)} scenarios")
    else:
        runs = [(f"custom_c{args.concurrency}_ctx{args.context}",
                 args.concurrency, args.context)]
        print(f"[03_profiler] Single scenario: {runs[0]}")

    all_timings: list[RequestTiming] = []

    async with aiohttp.ClientSession() as session:
        for scenario_name, concurrency, ctx_len in runs:
            rpb = args.requests_per_batch if args.requests_per_batch else concurrency

            print(f"\n{'─'*60}")
            print(f"[03_profiler] Scenario: {scenario_name}  "
                  f"(conc={concurrency}, ctx={ctx_len}, req/batch={rpb})")
            print(f"{'─'*60}")

            # Warmup
            if args.warmup > 0:
                print(f"  Warming up ({args.warmup} batch(es))...")
                for w in range(args.warmup):
                    await run_batch(session, url, args.model,
                                    f"{scenario_name}_warmup", w,
                                    concurrency, ctx_len, args.output_tokens, rpb)

            # Measurement batches
            for b in range(args.num_batches):
                batch_timings = await run_batch(
                    session, url, args.model,
                    scenario_name, b, concurrency, ctx_len,
                    args.output_tokens, rpb
                )
                all_timings.extend(batch_timings)

    # -------------------------------------------------------------------
    # Aggregate & save
    # -------------------------------------------------------------------
    # Convert to serializable dicts (raw timings at 20dp)
    timing_dicts = []
    for t in all_timings:
        d = {
            "scenario": t.scenario,
            "batch": t.batch,
            "request_idx": t.request_idx,
            "context_tokens": t.context_tokens,
            "output_tokens": t.output_tokens,
            "t_serialize_ms": round(t.t_serialize_ms, 20),
            "t_first_byte_ms": round(t.t_first_byte_ms, 20),
            "t_ttft_ms": round(t.t_ttft_ms, 20),
            "t_decode_ms": round(t.t_decode_ms, 20),
            "t_response_parse_ms": round(t.t_response_parse_ms, 20),
            "t_e2e_ms": round(t.t_e2e_ms, 20),
            "success": t.success,
            "error": t.error,
        }
        timing_dicts.append(d)

    # Aggregate by scenario — compute full stats per metric
    from collections import defaultdict
    scenario_groups = defaultdict(list)
    for t in all_timings:
        if t.success:
            scenario_groups[t.scenario].append(t)

    aggregates = {}
    for scenario, timings in sorted(scenario_groups.items()):
        if not timings:
            continue
        n = len(timings)

        serialize_stats = compute_stats([t.t_serialize_ms for t in timings])
        first_byte_stats = compute_stats([t.t_first_byte_ms for t in timings])
        ttft_stats = compute_stats([t.t_ttft_ms for t in timings])
        decode_stats = compute_stats([t.t_decode_ms for t in timings])
        parse_stats = compute_stats([t.t_response_parse_ms for t in timings])
        e2e_stats = compute_stats([t.t_e2e_ms for t in timings])

        # Derived: http_overhead ≈ first_byte (localhost)
        http_oh_vals = [t.t_first_byte_ms for t in timings]
        http_oh_stats = compute_stats(http_oh_vals)
        # Derived: prefill ≈ ttft - first_byte
        prefill_vals = [max(0, t.t_ttft_ms - t.t_first_byte_ms) for t in timings]
        prefill_stats = compute_stats(prefill_vals)
        # Derived: total = serialize + first_byte + decode (CPU + GPU)
        total_vals = [t.t_serialize_ms + t.t_first_byte_ms + t.t_decode_ms for t in timings]
        total_stats = compute_stats(total_vals)
        # CPU time = serialize + http_overhead
        cpu_vals = [t.t_serialize_ms + t.t_first_byte_ms for t in timings]
        cpu_stats = compute_stats(cpu_vals)
        # GPU time = prefill + decode
        gpu_vals = [max(0, t.t_ttft_ms - t.t_first_byte_ms) + t.t_decode_ms for t in timings]
        gpu_stats = compute_stats(gpu_vals)
        # Percentages from mean values
        cpu_pct = round((cpu_stats["mean"] / total_stats["mean"] * 100) if total_stats["mean"] > 0 else 0, 10)
        gpu_pct = round((gpu_stats["mean"] / total_stats["mean"] * 100) if total_stats["mean"] > 0 else 0, 10)

        aggregates[scenario] = {
            "count": n,
            "t_serialize_ms_mean": serialize_stats["mean"],
            "t_serialize_ms_p50": serialize_stats["p50"],
            "t_serialize_ms_p95": serialize_stats["p95"],
            "t_serialize_ms_min": serialize_stats["min"],
            "t_serialize_ms_max": serialize_stats["max"],
            "t_first_byte_ms_mean": first_byte_stats["mean"],
            "t_first_byte_ms_p50": first_byte_stats["p50"],
            "t_first_byte_ms_p95": first_byte_stats["p95"],
            "t_first_byte_ms_min": first_byte_stats["min"],
            "t_first_byte_ms_max": first_byte_stats["max"],
            "t_http_overhead_ms_mean": http_oh_stats["mean"],
            "t_http_overhead_ms_p50": http_oh_stats["p50"],
            "t_http_overhead_ms_p95": http_oh_stats["p95"],
            "t_http_overhead_ms_min": http_oh_stats["min"],
            "t_http_overhead_ms_max": http_oh_stats["max"],
            "t_ttft_ms_mean": ttft_stats["mean"],
            "t_ttft_ms_p50": ttft_stats["p50"],
            "t_ttft_ms_p95": ttft_stats["p95"],
            "t_ttft_ms_min": ttft_stats["min"],
            "t_ttft_ms_max": ttft_stats["max"],
            "t_prefill_ms_mean": prefill_stats["mean"],
            "t_prefill_ms_p50": prefill_stats["p50"],
            "t_prefill_ms_p95": prefill_stats["p95"],
            "t_prefill_ms_min": prefill_stats["min"],
            "t_prefill_ms_max": prefill_stats["max"],
            "t_decode_ms_mean": decode_stats["mean"],
            "t_decode_ms_p50": decode_stats["p50"],
            "t_decode_ms_p95": decode_stats["p95"],
            "t_decode_ms_min": decode_stats["min"],
            "t_decode_ms_max": decode_stats["max"],
            "t_response_parse_ms_mean": parse_stats["mean"],
            "t_response_parse_ms_p50": parse_stats["p50"],
            "t_response_parse_ms_p95": parse_stats["p95"],
            "t_response_parse_ms_min": parse_stats["min"],
            "t_response_parse_ms_max": parse_stats["max"],
            "t_e2e_ms_mean": e2e_stats["mean"],
            "t_e2e_ms_p50": e2e_stats["p50"],
            "t_e2e_ms_p95": e2e_stats["p95"],
            "t_e2e_ms_min": e2e_stats["min"],
            "t_e2e_ms_max": e2e_stats["max"],
            "t_total_ms_mean": total_stats["mean"],
            "t_total_ms_p50": total_stats["p50"],
            "t_total_ms_p95": total_stats["p95"],
            "t_total_ms_min": total_stats["min"],
            "t_total_ms_max": total_stats["max"],
            "cpu_time_ms_mean": cpu_stats["mean"],
            "cpu_time_ms_p50": cpu_stats["p50"],
            "cpu_time_ms_p95": cpu_stats["p95"],
            "cpu_time_ms_min": cpu_stats["min"],
            "cpu_time_ms_max": cpu_stats["max"],
            "gpu_time_ms_mean": gpu_stats["mean"],
            "gpu_time_ms_p50": gpu_stats["p50"],
            "gpu_time_ms_p95": gpu_stats["p95"],
            "gpu_time_ms_min": gpu_stats["min"],
            "gpu_time_ms_max": gpu_stats["max"],
            "cpu_percent": cpu_pct,
            "gpu_percent": gpu_pct,
        }

    output = {
        "benchmark": "e2e_request_profiler",
        "server_url": url,
        "model": args.model,
        "num_batches": args.num_batches,
        "output_tokens_target": args.output_tokens,
        "runs": runs,
        "aggregates": aggregates,
        "raw_timings": timing_dicts,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[03_profiler] Results saved to {out_path}")

    # Summary table
    print("\n" + "=" * 95)
    print("SUMMARY: E2E Request Time Decomposition")
    print("=" * 95)
    header = (f"{'Scenario':<16s} {'N':>4s} {'Serialize':>10s} {'1st Byte':>10s} "
              f"{'TTFT':>10s} {'Decode':>10s} {'Parse':>10s} {'E2E':>10s}")
    print(header)
    print("-" * 95)
    for scenario, agg in sorted(aggregates.items()):
        print(f"{scenario:<16s} {agg['count']:>4d} "
              f"{agg['t_serialize_ms_mean']:>8.2f}ms "
              f"{agg['t_first_byte_ms_mean']:>8.1f}ms "
              f"{agg['t_ttft_ms_mean']:>8.1f}ms "
              f"{agg['t_decode_ms_mean']:>8.1f}ms "
              f"{agg['t_response_parse_ms_mean']:>8.2f}ms "
              f"{agg['t_e2e_ms_mean']:>8.1f}ms")
    print("=" * 95)

if __name__ == "__main__":
    asyncio.run(main())
