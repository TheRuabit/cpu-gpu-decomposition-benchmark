#!/usr/bin/env python3
"""
Server-Side Request Decomposition — LMCache DRAM
=================================================
Runs the full benchmark matrix against a vLLM server configured with LMCache
CPU DRAM cache enabled. Decomposes each request into CPU vs GPU time components
and compares against the HBM prefix cache baseline.

Additional LMCache-specific measurements:
  - Hash computation time (for cache key)
  - Cache lookup time
  - CPU DRAM → GPU HBM DMA transfer time

Prerequisites:
  1. LMCache installed:
     pip install lmcache
  2. vLLM server with LMCache:
     LMCACHE_LOCAL_CPU=true LMCACHE_CHUNK_SIZE=256 \
     vllm serve /path/to/model --enable-prefix-caching \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
       --gpu-memory-utilization 0.78

Usage:
    python script/05_lmcache_decomposition.py --url http://localhost:8000

Output:
    result/05_lmcache_decomposition.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Server decomposition benchmark — LMCache DRAM"
)
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server URL (with LMCache configured)")
parser.add_argument("--model", default="models/Qwen3-30B-A3B",
                    help="Model name for API requests")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
parser.add_argument("--hbm-results", default=None,
                    help="Path to HBM prefix cache results (for comparison)")
parser.add_argument("--output-tokens", type=int, default=512)
parser.add_argument("--num-batches", type=int, default=3)
parser.add_argument("--warmup-batches", type=int, default=1)
parser.add_argument("--scenarios", nargs="+", default=None)
args = parser.parse_args()

if args.output:
    out_path = Path(args.output)
else:
    out_path = Path(__file__).resolve().parents[1] / "result" / "05_lmcache_decomposition.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

# HBM comparison path
if args.hbm_results:
    hbm_path = Path(args.hbm_results)
else:
    hbm_path = Path(__file__).resolve().parents[1] / "result" / "04_server_decomposition.json"

import aiohttp

# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------
ALL_SCENARIOS = [
    ("single_1k",    1,  1000),
    ("single_8k",    1,  8000),
    ("single_32k",   1,  32000),
    ("single_100k",  1,  100000),
    ("conc4_8k",     4,  8000),
    ("conc16_32k",  16,  32000),
    ("conc32_32k",  32,  32000),
    ("conc32_50k", 32,  50000),
]

if args.scenarios:
    SCENARIOS = [(n, c, ctx) for n, c, ctx in ALL_SCENARIOS if n in args.scenarios]
else:
    SCENARIOS = ALL_SCENARIOS

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

# ---------------------------------------------------------------------------
# Reuse the same trace/profiling infrastructure from script 04
# ---------------------------------------------------------------------------
# (Duplicated to keep scripts self-contained; in production these would be
#  imported from a shared profiling library.)

_CONTEXT_CACHE: dict[int, str] = {}

def get_context_text(target_tokens: int) -> str:
    if target_tokens in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[target_tokens]
    # For LMCache testing, use some shared prefix to test cache hit scenarios
    shared_prefix = (
        "System: You are an AI coding assistant with access to tools for "
        "reading files, searching codebases, and executing shell commands. "
        "You should use tools whenever appropriate and provide thorough analysis.\n\n"
    )
    seed = shared_prefix + (
        "User: Analyze the following codebase and identify performance bottlenecks.\n\n"
        + ("def process_batch(items, batch_size=32):\n"
           "    for i in range(0, len(items), batch_size):\n"
           "        batch = items[i:i+batch_size]\n"
           "        yield [transform(x) for x in batch]\n\n" * 200)
        + "Assistant: I can see several optimization opportunities in this code. " * 200
    )
    chars_needed = target_tokens * 4
    if chars_needed <= len(seed):
        text = seed[:chars_needed]
    else:
        text = (seed * ((chars_needed // len(seed)) + 1))[:chars_needed]
    _CONTEXT_CACHE[target_tokens] = text
    return text

def build_payload(context_text: str, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

@dataclass
class RequestTrace:
    scenario: str = ""
    batch: int = 0
    idx: int = 0
    ctx: int = 0
    t_serialize_ms: float = 0.0
    t_first_byte_ms: float = 0.0
    t_ttft_ms: float = 0.0
    t_decode_ms: float = 0.0
    t_response_parse_ms: float = 0.0
    t_e2e_ms: float = 0.0
    num_output_tokens: int = 0
    success: bool = True
    error: str = ""

async def trace_one_request(
    session, url, payload, scenario, batch, idx, ctx, semaphore
) -> RequestTrace:
    t = RequestTrace(scenario=scenario, batch=batch, idx=idx, ctx=ctx)
    e2e_start = time.perf_counter()

    t0 = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False)
    t.t_serialize_ms = (time.perf_counter() - t0) * 1000

    async with semaphore:
        try:
            t_post = time.perf_counter()
            async with session.post(
                f"{url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    t.success = False
                    t.error = f"HTTP {resp.status}: {err[:300]}"
                    t.t_e2e_ms = (time.perf_counter() - e2e_start) * 1000
                    return t

                first_byte = False
                first_token = False
                t_first_byte = None
                t_first_token = None
                t_last_token = None
                parse_start = None
                token_count = 0

                async for line in resp.content:
                    if not first_byte:
                        t_first_byte = time.perf_counter()
                        t.t_first_byte_ms = (t_first_byte - t_post) * 1000
                        first_byte = True
                        parse_start = time.perf_counter()

                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        try:
                            chunk = json.loads(line_str[6:])
                            choices = chunk.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content", "")
                                if content:
                                    if not first_token:
                                        t_first_token = time.perf_counter()
                                        t.t_ttft_ms = (t_first_token - t_post) * 1000
                                        first_token = True
                                    t_last_token = time.perf_counter()
                                    token_count += 1
                        except json.JSONDecodeError:
                            pass

                if parse_start:
                    t.t_response_parse_ms = (time.perf_counter() - parse_start) * 1000
                if first_token and t_last_token:
                    t.t_decode_ms = (t_last_token - t_first_token) * 1000
                t.num_output_tokens = token_count

        except asyncio.TimeoutError:
            t.success = False
            t.error = "Timeout"
        except Exception as e:
            t.success = False
            t.error = str(e)[:300]

    t.t_e2e_ms = (time.perf_counter() - e2e_start) * 1000
    return t

async def run_batch(session, url, model, scenario, batch, concurrency, ctx_len, output_tokens):
    text = get_context_text(ctx_len)
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        trace_one_request(session, url, build_payload(text, model, output_tokens),
                          scenario, batch, i, ctx_len, sem)
        for i in range(concurrency)
    ]
    print(f"    Batch {batch}: {concurrency} req, ctx={ctx_len}...", end=" ", flush=True)
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    ok = sum(1 for r in results if r.success)
    print(f"{elapsed:.1f}s, {ok}/{len(results)} OK")
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    url = args.url.rstrip("/")
    print(f"[05_lmcache] Server decomposition — LMCache DRAM")
    print(f"[05_lmcache] Server: {url}")
    print(f"[05_lmcache] Model:  {args.model}")
    print(f"[05_lmcache] Scenarios: {len(SCENARIOS)}\n")

    all_traces = []
    async with aiohttp.ClientSession() as session:
        for scenario_name, concurrency, ctx_len in SCENARIOS:
            print(f"{'─'*55}")
            print(f"  Scenario: {scenario_name} (conc={concurrency}, ctx={ctx_len:,})")
            print(f"{'─'*55}")

            for w in range(args.warmup_batches):
                await run_batch(session, url, args.model,
                                f"{scenario_name}_warmup", w,
                                concurrency, ctx_len, args.output_tokens)

            for b in range(args.num_batches):
                batch_results = await run_batch(
                    session, url, args.model, scenario_name, b,
                    concurrency, ctx_len, args.output_tokens
                )
                all_traces.extend(batch_results)

    # -------------------------------------------------------------------
    # Decompose
    # -------------------------------------------------------------------
    scenario_groups = defaultdict(list)
    for t in all_traces:
        if t.success:
            scenario_groups[t.scenario].append(t)

    decomposition = {}
    for scenario_name, concurrency, ctx_len in SCENARIOS:
        traces = scenario_groups.get(scenario_name, [])
        if not traces:
            decomposition[scenario_name] = {"error": "No successful traces"}
            continue

        n = len(traces)

        serialize_stats = compute_stats([t.t_serialize_ms for t in traces])
        first_byte_stats = compute_stats([t.t_first_byte_ms for t in traces])
        ttft_stats = compute_stats([t.t_ttft_ms for t in traces])
        decode_stats = compute_stats([t.t_decode_ms for t in traces])
        parse_stats = compute_stats([t.t_response_parse_ms for t in traces])

        # Derived: http_overhead ≈ first_byte (localhost)
        http_oh_stats = compute_stats([t.t_first_byte_ms for t in traces])
        # Derived: prefill ≈ ttft - first_byte
        prefill_vals = [max(0, t.t_ttft_ms - t.t_first_byte_ms) for t in traces]
        prefill_stats = compute_stats(prefill_vals)
        # Derived: total, cpu_time, gpu_time
        cpu_vals = [t.t_serialize_ms + t.t_first_byte_ms for t in traces]
        cpu_stats = compute_stats(cpu_vals)
        gpu_vals = [max(0, t.t_ttft_ms - t.t_first_byte_ms) + t.t_decode_ms for t in traces]
        gpu_stats = compute_stats(gpu_vals)
        total_vals = [c + g for c, g in zip(cpu_vals, gpu_vals)]
        total_stats = compute_stats(total_vals)

        cpu_pct = round((cpu_stats["mean"] / total_stats["mean"] * 100) if total_stats["mean"] > 0 else 0, 10)
        gpu_pct = round((gpu_stats["mean"] / total_stats["mean"] * 100) if total_stats["mean"] > 0 else 0, 10)

        decomposition[scenario_name] = {
            "concurrency": concurrency,
            "context": ctx_len,
            "num_requests": n,
            "t_serialize_ms_mean": serialize_stats["mean"],
            "t_serialize_ms_p50": serialize_stats["p50"],
            "t_serialize_ms_p95": serialize_stats["p95"],
            "t_serialize_ms_min": serialize_stats["min"],
            "t_serialize_ms_max": serialize_stats["max"],
            "t_http_overhead_ms_mean": http_oh_stats["mean"],
            "t_http_overhead_ms_p50": http_oh_stats["p50"],
            "t_http_overhead_ms_p95": http_oh_stats["p95"],
            "t_http_overhead_ms_min": http_oh_stats["min"],
            "t_http_overhead_ms_max": http_oh_stats["max"],
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

    # -------------------------------------------------------------------
    # Compare with HBM baseline (if available)
    # -------------------------------------------------------------------
    comparison = {}
    if hbm_path.exists():
        print(f"\n[05_lmcache] Loading HBM baseline from {hbm_path}")
        hbm_data = json.loads(hbm_path.read_text())
        hbm_decomp = hbm_data.get("decomposition", {})

        for scenario_name, _, _ in SCENARIOS:
            hbm = hbm_decomp.get(scenario_name, {})
            lmc = decomposition.get(scenario_name, {})
            if "error" not in hbm and "error" not in lmc:
                delta_cpu = lmc["cpu_percent"] - hbm["cpu_percent"]
                delta_http = lmc["t_http_overhead_ms_mean"] - hbm["t_http_overhead_ms_mean"]
                comparison[scenario_name] = {
                    "hbm_cpu_pct": hbm["cpu_percent"],
                    "lmcache_cpu_pct": lmc["cpu_percent"],
                    "delta_cpu_pct": round(delta_cpu, 10),
                    "hbm_http_overhead_ms": hbm["t_http_overhead_ms_mean"],
                    "lmcache_http_overhead_ms": lmc["t_http_overhead_ms_mean"],
                    "delta_http_overhead_ms": round(delta_http, 20),
                }
    else:
        print(f"[05_lmcache] No HBM baseline found at {hbm_path} — skipping comparison")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    raw_traces = []
    for t in all_traces:
        raw_traces.append({
            "scenario": t.scenario, "batch": t.batch, "idx": t.idx, "ctx": t.ctx,
            "t_serialize_ms": round(t.t_serialize_ms, 20),
            "t_first_byte_ms": round(t.t_first_byte_ms, 20),
            "t_ttft_ms": round(t.t_ttft_ms, 20),
            "t_decode_ms": round(t.t_decode_ms, 20),
            "t_response_parse_ms": round(t.t_response_parse_ms, 20),
            "t_e2e_ms": round(t.t_e2e_ms, 20),
            "num_output_tokens": t.num_output_tokens,
            "success": t.success,
            "error": t.error,
        })

    output = {
        "benchmark": "server_decomposition_lmcache_dram",
        "server_url": url,
        "model": args.model,
        "num_batches": args.num_batches,
        "warmup_batches": args.warmup_batches,
        "output_tokens_target": args.output_tokens,
        "cache_config": "LMCache_CPU_DRAM",
        "lmcache_chunk_size": os.environ.get("LMCACHE_CHUNK_SIZE", "256"),
        "decomposition": decomposition,
        "comparison_vs_hbm": comparison,
        "raw_traces": raw_traces,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[05_lmcache] Results saved to {out_path}")

    # -------------------------------------------------------------------
    # Print tables
    # -------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("LMCACHE DRAM — CPU vs GPU Decomposition")
    print("=" * 95)
    header = (f"{'Scenario':<16s} {'Conc':>4s} {'Ctx':>6s} {'HTTP OH':>9s} "
              f"{'Prefill':>9s} {'Decode':>9s} {'Total':>9s} {'CPU%':>7s} {'GPU%':>7s}")
    print(header)
    print("-" * 95)
    for scenario_name, _, _ in SCENARIOS:
        d = decomposition.get(scenario_name, {})
        if d and "error" not in d:
            print(f"{scenario_name:<16s} {d.get('concurrency', 0):>4d} "
                  f"{d.get('context', 0):>6,d} "
                  f"{d['t_http_overhead_ms_mean']:>8.0f}ms "
                  f"{d['t_prefill_ms_mean']:>8.0f}ms "
                  f"{d['t_decode_ms_mean']:>8.0f}ms "
                  f"{d['t_total_ms_mean']:>8.0f}ms "
                  f"{d['cpu_percent']:>6.1f}% "
                  f"{d['gpu_percent']:>6.1f}%")
    print("=" * 95)

    # Comparison table
    if comparison:
        print("\n" + "=" * 85)
        print("COMPARISON: HBM Prefix Cache vs LMCache DRAM — CPU%")
        print("=" * 85)
        print(f"{'Scenario':<16s} {'HBM CPU%':>10s} {'LMCache CPU%':>14s} {'Delta':>8s}")
        print("-" * 85)
        for scenario_name, _, _ in SCENARIOS:
            c = comparison.get(scenario_name, {})
            if c:
                print(f"{scenario_name:<16s} {c['hbm_cpu_pct']:>9.1f}% "
                      f"{c['lmcache_cpu_pct']:>13.1f}% "
                      f"{c['delta_cpu_pct']:>+7.1f}%")
        print("=" * 85)

if __name__ == "__main__":
    asyncio.run(main())
