#!/usr/bin/env python3
"""
Server-Side Request Decomposition — HBM Prefix Cache
=====================================================
Runs the full benchmark matrix against a vLLM server configured with HBM prefix
cache enabled (--enable-prefix-caching). Collects client-side timing AND server
Prometheus metrics to decompose each request into:

  t_serialize        — client JSON serialization
  t_http_overhead    — tokenization + scheduling + queue wait + KV cache lookup
  t_server_prefill   — GPU attention over input tokens
  t_decode           — GPU autoregressive generation
  t_response_parse   — client SSE parsing

The decomposition uses:
  - Client:  measured t_serialize, t_first_byte (≈ t_http_overhead on localhost),
             t_decode, t_response_parse
  - Server:  vLLM /metrics → vllm:time_to_first_token_seconds (histogram),
             vllm:time_per_output_token_seconds (histogram),
             vllm:e2e_request_latency_seconds
  - t_prefill is estimated as: TTFT(server) - t_http_overhead

Reference: BLOG.md Section 3.1 "The CPU-GPU Split"

Prerequisites:
  1. vLLM server running with HBM prefix cache:
     vllm serve /path/to/model --enable-prefix-caching --gpu-memory-utilization 0.85

Usage:
    python script/04_server_decomposition.py --url http://localhost:8000

Output:
    result/04_server_decomposition.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Server decomposition benchmark — HBM prefix cache"
)
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server URL")
parser.add_argument("--model", default="./models/Qwen3-30B-A3B",
                    help="Model name for API requests")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
parser.add_argument("--output-tokens", type=int, default=64,
                    help="Max output tokens per request")
parser.add_argument("--num-batches", type=int, default=3,
                    help="Number of measurement batches per scenario")
parser.add_argument("--warmup-batches", type=int, default=1,
                    help="Warmup batches before measurement")
parser.add_argument("--scenarios", nargs="+",
                    default=None,
                    help="Specific scenarios to run (default: all 8)")
parser.add_argument("--fetch-metrics", action="store_true", default=True,
                    help="Fetch vLLM Prometheus metrics before/after each scenario")
parser.add_argument("--metrics-port", type=int, default=8000,
                    help="Port for vLLM metrics endpoint (usually same as API port)")
args = parser.parse_args()

if args.output:
    out_path = Path(args.output)
else:
    out_path = Path(__file__).resolve().parents[1] / "result" / "04_server_decomposition.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports (after CLI so --help is fast)
# ---------------------------------------------------------------------------
import aiohttp

# ---------------------------------------------------------------------------
# Test matrix (matching blog Section 2.3)
# ---------------------------------------------------------------------------
ALL_SCENARIOS = [
    ("single_1k",    1,  1000),
    ("single_8k",    1,  8000),
    ("single_32k",   1,  32000),
    ("single_50k",  1,  50000),
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
# Context text generator
# ---------------------------------------------------------------------------
_CONTEXT_CACHE: dict[int, str] = {}

def get_context_text(target_tokens: int) -> str:
    if target_tokens in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[target_tokens]
    seed = (
        "System: You are a helpful AI assistant with tool access.\n\n"
        "User: Analyze this codebase for performance issues.\n\n"
        + ("def process(items):\n    return [transform(x) for x in items]\n\n" * 200)
        + "Assistant: The key bottleneck is the sequential processing loop. " * 200
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
# vLLM Metrics Fetcher
# ---------------------------------------------------------------------------
async def fetch_vllm_metrics(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch and parse vLLM Prometheus metrics into a dict of {metric_name: value}."""
    try:
        async with session.get(f"{url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return {"_error": f"HTTP {resp.status}"}
            text = await resp.text()
    except Exception as e:
        return {"_error": str(e)}

    metrics = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        # Parse: metric_name{labels} value
        # Or: metric_name value  (no labels)
        if "{" in line:
            name, rest = line.split("{", 1)
            # Extract value after the closing brace
            if "} " in rest:
                _, value_str = rest.rsplit("} ", 1)
            else:
                continue
        else:
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                value_str = parts[1]
            else:
                continue
        try:
            metrics[name] = float(value_str)
        except ValueError:
            pass
    return metrics

def summarize_histogram_metrics(before: dict, after: dict, metric_name: str) -> dict:
    """Extract _sum and _count delta for a histogram metric."""
    sum_key = f"{metric_name}_sum"
    count_key = f"{metric_name}_count"
    result = {}
    if sum_key in before and sum_key in after:
        result["delta_sum"] = round(after[sum_key] - before[sum_key], 20)
        result["delta_count"] = round(after[count_key] - before[count_key], 20)
        if result["delta_count"] > 0:
            result["avg_seconds"] = round(result["delta_sum"] / result["delta_count"], 20)
    return result

# ---------------------------------------------------------------------------
# Single request profiler
# ---------------------------------------------------------------------------
@dataclass
class RequestTrace:
    scenario: str = ""
    batch: int = 0
    idx: int = 0
    ctx: int = 0
    t_serialize_ms: float = 0.0
    t_first_byte_ms: float = 0.0    # HTTP POST → first SSE byte
    t_ttft_ms: float = 0.0          # HTTP POST → first content token
    t_decode_ms: float = 0.0
    t_response_parse_ms: float = 0.0
    t_e2e_ms: float = 0.0
    num_output_tokens: int = 0
    success: bool = True
    error: str = ""

async def trace_one_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    scenario: str,
    batch: int,
    idx: int,
    ctx: int,
    semaphore: asyncio.Semaphore,
) -> RequestTrace:
    t = RequestTrace(scenario=scenario, batch=batch, idx=idx, ctx=ctx)
    e2e_start = time.perf_counter()

    # serialize
    t0 = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False)
    t.t_serialize_ms = (time.perf_counter() - t0) * 1000
    
    t_post = time.perf_counter()

    async with semaphore:
        try:
            t_first_byte = None
            
            async with session.post(
                f"{url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                t_first_byte = time.perf_counter()
                t.t_first_byte_ms = (t_first_byte - t_post) * 1000
                if resp.status != 200:
                    err = await resp.text()
                    t.success = False
                    t.error = f"HTTP {resp.status}: {err[:300]}"
                    t.t_e2e_ms = (time.perf_counter() - e2e_start) * 1000
                    return t
                    

                first_byte = False
                first_token = False
                # t_first_byte = None
                t_first_token = None
                t_last_token = None
                parse_start = None
                token_count = 0
                


                async for line in resp.content:
                    if not first_byte:
                        
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
    ctx_len: int,
    output_tokens: int,
) -> list[RequestTrace]:
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
    print(f"[04_hbm] Server decomposition — HBM Prefix Cache")
    print(f"[04_hbm] Server: {url}")
    print(f"[04_hbm] Model:  {args.model}")
    print(f"[04_hbm] Scenarios: {len(SCENARIOS)}")
    print(f"[04_hbm] Batches: {args.num_batches} (+{args.warmup_batches} warmup)\n")

    all_traces: list[RequestTrace] = []
    scenario_metrics: dict[str, dict] = {}

    async with aiohttp.ClientSession() as session:
        for scenario_name, concurrency, ctx_len in SCENARIOS:
            print(f"{'─'*55}")
            print(f"  Scenario: {scenario_name} (conc={concurrency}, ctx={ctx_len:,})")
            print(f"{'─'*55}")

            # Fetch pre-scenario metrics
            metrics_before = {}
            if args.fetch_metrics:
                metrics_before = await fetch_vllm_metrics(session, url)

            # Warmup
            for w in range(args.warmup_batches):
                await run_batch(session, url, args.model,
                                f"{scenario_name}_warmup", w,
                                concurrency, ctx_len, args.output_tokens)

            # Measurement
            scenario_traces = []
            for b in range(args.num_batches):
                batch_results = await run_batch(
                    session, url, args.model,
                    scenario_name, b, concurrency, ctx_len, args.output_tokens
                )
                scenario_traces.extend(batch_results)

            all_traces.extend(scenario_traces)

            # Fetch post-scenario metrics
            metrics_after = {}
            if args.fetch_metrics:
                await asyncio.sleep(0.5)  # let metrics settle
                metrics_after = await fetch_vllm_metrics(session, url)

            # Compute server-side metrics deltas
            ttft_stats = summarize_histogram_metrics(
                metrics_before, metrics_after, "vllm:time_to_first_token_seconds"
            )
            tpot_stats = summarize_histogram_metrics(
                metrics_before, metrics_after, "vllm:time_per_output_token_seconds"
            )
            e2e_stats = summarize_histogram_metrics(
                metrics_before, metrics_after, "vllm:e2e_request_latency_seconds"
            )

            scenario_metrics[scenario_name] = {
                "concurrency": concurrency,
                "context_tokens": ctx_len,
                "ttft_server": ttft_stats,
                "tpot_server": tpot_stats,
                "e2e_server": e2e_stats,
            }

    # -------------------------------------------------------------------
    # Decompose & aggregate
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

        # Compute per-metric stats from per-request values (20dp for times)
        serialize_stats = compute_stats([t.t_serialize_ms for t in traces])
        first_byte_stats = compute_stats([t.t_first_byte_ms for t in traces])
        ttft_stats = compute_stats([t.t_ttft_ms for t in traces])
        decode_stats = compute_stats([t.t_decode_ms for t in traces])
        parse_stats = compute_stats([t.t_response_parse_ms for t in traces])
        e2e_stats = compute_stats([t.t_e2e_ms for t in traces])

        # Derived: t_http_overhead ≈ t_first_byte (on localhost, network ~0)
        http_oh_stats = compute_stats([t.t_first_byte_ms for t in traces])
        # Derived: t_prefill ≈ TTFT - first_byte (first token after CPU overhead)
        prefill_vals = [max(0, t.t_ttft_ms - t.t_first_byte_ms) for t in traces]
        prefill_stats = compute_stats(prefill_vals)
        # Derived: total = cpu_time + gpu_time
        cpu_vals = [t.t_serialize_ms + t.t_first_byte_ms for t in traces]
        cpu_stats = compute_stats(cpu_vals)
        gpu_vals = [max(0, t.t_ttft_ms - t.t_first_byte_ms) + t.t_decode_ms for t in traces]
        gpu_stats = compute_stats(gpu_vals)
        total_vals = [c + g for c, g in zip(cpu_vals, gpu_vals)]
        total_stats = compute_stats(total_vals)

        # Percentages from mean values (10dp)
        cpu_pct = round((cpu_stats["mean"] / total_stats["mean"] * 100) if total_stats["mean"] > 0 else 0, 10)
        gpu_pct = round((gpu_stats["mean"] / total_stats["mean"] * 100) if total_stats["mean"] > 0 else 0, 10)

        decomposition[scenario_name] = {
            "concurrency": concurrency,
            "context": ctx_len,
            "num_requests": n,
            # Serialize
            "t_serialize_ms_mean": serialize_stats["mean"],
            "t_serialize_ms_p50": serialize_stats["p50"],
            "t_serialize_ms_p95": serialize_stats["p95"],
            "t_serialize_ms_min": serialize_stats["min"],
            "t_serialize_ms_max": serialize_stats["max"],
            # HTTP overhead
            "t_http_overhead_ms_mean": http_oh_stats["mean"],
            "t_http_overhead_ms_p50": http_oh_stats["p50"],
            "t_http_overhead_ms_p95": http_oh_stats["p95"],
            "t_http_overhead_ms_min": http_oh_stats["min"],
            "t_http_overhead_ms_max": http_oh_stats["max"],
            # Prefill
            "t_prefill_ms_mean": prefill_stats["mean"],
            "t_prefill_ms_p50": prefill_stats["p50"],
            "t_prefill_ms_p95": prefill_stats["p95"],
            "t_prefill_ms_min": prefill_stats["min"],
            "t_prefill_ms_max": prefill_stats["max"],
            # Decode
            "t_decode_ms_mean": decode_stats["mean"],
            "t_decode_ms_p50": decode_stats["p50"],
            "t_decode_ms_p95": decode_stats["p95"],
            "t_decode_ms_min": decode_stats["min"],
            "t_decode_ms_max": decode_stats["max"],
            # Response parse
            "t_response_parse_ms_mean": parse_stats["mean"],
            "t_response_parse_ms_p50": parse_stats["p50"],
            "t_response_parse_ms_p95": parse_stats["p95"],
            "t_response_parse_ms_min": parse_stats["min"],
            "t_response_parse_ms_max": parse_stats["max"],
            # Total
            "t_total_ms_mean": total_stats["mean"],
            "t_total_ms_p50": total_stats["p50"],
            "t_total_ms_p95": total_stats["p95"],
            "t_total_ms_min": total_stats["min"],
            "t_total_ms_max": total_stats["max"],
            # CPU time
            "cpu_time_ms_mean": cpu_stats["mean"],
            "cpu_time_ms_p50": cpu_stats["p50"],
            "cpu_time_ms_p95": cpu_stats["p95"],
            "cpu_time_ms_min": cpu_stats["min"],
            "cpu_time_ms_max": cpu_stats["max"],
            # GPU time
            "gpu_time_ms_mean": gpu_stats["mean"],
            "gpu_time_ms_p50": gpu_stats["p50"],
            "gpu_time_ms_p95": gpu_stats["p95"],
            "gpu_time_ms_min": gpu_stats["min"],
            "gpu_time_ms_max": gpu_stats["max"],
            # Percentages (single values at 10dp)
            "cpu_percent": cpu_pct,
            "gpu_percent": gpu_pct,
        }

        print(f"\n  {scenario_name}: CPU={cpu_pct:.1f}% GPU={gpu_pct:.1f}%  "
              f"(HTTP OH={http_oh_stats['mean']:.0f}ms, Prefill={prefill_stats['mean']:.0f}ms, "
              f"Decode={decode_stats['mean']:.0f}ms)")

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
        "benchmark": "server_decomposition_hbm_prefix_cache",
        "server_url": url,
        "model": args.model,
        "num_batches": args.num_batches,
        "warmup_batches": args.warmup_batches,
        "output_tokens_target": args.output_tokens,
        "cache_config": "HBM_prefix_cache",
        "decomposition": decomposition,
        "server_metrics": scenario_metrics,
        "raw_traces": raw_traces,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[04_hbm] Results saved to {out_path}")

    # -------------------------------------------------------------------
    # Comparison table (matching blog format)
    # -------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("HBM PREFIX CACHE — CPU vs GPU Decomposition")
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

if __name__ == "__main__":
    asyncio.run(main())
