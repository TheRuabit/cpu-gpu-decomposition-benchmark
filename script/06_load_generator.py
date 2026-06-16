#!/usr/bin/env python3
"""
Concurrent Load Generator
==========================
Drives concurrent agentic LLM requests against a vLLM server using asyncio.
Supports the full benchmark matrix: concurrency 1–32, context 1k–100k tokens.

Features:
  - Configurable warmup + measurement phases
  - Real-time progress reporting
  - Per-request and aggregate timing collection
  - Prometheus metrics capture (vLLM /metrics endpoint)
  - Throttled ramp-up to avoid overwhelming the scheduler instantly

Usage:
    # Single scenario
    python script/06_load_generator.py --url http://localhost:8000 \
        --concurrency 4 --context 8000 --duration 60

    # Full matrix
    python script/06_load_generator.py --url http://localhost:8000 --matrix

    # With GPU monitoring (runs nvidia-smi in background)
    python script/06_load_generator.py --url http://localhost:8000 --matrix --gpu-monitor

Output:
    result/06_load_generator.json
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Concurrent load generator for vLLM benchmarks")
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server base URL")
parser.add_argument("--model", default="models/Qwen3-30B-A3B",
                    help="Model name for API")
parser.add_argument("--concurrency", type=int, default=4,
                    help="Number of concurrent requests")
parser.add_argument("--context", type=int, default=8000,
                    help="Target context length in tokens")
parser.add_argument("--output-tokens", type=int, default=512,
                    help="Max output tokens per request")
parser.add_argument("--duration", type=float, default=0,
                    help="Run duration in seconds (0 = run fixed number of batches)")
parser.add_argument("--num-batches", type=int, default=3,
                    help="Number of batches (when duration=0)")
parser.add_argument("--warmup-batches", type=int, default=1)
parser.add_argument("--ramp-up-seconds", type=float, default=2.0,
                    help="Gradually ramp up concurrency over this many seconds")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
parser.add_argument("--matrix", action="store_true",
                    help="Run full test matrix")
parser.add_argument("--gpu-monitor", action="store_true",
                    help="Launch GPU monitor subprocess during benchmark")
parser.add_argument("--metrics-interval", type=float, default=2.0,
                    help="How often to sample vLLM metrics (seconds)")
args = parser.parse_args()

if args.output:
    out_path = Path(args.output)
else:
    out_path = Path(__file__).resolve().parents[1] / "result" / "06_load_generator.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

import aiohttp

# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------
MATRIX = [
    ("single_1k",    1,  1000),
    ("single_8k",    1,  8000),
    ("single_32k",   1,  32000),
    ("single_50k",  1,  50000),
    ("conc4_8k",     4,  8000),
    ("conc16_32k",  16,  32000),
    ("conc32_32k",  32,  32000),
    ("conc32_50k", 32,  50000),
]

# ---------------------------------------------------------------------------
# Context generator
# ---------------------------------------------------------------------------
_CONTEXT_CACHE = {}

def get_context_text(target_tokens: int) -> str:
    if target_tokens in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[target_tokens]
    seed = (
        "System: You are an AI coding assistant with tools for reading files, "
        "searching code, and executing shell commands. Be thorough and precise.\n\n"
        "User: Please analyze this codebase for performance issues and suggest "
        "optimizations. Consider CPU, memory, and IO bottlenecks.\n\n"
        + ("def process_pipeline(data):\n"
           "    cleaned = clean(data)\n"
           "    transformed = transform(cleaned)\n"
           "    validated = validate(transformed)\n"
           "    return validated\n\n" * 200)
        + "Assistant: I'll analyze the pipeline step by step. " * 200
    )
    chars_needed = target_tokens * 4
    if chars_needed <= len(seed):
        text = seed[:chars_needed]
    else:
        text = (seed * ((chars_needed // len(seed)) + 1))[:chars_needed]
    _CONTEXT_CACHE[target_tokens] = text
    return text

def build_payload(ctx_text: str, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ctx_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

# ---------------------------------------------------------------------------
# Metrics sampler
# ---------------------------------------------------------------------------
async def sample_metrics(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch vLLM Prometheus metrics."""
    try:
        async with session.get(f"{url}/metrics", timeout=aiohttp.ClientTimeout(total=3)) as resp:
            if resp.status == 200:
                text = await resp.text()
                metrics = {}
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if "{" in line:
                        name, rest = line.split("{", 1)
                        if "} " in rest:
                            _, val = rest.rsplit("} ", 1)
                        else:
                            continue
                    else:
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        name, val = parts[0], parts[1]
                    try:
                        metrics[name] = float(val)
                    except ValueError:
                        pass
                return metrics
    except Exception:
        pass
    return {}

# ---------------------------------------------------------------------------
# GPU monitor CSV parser
# ---------------------------------------------------------------------------
def parse_gpu_csv(csv_path: Path) -> list[dict]:
    """Parse GPU monitor CSV into list of samples with unix timestamps."""
    samples = []
    if not csv_path.exists():
        return samples
    try:
        with open(csv_path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    sample = {
                        "unix_timestamp": float(row.get("unix_timestamp", 0)),
                        "gpu_index": int(row.get("gpu_index", 0)),
                        "utilization_gpu_pct": _parse_float(row.get("utilization_gpu_pct")),
                        "utilization_memory_pct": _parse_float(row.get("utilization_memory_pct")),
                        "memory_used_mib": _parse_float(row.get("memory_used_mib")),
                        "memory_total_mib": _parse_float(row.get("memory_total_mib")),
                        "temperature_gpu_c": _parse_float(row.get("temperature_gpu_c")),
                        "power_draw_w": _parse_float(row.get("power_draw_w")),
                        "clocks_sm_mhz": _parse_float(row.get("clocks_sm_mhz")),
                        "clocks_memory_mhz": _parse_float(row.get("clocks_memory_mhz")),
                    }
                    samples.append(sample)
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"    [WARN] Failed to parse GPU CSV: {e}")
    return samples


def _parse_float(val: Optional[str]) -> Optional[float]:
    """Parse a float value, returning None for empty/invalid."""
    if val is None:
        return None
    val = val.strip()
    if not val or val.lower() in ("[not supported]", "[unknown]", "n/a", ""):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def correlate_gpu_to_scenarios(
    gpu_samples: list[dict],
    scenario_times: dict[str, tuple[float, float]],
) -> dict[str, dict]:
    """Correlate GPU samples to benchmark scenarios using timestamps.

    Args:
        gpu_samples: List of GPU sample dicts with 'unix_timestamp' and 'gpu_index'.
        scenario_times: Dict mapping scenario_name -> (start_unix, end_unix).

    Returns:
        Dict mapping scenario_name -> gpu_stats dict with per-GPU averages.
    """
    result = {}
    for scenario_name, (t_start, t_end) in scenario_times.items():
        # Collect samples that fall within this scenario's time window
        window_samples = [
            s for s in gpu_samples
            if t_start <= s["unix_timestamp"] <= t_end
        ]
        if not window_samples:
            result[scenario_name] = {"error": "No GPU samples in window"}
            continue

        # Group by GPU index
        by_gpu = defaultdict(list)
        for s in window_samples:
            by_gpu[s["gpu_index"]].append(s)

        gpu_stats = {}
        all_utils = []
        for gpu_idx, samples in sorted(by_gpu.items()):
            utils = [s["utilization_gpu_pct"] for s in samples
                     if s["utilization_gpu_pct"] is not None]
            mem_utils = [s["utilization_memory_pct"] for s in samples
                        if s["utilization_memory_pct"] is not None]
            mem_used = [s["memory_used_mib"] for s in samples
                       if s["memory_used_mib"] is not None]
            temps = [s["temperature_gpu_c"] for s in samples
                    if s["temperature_gpu_c"] is not None]
            powers = [s["power_draw_w"] for s in samples
                     if s["power_draw_w"] is not None]

            gpu_stats[f"gpu{gpu_idx}"] = {
                "num_samples": len(samples),
                "avg_utilization_pct": round(sum(utils) / len(utils), 10) if utils else None,
                "peak_utilization_pct": round(max(utils), 10) if utils else None,
                "avg_memory_utilization_pct": round(sum(mem_utils) / len(mem_utils), 10) if mem_utils else None,
                "avg_memory_used_mib": round(sum(mem_used) / len(mem_used), 20) if mem_used else None,
                "avg_temperature_c": round(sum(temps) / len(temps), 20) if temps else None,
                "avg_power_w": round(sum(powers) / len(powers), 20) if powers else None,
            }
            all_utils.extend(utils)

        gpu_stats["overall_avg_utilization_pct"] = (
            round(sum(all_utils) / len(all_utils), 10) if all_utils else None
        )
        result[scenario_name] = gpu_stats

    return result


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
# Worker
# ---------------------------------------------------------------------------
@dataclass
class ReqResult:
    idx: int = 0
    ttft_ms: float = 0.0
    decode_ms: float = 0.0
    e2e_ms: float = 0.0
    output_tokens: int = 0
    success: bool = True
    error: str = ""

async def worker(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    idx: int,
    semaphore: asyncio.Semaphore,
) -> ReqResult:
    """Send one streaming request and return timing."""
    r = ReqResult(idx=idx)
    body = json.dumps(payload, ensure_ascii=False)
    t_start = time.perf_counter()

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
                    r.success = False
                    r.error = f"HTTP {resp.status}"
                    r.e2e_ms = (time.perf_counter() - t_start) * 1000
                    return r

                first_token = False
                t_first_token = None
                t_last_token = None

                async for line in resp.content:
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
                                        r.ttft_ms = (t_first_token - t_post) * 1000
                                        first_token = True
                                    t_last_token = time.perf_counter()
                                    r.output_tokens += 1
                        except json.JSONDecodeError:
                            pass

                if first_token and t_last_token:
                    r.decode_ms = (t_last_token - t_first_token) * 1000

        except asyncio.TimeoutError:
            r.success = False
            r.error = "Timeout"
        except Exception as e:
            r.success = False
            r.error = str(e)[:300]

    r.e2e_ms = (time.perf_counter() - t_start) * 1000
    return r

# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------
async def run_scenario(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    scenario_name: str,
    concurrency: int,
    ctx_len: int,
    output_tokens: int,
    num_batches: int,
    warmup_batches: int,
    ramp_up_s: float,
    metrics_interval: float,
    collect_metrics: bool,
) -> dict:
    """Run warmup + measurement batches for one scenario."""
    ctx_text = get_context_text(ctx_len)
    payload = build_payload(ctx_text, model, output_tokens)
    sem = asyncio.Semaphore(concurrency)

    # Warmup
    for w in range(warmup_batches):
        tasks = [worker(session, url, payload, i, sem) for i in range(concurrency)]
        await asyncio.gather(*tasks)

    # Metrics sampling task
    metrics_samples = []
    metrics_task = None

    async def metrics_loop():
        while True:
            m = await sample_metrics(session, url)
            if m:
                metrics_samples.append({
                    "timestamp": time.time(),
                    "metrics": m,
                })
            await asyncio.sleep(metrics_interval)

    if collect_metrics:
        metrics_task = asyncio.create_task(metrics_loop())

    # Measurement batches
    all_results = []
    ramp_delay = ramp_up_s / concurrency if concurrency > 1 and ramp_up_s > 0 else 0

    t_scenario_start = time.time()  # wall-clock for GPU correlation
    for b in range(num_batches):
        batch_tasks = []
        for i in range(concurrency):
            if ramp_delay > 0:
                await asyncio.sleep(ramp_delay)
            batch_tasks.append(worker(session, url, payload, i, sem))

        print(f"    Batch {b+1}/{num_batches}: {concurrency} req, ctx={ctx_len}...",
              end=" ", flush=True)
        t0 = time.perf_counter()
        results = await asyncio.gather(*batch_tasks)
        elapsed = time.perf_counter() - t0
        ok = sum(1 for r in results if r.success)
        print(f"{elapsed:.1f}s, {ok}/{len(results)} OK")
        all_results.extend(results)
    t_scenario_end = time.time()  # wall-clock for GPU correlation

    if metrics_task:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass

    # Aggregate
    successful = [r for r in all_results if r.success]
    if not successful:
        return {"error": "No successful requests", "total_requests": len(all_results),
                "start_time": t_scenario_start, "end_time": t_scenario_end,
                "metrics_samples": metrics_samples}

    n = len(successful)
    ttft_stats = compute_stats([r.ttft_ms for r in successful])
    decode_stats = compute_stats([r.decode_ms for r in successful])
    e2e_stats = compute_stats([r.e2e_ms for r in successful])

    agg = {
        "scenario": scenario_name,
        "concurrency": concurrency,
        "context_tokens": ctx_len,
        "total_requests": len(all_results),
        "successful": n,
        "failed": len(all_results) - n,
        "t_ttft_ms_mean": ttft_stats["mean"],
        "t_ttft_ms_p50": ttft_stats["p50"],
        "t_ttft_ms_p95": ttft_stats["p95"],
        "t_ttft_ms_min": ttft_stats["min"],
        "t_ttft_ms_max": ttft_stats["max"],
        "t_decode_ms_mean": decode_stats["mean"],
        "t_decode_ms_p50": decode_stats["p50"],
        "t_decode_ms_p95": decode_stats["p95"],
        "t_decode_ms_min": decode_stats["min"],
        "t_decode_ms_max": decode_stats["max"],
        "t_e2e_ms_mean": e2e_stats["mean"],
        "t_e2e_ms_p50": e2e_stats["p50"],
        "t_e2e_ms_p95": e2e_stats["p95"],
        "t_e2e_ms_min": e2e_stats["min"],
        "t_e2e_ms_max": e2e_stats["max"],
        "total_output_tokens": sum(r.output_tokens for r in successful),
        "avg_output_tokens": round(sum(r.output_tokens for r in successful) / n, 10),
        "throughput_req_per_s": round(n / (e2e_stats["mean"] / 1000 / n), 10)
            if n > 0 else 0,
        "start_time": t_scenario_start,
        "end_time": t_scenario_end,
        "metrics_samples": metrics_samples,
    }
    return agg

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    url = args.url.rstrip("/")
    print(f"[06_loadgen] Load Generator")
    print(f"[06_loadgen] Server: {url}")

    if args.matrix:
        runs = MATRIX
        print(f"[06_loadgen] Full matrix: {len(runs)} scenarios")
    else:
        runs = [(f"custom_c{args.concurrency}_ctx{args.context}",
                 args.concurrency, args.context)]

    # GPU monitor (if enabled)
    gpu_monitor_proc = None
    if args.gpu_monitor:
        monitor_out = out_path.parent / "06_gpu_monitor_log.csv"
        print(f"[06_loadgen] Starting GPU monitor → {monitor_out}")
        gpu_monitor_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve().parent / "07_gpu_monitor.py"),
            "--interval", "0.5",
            "--output", str(monitor_out),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.sleep(1.0)  # Let monitor start

    results = {}
    async with aiohttp.ClientSession() as session:
        for scenario_name, concurrency, ctx_len in runs:
            print(f"\n{'─'*55}")
            print(f"  Scenario: {scenario_name}  conc={concurrency}  ctx={ctx_len:,}")
            print(f"{'─'*55}")

            agg = await run_scenario(
                session, url, args.model,
                scenario_name, concurrency, ctx_len,
                args.output_tokens,
                args.num_batches, args.warmup_batches,
                args.ramp_up_seconds, args.metrics_interval,
                collect_metrics=True,
            )
            results[scenario_name] = agg

            if "error" not in agg:
                print(f"    → avg TTFT={agg['t_ttft_ms_mean']:.0f}ms  "
                      f"avg Decode={agg['t_decode_ms_mean']:.0f}ms  "
                      f"avg E2E={agg['t_e2e_ms_mean']:.0f}ms  "
                      f"p95={agg['t_e2e_ms_p95']:.0f}ms")

    # Stop GPU monitor
    gpu_stats_by_scenario = {}
    if gpu_monitor_proc:
        gpu_monitor_proc.terminate()
        try:
            await asyncio.wait_for(gpu_monitor_proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            gpu_monitor_proc.kill()
        print("[06_loadgen] GPU monitor stopped")

        # Parse GPU CSV and correlate samples to scenarios
        gpu_csv_path = out_path.parent / "06_gpu_monitor_log.csv"
        if gpu_csv_path.exists():
            print(f"[06_loadgen] Parsing GPU monitor data: {gpu_csv_path}")
            gpu_samples = parse_gpu_csv(gpu_csv_path)
            print(f"[06_loadgen]   {len(gpu_samples)} GPU samples loaded")

            # Build scenario time windows from results
            scenario_times = {}
            for scenario_name, r in results.items():
                t_start = r.get("start_time")
                t_end = r.get("end_time")
                if t_start and t_end:
                    scenario_times[scenario_name] = (t_start, t_end)

            if scenario_times and gpu_samples:
                gpu_stats_by_scenario = correlate_gpu_to_scenarios(
                    gpu_samples, scenario_times
                )
                # Embed GPU stats into each scenario result
                for scenario_name, gpu_stats in gpu_stats_by_scenario.items():
                    if scenario_name in results:
                        results[scenario_name]["gpu_stats"] = gpu_stats
                        if "overall_avg_utilization_pct" in gpu_stats:
                            print(f"    {scenario_name}: GPU avg util = "
                                  f"{gpu_stats['overall_avg_utilization_pct']}%")
        else:
            print("[06_loadgen] WARNING: GPU monitor CSV not found")

    # Save
    output = {
        "benchmark": "load_generator",
        "server_url": url,
        "model": args.model,
        "num_batches": args.num_batches,
        "warmup_batches": args.warmup_batches,
        "output_tokens_target": args.output_tokens,
        "ramp_up_seconds": args.ramp_up_seconds,
        "scenario_results": results,
        "gpu_monitor_enabled": args.gpu_monitor,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[06_loadgen] Results saved to {out_path}")

    # Summary table
    has_gpu = any(r.get("gpu_stats") for r in results.values())
    print("\n" + "=" * 90)
    print("LOAD TEST SUMMARY")
    print("=" * 90)
    if has_gpu:
        header = (f"{'Scenario':<16s} {'Conc':>4s} {'Ctx':>6s} {'N':>4s} "
                  f"{'TTFT':>8s} {'Decode':>8s} {'E2E':>8s} {'p95':>8s} {'tput':>8s} {'GPU%':>6s}")
    else:
        header = (f"{'Scenario':<16s} {'Conc':>4s} {'Ctx':>6s} {'N':>4s} "
                  f"{'TTFT':>8s} {'Decode':>8s} {'E2E':>8s} {'p95':>8s} {'tput':>8s}")
    print(header)
    print("-" * 90)
    for scenario_name, _, _ in runs:
        r = results.get(scenario_name, {})
        if r and "error" not in r:
            gpu_str = ""
            if has_gpu:
                gs = r.get("gpu_stats", {})
                gpu_avg = gs.get("overall_avg_utilization_pct", "-")
                gpu_str = f"{gpu_avg:>5.0f}%" if isinstance(gpu_avg, (int, float)) else f"{str(gpu_avg):>6s}"
            print(f"{scenario_name:<16s} {r['concurrency']:>4d} "
                  f"{r['context_tokens']:>6,d} {r['successful']:>4d} "
                  f"{r['t_ttft_ms_mean']:>7.0f}ms {r['t_decode_ms_mean']:>7.0f}ms "
                  f"{r['t_e2e_ms_mean']:>7.0f}ms {r['t_e2e_ms_p95']:>7.0f}ms "
                  f"{r['throughput_req_per_s']:>7.2f}/s"
                  + (f" {gpu_str}" if has_gpu else ""))
    print("=" * 90)

if __name__ == "__main__":
    asyncio.run(main())
