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
    ("single_100k",  1,  100000),
    ("conc4_8k",     4,  8000),
    ("conc16_32k",  16,  32000),
    ("conc32_32k",  32,  32000),
    ("conc32_100k", 32,  100000),
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

    if metrics_task:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass

    # Aggregate
    successful = [r for r in all_results if r.success]
    if not successful:
        return {"error": "No successful requests", "total_requests": len(all_results)}

    n = len(successful)
    agg = {
        "scenario": scenario_name,
        "concurrency": concurrency,
        "context_tokens": ctx_len,
        "total_requests": len(all_results),
        "successful": n,
        "failed": len(all_results) - n,
        "avg_ttft_ms": round(sum(r.ttft_ms for r in successful) / n, 2),
        "avg_decode_ms": round(sum(r.decode_ms for r in successful) / n, 2),
        "avg_e2e_ms": round(sum(r.e2e_ms for r in successful) / n, 2),
        "min_e2e_ms": round(min(r.e2e_ms for r in successful), 2),
        "max_e2e_ms": round(max(r.e2e_ms for r in successful), 2),
        "p50_e2e_ms": round(sorted(r.e2e_ms for r in successful)[n // 2], 2),
        "p95_e2e_ms": round(sorted(r.e2e_ms for r in successful)[int(n * 0.95)], 2),
        "p99_e2e_ms": round(sorted(r.e2e_ms for r in successful)[int(n * 0.99)], 2),
        "total_output_tokens": sum(r.output_tokens for r in successful),
        "avg_output_tokens": round(sum(r.output_tokens for r in successful) / n, 1),
        "throughput_req_per_s": round(n / (sum(r.e2e_ms for r in successful) / 1000 / n), 2)
            if n > 0 else 0,
    }
    return agg

# ---------------------------------------------------------------------------
# GPU Monitor subprocess
# ---------------------------------------------------------------------------
def launch_gpu_monitor(output_dir: str, interval: float = 0.5) -> Optional[asyncio.subprocess.Process]:
    """Launch nvidia-smi monitoring in a subprocess."""
    monitor_script = Path(__file__).resolve().parent / "07_gpu_monitor.py"
    if not monitor_script.exists():
        print("[06_loadgen] WARNING: GPU monitor script not found, skipping")
        return None
    # Will be started by the caller
    return None

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
                print(f"    → avg TTFT={agg['avg_ttft_ms']:.0f}ms  "
                      f"avg Decode={agg['avg_decode_ms']:.0f}ms  "
                      f"avg E2E={agg['avg_e2e_ms']:.0f}ms  "
                      f"p95={agg['p95_e2e_ms']:.0f}ms")

    # Stop GPU monitor
    if gpu_monitor_proc:
        gpu_monitor_proc.terminate()
        try:
            await asyncio.wait_for(gpu_monitor_proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            gpu_monitor_proc.kill()
        print("[06_loadgen] GPU monitor stopped")

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
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[06_loadgen] Results saved to {out_path}")

    # Summary table
    print("\n" + "=" * 90)
    print("LOAD TEST SUMMARY")
    print("=" * 90)
    header = (f"{'Scenario':<16s} {'Conc':>4s} {'Ctx':>6s} {'N':>4s} "
              f"{'TTFT':>8s} {'Decode':>8s} {'E2E':>8s} {'p95':>8s} {'tput':>8s}")
    print(header)
    print("-" * 90)
    for scenario_name, _, _ in runs:
        r = results.get(scenario_name, {})
        if r and "error" not in r:
            print(f"{scenario_name:<16s} {r['concurrency']:>4d} "
                  f"{r['context_tokens']:>6,d} {r['successful']:>4d} "
                  f"{r['avg_ttft_ms']:>7.0f}ms {r['avg_decode_ms']:>7.0f}ms "
                  f"{r['avg_e2e_ms']:>7.0f}ms {r['p95_e2e_ms']:>7.0f}ms "
                  f"{r['throughput_req_per_s']:>7.2f}/s")
    print("=" * 90)

if __name__ == "__main__":
    asyncio.run(main())
