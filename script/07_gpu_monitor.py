#!/usr/bin/env python3
"""
GPU Utilization Monitor (nvidia-smi)
=====================================
Samples GPU utilization, memory usage, temperature, and power draw at a
configurable interval using nvidia-smi. Designed to run alongside benchmark
scripts to capture GPU activity during inference workloads.

Metrics collected per GPU:
  - utilization.gpu       [%]
  - utilization.memory    [%]
  - memory.used           [MiB]
  - memory.total          [MiB]
  - temperature.gpu       [°C]
  - power.draw            [W]
  - clocks.sm             [MHz]
  - clocks.memory         [MHz]

Output format: CSV with timestamp column, suitable for pandas/matplotlib.

Usage:
    # Run standalone (continuous until Ctrl+C)
    python script/07_gpu_monitor.py --interval 0.5 --output result/gpu_monitor.csv

    # Run for a fixed duration
    python script/07_gpu_monitor.py --interval 0.5 --duration 120

    # JSON output
    python script/07_gpu_monitor.py --interval 0.5 --output result/gpu_monitor.json --format json

Reference: BLOG.md Section 3.1 "GPU Activity Ratio"
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="GPU utilization monitor via nvidia-smi")
parser.add_argument("--interval", type=float, default=0.5,
                    help="Sampling interval in seconds (default: 0.5)")
parser.add_argument("--duration", type=float, default=0,
                    help="Total run duration in seconds (0 = run until interrupted)")
parser.add_argument("--output", default=None,
                    help="Output file path")
parser.add_argument("--format", choices=["csv", "json"], default="csv",
                    help="Output format")
parser.add_argument("--gpus", default="0,1",
                    help="GPU indices to monitor (default: 0,1)")
args = parser.parse_args()

if args.output:
    out_path = Path(args.output)
else:
    ext = "json" if args.format == "json" else "csv"
    out_path = Path(__file__).resolve().parents[1] / "result" / f"07_gpu_monitor.{ext}"
out_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# nvidia-smi query
# ---------------------------------------------------------------------------
# Build the nvidia-smi query command
QUERY_FIELDS = [
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.memory",
]

QUERY_STRING = ",".join(QUERY_FIELDS)

def get_gpu_stats(gpu_ids: list[int]) -> list[dict]:
    """Query nvidia-smi for current GPU stats. Returns list of dicts, one per GPU."""
    id_list = ",".join(str(i) for i in gpu_ids)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={QUERY_STRING}",
                "--format=csv,noheader,nounits",
                f"--id={id_list}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        samples = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            values = [v.strip() for v in line.split(",")]
            if len(values) >= len(QUERY_FIELDS):
                sample = {
                    "timestamp": values[0],
                    "gpu_index": int(values[1]),
                    "utilization_gpu_pct": _parse_float(values[2]),
                    "utilization_memory_pct": _parse_float(values[3]),
                    "memory_used_mib": _parse_float(values[4]),
                    "memory_total_mib": _parse_float(values[5]),
                    "temperature_gpu_c": _parse_float(values[6]),
                    "power_draw_w": _parse_float(values[7]),
                    "clocks_sm_mhz": _parse_float(values[8]),
                    "clocks_memory_mhz": _parse_float(values[9]),
                    "unix_timestamp": time.time(),
                }
                samples.append(sample)
        return samples
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[07_gpu] nvidia-smi error: {e}", file=sys.stderr)
        return []

def _parse_float(s: str) -> Optional[float]:
    """Parse a float, returning None for [Not Supported] or empty."""
    s = s.strip()
    if not s or s.lower() in ("[not supported]", "[unknown]", "n/a", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "unix_timestamp", "timestamp", "gpu_index",
    "utilization_gpu_pct", "utilization_memory_pct",
    "memory_used_mib", "memory_total_mib",
    "temperature_gpu_c", "power_draw_w",
    "clocks_sm_mhz", "clocks_memory_mhz",
]

class CsvWriter:
    def __init__(self, path: Path):
        self.fh = open(path, "w", newline="")
        self.writer = csv.writer(self.fh)
        self.header_written = False

    def write(self, samples: list[dict]):
        if not samples:
            return
        if not self.header_written:
            self.writer.writerow(CSV_COLUMNS)
            self.header_written = True
        for s in samples:
            row = [s.get(k) for k in CSV_COLUMNS]
            self.writer.writerow(row)

    def close(self):
        self.fh.close()

class JsonWriter:
    def __init__(self, path: Path):
        self.path = path
        self.samples = []

    def write(self, samples: list[dict]):
        self.samples.extend(samples)

    def close(self):
        output = {
            "benchmark": "gpu_monitor",
            "interval_seconds": args.interval,
            "num_samples": len(self.samples),
            "gpus_monitored": args.gpus,
            "samples": self.samples,
        }
        self.path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------
def main():
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    print(f"[07_gpu] Monitoring GPUs: {gpu_ids}")
    print(f"[07_gpu] Interval: {args.interval}s")
    print(f"[07_gpu] Output:   {out_path} ({args.format})")

    writer = CsvWriter(out_path) if args.format == "csv" else JsonWriter(out_path)
    start_time = time.time()
    sample_count = 0

    try:
        while True:
            samples = get_gpu_stats(gpu_ids)
            if samples:
                writer.write(samples)
                sample_count += 1
                if sample_count % 20 == 0:
                    gpu0 = next((s for s in samples if s["gpu_index"] == gpu_ids[0]), None)
                    if gpu0:
                        util = gpu0.get("utilization_gpu_pct", "N/A")
                        mem = gpu0.get("utilization_memory_pct", "N/A")
                        print(f"[07_gpu] Sample {sample_count}: "
                              f"GPU0 util={util}% mem={mem}%")

            elapsed = time.time() - start_time
            if args.duration > 0 and elapsed >= args.duration:
                print(f"[07_gpu] Duration {args.duration}s reached, stopping.")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n[07_gpu] Interrupted after {sample_count} samples, "
              f"{time.time() - start_time:.1f}s")

    finally:
        writer.close()
        print(f"[07_gpu] {sample_count} samples written to {out_path}")

        # Quick summary
        if args.format == "json" and hasattr(writer, 'samples'):
            samples = writer.samples
            if samples:
                gpu0_samples = [s for s in samples if s.get("gpu_index") == gpu_ids[0]]
                if gpu0_samples:
                    utils = [s.get("utilization_gpu_pct") for s in gpu0_samples
                             if s.get("utilization_gpu_pct") is not None]
                    if utils:
                        print(f"[07_gpu] GPU0 avg util: {sum(utils)/len(utils):.1f}%  "
                              f"min: {min(utils):.1f}%  max: {max(utils):.1f}%")

if __name__ == "__main__":
    main()
