#!/bin/bash
# ==============================================================================
# Full Benchmark Orchestration
# ==============================================================================
# Runs the complete benchmark pipeline in the recommended execution order:
#
#   Phase 1 — Micro-benchmarks (no GPU needed)
#     01_tokenizer_benchmark.py
#     02_cpu_component_benchmark.py
#
#   Phase 2 — E2E profiling (requires vLLM server)
#     04_server_decomposition.py (HBM prefix cache)
#     05_lmcache_decomposition.py (LMCache DRAM)
#     03_request_profiler.py (client-side)
#
#   Phase 3 — Load testing
#     06_load_generator.py (full matrix + GPU monitoring)
#
#   Phase 4 — Analysis
#     08_analysis.py (comparison charts + report)
#
# Usage:
#   bash run_all.sh                          # Run everything
#   bash run_all.sh --phase 1                # Micro-benchmarks only
#   bash run_all.sh --phase 2 --url http://...  # Server tests
#   bash run_all.sh --phase 4                # Analysis only
#   bash run_all.sh --pilot                  # Small pilot run (1k/8k, conc=1)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
PHASE="all"
SERVER_URL="http://localhost:8000"
MODEL="./models/Qwen3-30B-A3B"
PILOT=false
PYTHON="python"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --url) SERVER_URL="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --pilot) PILOT=true; shift ;;
        --python) PYTHON="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate 2>/dev/null || true
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate 2>/dev/null || true
fi

# Ensure output directories exist
mkdir -p result

echo "============================================"
echo " CPU-GPU Co-Design Benchmark Pipeline"
echo "============================================"
echo " Server:  $SERVER_URL"
echo " Model:   $MODEL"
echo " Phase:   $PHASE"
echo " Pilot:   $PILOT"
echo "============================================"
echo ""

# -------------------------------------------------------------------
# Phase 1: Micro-benchmarks (no GPU needed)
# -------------------------------------------------------------------
run_phase1() {
    echo "==========================================="
    echo " PHASE 1: CPU Micro-Benchmarks"
    echo "==========================================="

    echo ""
    echo "--- 01 Tokenizer Benchmark ---"
    $PYTHON script/01_tokenizer_benchmark.py --model "$MODEL"
    echo "  ✓ result/01_tokenizer_benchmark.json"

    echo ""
    echo "--- 02 CPU Component Benchmark ---"
    $PYTHON script/02_cpu_component_benchmark.py --model "$MODEL"
    echo "  ✓ result/02_cpu_component_benchmark.json"

    echo ""
    echo "[Phase 1] Complete."
}

# -------------------------------------------------------------------
# Phase 2: E2E profiling (needs vLLM server)
# -------------------------------------------------------------------
run_phase2() {
    echo "==========================================="
    echo " PHASE 2: E2E Request Profiling"
    echo "==========================================="

    # Server health check
    echo "[Phase2] Checking server at $SERVER_URL ..."
    if ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo "[Phase2] ERROR: Cannot reach vLLM server at $SERVER_URL"
        echo "[Phase2] Start the server first:"
        echo "  vllm serve ./models/Qwen3-30B-A3B --enable-prefix-caching --gpu-memory-utilization 0.85 --host 0.0.0.0 --port 8000"
        exit 1
    fi
    echo "[Phase2] Server is reachable."

    if $PILOT; then
        echo "[Phase2] Running PILOT (small scale)..."
        PILOT_SCENARIOS="single_1k single_8k"

        echo ""
        echo "--- 04 HBM Prefix Cache Decomposition (pilot) ---"
        $PYTHON script/04_server_decomposition.py \
            --url "$SERVER_URL" --model "$MODEL" \
            --scenarios $PILOT_SCENARIOS \
            --num-batches 2 --warmup-batches 1
        echo "  ✓ result/04_server_decomposition.json"

        echo ""
        echo "--- 03 Request Profiler (pilot) ---"
        $PYTHON script/03_request_profiler.py \
            --url "$SERVER_URL" --model "$MODEL" \
            --context 1000 --concurrency 1 --num-batches 2
        echo "  ✓ result/03_request_profiler.json"
    else
        echo ""
        echo "--- 04 HBM Prefix Cache Decomposition ---"
        $PYTHON script/04_server_decomposition.py \
            --url "$SERVER_URL" --model "$MODEL" \
            --num-batches 3 --warmup-batches 1
        echo "  ✓ result/04_server_decomposition.json"

        # GPU monitor during the gap
        echo ""
        echo "--- GPU Monitor (30s idle sample) ---"
        $PYTHON script/07_gpu_monitor.py --interval 0.5 --duration 30 \
            --output result/07_gpu_idle.csv &
        sleep 32

        echo ""
        echo "--- 03 Request Profiler (full matrix) ---"
        $PYTHON script/03_request_profiler.py \
            --url "$SERVER_URL" --model "$MODEL" \
            --matrix --num-batches 3
        echo "  ✓ result/03_request_profiler.json"
    fi

    echo ""
    echo "[Phase 2] Complete."
}

# -------------------------------------------------------------------
# Phase 3: LMCache + Load testing
# -------------------------------------------------------------------
run_phase3() {
    echo "==========================================="
    echo " PHASE 3: LMCache & Load Testing"
    echo "==========================================="

    # Check if LMCache server config is available
    echo "[Phase3] NOTE: For LMCache tests, restart vLLM with LMCache config:"
    echo "  LMCACHE_LOCAL_CPU=true LMCACHE_CHUNK_SIZE=256 \\"
    echo "  vllm serve ./models/Qwen3-30B-A3B --enable-prefix-caching \\"
    echo "    --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \\"
    echo "    --gpu-memory-utilization 0.78 --host 0.0.0.0 --port 8000"
    echo ""

    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo "[Phase3] Server is running. Run LMCache decomposition..."
        echo ""

        if $PILOT; then
            echo "--- 05 LMCache Decomposition (pilot) ---"
            $PYTHON script/05_lmcache_decomposition.py \
                --url "$SERVER_URL" --model "$MODEL" \
                --scenarios single_1k single_8k conc4_8k \
                --num-batches 2 --warmup-batches 1
        else
            echo "--- 05 LMCache Decomposition ---"
            $PYTHON script/05_lmcache_decomposition.py \
                --url "$SERVER_URL" --model "$MODEL" \
                --num-batches 3 --warmup-batches 1
        fi
        echo "  ✓ result/05_lmcache_decomposition.json"

        echo ""
        echo "--- 06 Load Generator (with GPU monitor) ---"
        $PYTHON script/06_load_generator.py \
            --url "$SERVER_URL" --model "$MODEL" \
            --matrix --num-batches 3 --gpu-monitor
        echo "  ✓ result/06_load_generator.json"
    else
        echo "[Phase3] WARNING: Server not reachable. Skipping LMCache + load tests."
        echo "[Phase3] Start LMCache-configured server and re-run: bash run_all.sh --phase 3"
    fi

    echo ""
    echo "[Phase 3] Complete."
}

# -------------------------------------------------------------------
# Phase 4: Analysis & report
# -------------------------------------------------------------------
run_phase4() {
    echo "==========================================="
    echo " PHASE 4: Results Analysis"
    echo "==========================================="

    ANALYSIS_SCRIPT="script/08_analysis.py"
    if [ -f "$ANALYSIS_SCRIPT" ]; then
        $PYTHON "$ANALYSIS_SCRIPT"
        echo "  ✓ result/08_analysis_report.md"
    else
        echo "[Phase4] Analysis script not found: $ANALYSIS_SCRIPT"
        echo "[Phase4] Manually review results in ./result/"
    fi
}

# -------------------------------------------------------------------
# Dispatch
# -------------------------------------------------------------------
case "$PHASE" in
    all)
        run_phase1
        echo ""
        read -p "Press Enter to continue to Phase 2 (ensure vLLM server is running)..."
        run_phase2
        echo ""
        read -p "Press Enter to continue to Phase 3 (restart server with LMCache if needed)..."
        run_phase3
        echo ""
        run_phase4
        ;;
    1) run_phase1 ;;
    2) run_phase2 ;;
    3) run_phase3 ;;
    4) run_phase4 ;;
    *)
        echo "Unknown phase: $PHASE (use 1, 2, 3, 4, or 'all')"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo " Benchmark pipeline complete!"
echo " Results: ./result/"
echo " Report:  ./PROJECT.md"
echo "============================================"
