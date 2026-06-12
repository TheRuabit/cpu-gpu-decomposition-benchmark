#!/bin/bash
# ==============================================================================
# Environment Setup — CPU-GPU Co-Design Reproduction
# ==============================================================================
# Sets up a Python 3.11 virtual environment with version-pinned dependencies
# for benchmarking the CPU/GPU split on NVIDIA A800 GPUs.
#
# Target hardware:
#   GPU:     2× NVIDIA A800-SXM4-80GB (Ampere, sm_80)
#   CUDA:    12.8
#   Driver:  570.172.08 (R570 branch)
#   Python:  3.11
#
# Pinned versions (compatibility-tested for CUDA 12.8 + Driver 570):
#   PyTorch  2.11.0  (cu128) — last release with official CUDA 12.8 wheels
#   vLLM     0.21.0           — community-verified with CUDA 12.8
#   LMCache  0.4.6            — latest stable (source-build for cu128 if needed)
#
# CUDA 12.8 constraint: Driver 570 supports CUDA ≤ 12.8. Wheels built for
# CUDA 12.9+ will NOT work. All packages MUST target CUDA 12.8 or earlier.
#
# Usage:
#   bash setup.sh
#   bash setup.sh --fresh   # Delete .venv and start from scratch
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FRESH=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh) FRESH=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# -------------------------------------------------------------------
# Version pins (CUDA 12.8 compatible)
# -------------------------------------------------------------------
# Why these versions:
#   torch 2.11.0 — last with cu128 wheels; 2.12 deprecated CUDA 12.8
#   vllm  0.21.0 — community-verified; has cu128-compatible wheel
#   lmcache 0.4.6 — latest stable (May 2026); source-build if needed
# -------------------------------------------------------------------
TORCH_VERSION="2.11.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
VLLM_VERSION="0.21.0"
LMCACHE_VERSION="0.4.6"

# Other deps (minimum versions)
TRANSFORMERS_VERSION="4.45.0"
AIOHTTP_VERSION="3.9.0"
MATPLOTLIB_VERSION="3.8.0"
NUMPY_VERSION="1.26.0"
PANDAS_VERSION="2.1.0"

echo "============================================"
echo " CPU-GPU Co-Design Reproduction — Setup"
echo " Target: NVIDIA A800 ×2 | CUDA 12.8 | Driver 570 | Python 3.11"
echo "============================================"
echo ""

# -------------------------------------------------------------------
# Python version check
# -------------------------------------------------------------------
PYTHON=$(which python3.11 2>/dev/null || which python3 || which python)
PY_VER=$($PYTHON --version 2>&1)
echo "[setup] Using: $PY_VER"

if ! echo "$PY_VER" | grep -q "3\.11"; then
    echo "[setup] WARNING: Expected Python 3.11, got: $PY_VER"
    echo "[setup] Install Python 3.11 and re-run, or continue at your own risk."
    echo "[setup] Recommended: apt install python3.11 python3.11-venv (Ubuntu/Debian)"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# -------------------------------------------------------------------
# CUDA / Driver check
# -------------------------------------------------------------------
echo ""
echo "[setup] Checking NVIDIA driver and CUDA..."
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: //' | cut -d' ' -f1)
    echo "[setup]   Driver: $DRIVER_VER"
    echo "[setup]   CUDA:   $CUDA_VER"

    if [[ -n "$CUDA_VER" ]]; then
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        if [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -le 8 ]]; then
            echo "[setup]   ✓ CUDA $CUDA_VER ≤ 12.8 — compatible with Driver 570"
        elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -gt 8 ]]; then
            echo "[setup]   ⚠ CUDA $CUDA_VER > 12.8 — Driver 570 may not support this"
            echo "[setup]     Packages built for CUDA 12.9+ may fail at runtime"
        fi
    fi
else
    echo "[setup]   WARNING: nvidia-smi not found. GPU monitoring won't work."
    echo "[setup]   Continue only if building on non-GPU machine (micro-benchmarks only)."
fi

# -------------------------------------------------------------------
# Virtual environment
# -------------------------------------------------------------------
if $FRESH && [ -d ".venv" ]; then
    echo ""
    echo "[setup] Removing existing .venv (--fresh)..."
    rm -r .venv
fi

if [ ! -d ".venv" ]; then
    echo ""
    echo "[setup] Creating Python 3.11 virtual environment..."
    $PYTHON -m venv .venv --clear
fi

echo "[setup] Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

# Verify pip is from venv
PIP_PATH=$(which pip)
if ! echo "$PIP_PATH" | grep -q ".venv"; then
    echo "[setup] ERROR: pip is not from .venv: $PIP_PATH"
    echo "[setup] Deactivate any other venvs and re-run."
    exit 1
fi
echo "[setup]   pip: $PIP_PATH"

# -------------------------------------------------------------------
# Upgrade pip
# -------------------------------------------------------------------
echo ""
echo "[setup] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet

# -------------------------------------------------------------------
# Step 1: PyTorch with CUDA 12.8 (cu128)
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Step 1/4: PyTorch $TORCH_VERSION (CUDA 12.8)"
echo "============================================"

echo "[setup] Installing torch==$TORCH_VERSION from cu128 index..."
pip install \
    "torch==$TORCH_VERSION" \
    --index-url "$TORCH_INDEX" \
    --extra-index-url "https://pypi.org/simple"

echo "[setup] Verifying PyTorch CUDA support..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    # Check CUDA compatibility
    cuda_ver = torch.version.cuda
    major, minor = map(int, cuda_ver.split('.'))
    if major == 12 and minor <= 8:
        print(f'  ✓ torch CUDA {cuda_ver} ≤ 12.8 — compatible with Driver 570')
    elif major == 12 and minor > 8:
        print(f'  ⚠ torch CUDA {cuda_ver} > 12.8 — may be incompatible with Driver 570')
    else:
        print(f'  ⚠ unexpected CUDA version: {cuda_ver}')
else:
    print('  WARNING: CUDA not available — GPU benchmarks will fail!')
"

# -------------------------------------------------------------------
# Step 2: vLLM (CUDA 12.8 compatible)
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Step 2/4: vLLM $VLLM_VERSION"
echo "============================================"

echo "[setup] Installing vllm==$VLLM_VERSION..."
# Install vLLM — it may pull its own torch; we re-install cu128 torch after
if pip install "vllm==$VLLM_VERSION" 2>&1 | tee /tmp/vllm_install.log; then
    echo "[setup] vLLM installed successfully."
else
    echo "[setup] vLLM $VLLM_VERSION failed — trying latest compatible version..."
    # Fallback: try without version pin (pip resolves best match)
    pip install vllm || {
        echo "[setup] ERROR: vLLM installation failed."
        echo "[setup] Building vLLM from source for CUDA 12.8 may be needed."
        echo "[setup] See: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/"
        exit 1
    }
fi

# Re-install torch cu128 (vLLM may have replaced it with a different CUDA build)
echo "[setup] Ensuring torch cu128 is preserved (vLLM may override)..."
pip install "torch==$TORCH_VERSION" \
    --index-url "$TORCH_INDEX" \
    --extra-index-url "https://pypi.org/simple" \
    --force-reinstall --no-deps 2>/dev/null || true

# Verify vLLM
echo "[setup] Verifying vLLM installation..."
python -c "
import vllm
print(f'  vLLM: {vllm.__version__}')
" 2>/dev/null || echo "[setup]   WARNING: Could not verify vLLM version"

# -------------------------------------------------------------------
# Step 3: LMCache
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Step 3/4: LMCache $LMCACHE_VERSION"
echo "============================================"

echo "[setup] Installing lmcache==$LMCACHE_VERSION..."
# LMCache default wheels may target CUDA 13.0; if the pre-built wheel fails,
# build from source which uses the system's torch/cuda
if pip install "lmcache==$LMCACHE_VERSION" 2>&1 | tee /tmp/lmcache_install.log; then
    echo "[setup] LMCache installed from pre-built wheel."
elif pip install "lmcache==$LMCACHE_VERSION" --no-build-isolation 2>&1 | tee /tmp/lmcache_install.log; then
    echo "[setup] LMCache installed from source (--no-build-isolation)."
else
    echo "[setup] WARNING: LMCache $LMCACHE_VERSION failed."
    echo "[setup] Trying latest lmcache..."
    pip install lmcache 2>/dev/null || {
        echo "[setup] WARNING: LMCache installation failed."
        echo "[setup] Phase 3 (LMCache benchmarks) will not work."
        echo "[setup] To install manually:"
        echo "  git clone https://github.com/LMCache/LMCache.git"
        echo "  cd LMCache && pip install -e . --no-build-isolation"
    }
fi

# -------------------------------------------------------------------
# Step 4: Remaining dependencies
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Step 4/4: Supporting packages"
echo "============================================"

echo "[setup] Installing transformers, aiohttp, analysis tools..."
pip install \
    "transformers>=$TRANSFORMERS_VERSION" \
    "tokenizers>=0.20.0" \
    "aiohttp>=$AIOHTTP_VERSION" \
    "matplotlib>=$MATPLOTLIB_VERSION" \
    "numpy>=$NUMPY_VERSION" \
    "pandas>=$PANDAS_VERSION"

# -------------------------------------------------------------------
# Final verification
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Verifying installation"
echo "============================================"

python -c "
import sys, torch, transformers, aiohttp, numpy
print(f'  Python:       {sys.version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count:    {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}:         {props.name} ({props.total_memory // (1024**3)} GB)')
print(f'  Transformers: {transformers.__version__}')
print(f'  aiohttp:      {aiohttp.__version__}')
print(f'  numpy:        {numpy.__version__}')
try:
    import vllm
    print(f'  vLLM:         {vllm.__version__}')
except ImportError:
    print('  vLLM:         NOT FOUND')
try:
    import lmcache
    print(f'  LMCache:      {lmcache.__version__}')
except ImportError:
    print('  LMCache:      NOT FOUND (Phase 3 benchmarks will skip)')
"

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo " Environment:"
echo "   Python:    3.11"
echo "   PyTorch:   $TORCH_VERSION  (CUDA 12.8 / cu128)"
echo "   vLLM:      $VLLM_VERSION"
echo "   LMCache:   $LMCACHE_VERSION"
echo "   GPU:       2× A800-SXM4-80GB"
echo ""
echo " Next steps:"
echo ""
echo " 1. Download the model (only after first compile of workflow):"
echo "    huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./models/Qwen3-30B-A3B"
echo ""
echo " 2. Run micro-benchmarks (no GPU needed):"
echo "    python script/01_tokenizer_benchmark.py"
echo "    python script/02_cpu_component_benchmark.py"
echo ""
echo " 3. Start vLLM server (HBM prefix cache baseline):"
echo "    vllm serve ./models/Qwen3-30B-A3B \\"
echo "      --tensor-parallel-size 1 \\"
echo "      --enable-prefix-caching \\"
echo "      --gpu-memory-utilization 0.85 \\"
echo "      --host 0.0.0.0 --port 8000"
echo ""
echo " 4. Run the full benchmark suite:"
echo "    bash run_all.sh --pilot      # Quick pilot test"
echo "    bash run_all.sh              # Full pipeline"
echo ""
