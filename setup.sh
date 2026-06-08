#!/bin/bash
# ==============================================================================
# Environment Setup — CPU-GPU Co-Design Reproduction
# ==============================================================================
# Sets up the Python environment and installs dependencies for benchmarking
# the CPU/GPU split in agentic LLM inference on NVIDIA A800 GPUs.
#
# Usage:
#   bash setup.sh
#   bash setup.sh --cuda  # Also install CUDA-specific packages
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " CPU-GPU Co-Design Reproduction — Setup"
echo "============================================"
echo ""

# --- Python version check ---
PYTHON=$(which python3 || which python)
echo "[setup] Python: $($PYTHON --version)"
echo "[setup] pip: $(pip --version 2>/dev/null || pip3 --version)"

# --- Create virtual environment if needed ---
if [ ! -d ".venv" ]; then
    echo "[setup] Creating virtual environment..."
    $PYTHON -m venv .venv
fi

echo "[setup] Activating virtual environment..."
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

# --- Core dependencies ---
echo ""
echo "[setup] Installing core dependencies..."

pip install --upgrade pip setuptools wheel

# LLM framework
pip install "vllm>=0.6.0"          # vLLM for NVIDIA CUDA

# Model & tokenizer
pip install "transformers>=4.45.0"
pip install "tokenizers>=0.20.0"

# Async HTTP client for benchmarks
pip install "aiohttp>=3.9.0"

# LMCache (for CPU DRAM cache experiments)
pip install "lmcache" 2>/dev/null || echo "[setup] NOTE: lmcache install failed — install separately if needed"

# Analysis & visualization
pip install "matplotlib>=3.8.0"
pip install "numpy>=1.26.0"
pip install "pandas>=2.1.0"

# --- CUDA check ---
echo ""
if command -v nvidia-smi &>/dev/null; then
    echo "[setup] NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
else
    echo "[setup] WARNING: nvidia-smi not found. GPU monitoring won't work."
    echo "[setup] If running on A800, ensure NVIDIA drivers are installed."
fi

# --- Model download instructions ---
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo " Next steps:"
echo ""
echo " 1. Download the model:"
echo "    huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./models/Qwen3-30B-A3B"
echo ""
echo " 2. Verify GPU visibility:"
echo "    nvidia-smi"
echo "    python -c \"import torch; print(f'CUDA devices: {torch.cuda.device_count()}')\""
echo ""
echo " 3. Start vLLM server (HBM prefix cache baseline):"
echo "    vllm serve ./models/Qwen3-30B-A3B \\"
echo "      --tensor-parallel-size 1 \\"
echo "      --enable-prefix-caching \\"
echo "      --gpu-memory-utilization 0.85 \\"
echo "      --host 0.0.0.0 --port 8000"
echo ""
echo " 4. Run micro-benchmarks (no GPU needed):"
echo "    python script/01_tokenizer_benchmark.py"
echo "    python script/02_cpu_component_benchmark.py"
echo ""
echo " 5. Run the full benchmark suite:"
echo "    bash run_all.sh"
echo ""
