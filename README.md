# CPU-GPU Co-Design for Agentic LLM Inference — Reproduction

Reproducing the CPU/GPU time decomposition benchmark from the [reference blog post](https://andyluo7.github.io/llm/amd/mi300x/vllm/lmcache/performance/2026/05/14/cpu-gpu-codesign-agentic-inference-mi300x/) on NVIDIA A800 GPUs using Qwen3-30B-A3B.

## Goal

Quantify, for each inference request, where time is actually spent:

| Component            | Where      | What It Captures                                                        |
| -------------------- | ---------- | ----------------------------------------------------------------------- |
| **t_serialize**      | Client CPU | JSON serialization of the request payload                               |
| **t_http_overhead**  | Server CPU | HTTP parsing + tokenization + scheduling + queue wait + KV cache lookup |
| **t_server_prefill** | Server GPU | Attention computation over all input tokens                             |
| **t_decode**         | Server GPU | Autoregressive token generation + streaming                             |
| **t_response_parse** | Client CPU | SSE chunk parsing + tool call extraction                                |

Then compare **HBM prefix cache** vs **LMCache CPU DRAM cache** to measure whether LMCache adds CPU overhead.

## Hardware

| Component | Specification                                      |
| --------- | -------------------------------------------------- |
| CPU       | Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz      |
| GPU       | 2× NVIDIA A800-SXM4-80GB (CUDA 0, CUDA 1)          |
| Model     | Qwen3-30B-A3B (MoE, ~60 GB FP16, fits single A800) |
| Framework | vLLM (CUDA) + LMCache                              |

### Why Qwen3-30B-A3B?

The reference blog used MiniMax-M2.5 (230 GB FP8) on 2× MI300X. Our A800 GPUs have different memory capacity (80 GB vs 192 GB), bandwidth, and software stack (CUDA vs ROCm). Qwen3-30B-A3B is a Mixture-of-Experts model (~60 GB FP16) that:

- Fits comfortably on a single A800 80 GB
- Shares the MoE architecture with MiniMax-M2.5
- Allows validation of the CPU-GPU breakdown methodology without OOM
- Can scale to TP=2 for multi-GPU scheduling overhead tests

## Project Structure

```
Reproduction/
├── PROJECT.md                          # This file
├── newPLAN.md                          # Build plan
├── setup.sh                            # Environment setup
├── run_all.sh                          # Full benchmark pipeline
├── .gitignore                          # Excludes reference/, models/, 3rd party
│
├── script/
│   ├── 01_tokenizer_benchmark.py       # Tokenizer CPU micro-benchmark
│   ├── 02_cpu_component_benchmark.py   # JSON/hash/SSE/detok micro-benchmarks
│   ├── 03_request_profiler.py          # E2E client-side request profiler
│   ├── 04_server_decomposition.py      # HBM prefix cache decomposition
│   ├── 05_lmcache_decomposition.py     # LMCache DRAM decomposition
│   ├── 06_load_generator.py            # Concurrent load generator
│   ├── 07_gpu_monitor.py              # nvidia-smi GPU utilization monitor
│   └── 08_analysis.py                 # Results analysis & report generator
│
├── result/                             # All benchmark outputs (*.json, *.png, *.md)
└── thirdparty/                         # third party code and software
```

## Quick Start

### 1. Setup Environment

```bash
cd Reproduction
bash setup.sh
source .venv/bin/activate
```

### 2. Download Model

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./models/Qwen3-30B-A3B
```

### 3. Run Micro-Benchmarks (no GPU needed)

These validate CPU-side measurements independently of the inference server:

```bash
python script/01_tokenizer_benchmark.py
python script/02_cpu_component_benchmark.py
```

### 4. Start vLLM Server — HBM Prefix Cache

```bash
vllm serve ./models/Qwen3-30B-A3B \
  --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
```

### 5. Run HBM Prefix Cache Benchmarks

```bash
python script/04_server_decomposition.py --url http://localhost:8000
python script/03_request_profiler.py --url http://localhost:8000 --matrix
```

### 6. Restart Server — LMCache DRAM

```bash
# Install LMCache if not already installed
pip install lmcache

# Restart server with LMCache
LMCACHE_LOCAL_CPU=true LMCACHE_CHUNK_SIZE=256 \
vllm serve ./models/Qwen3-30B-A3B \
  --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.78 \
  --host 0.0.0.0 --port 8000
```

### 7. Run LMCache Benchmarks

```bash
python script/05_lmcache_decomposition.py --url http://localhost:8000
python script/06_load_generator.py --url http://localhost:8000 --matrix --gpu-monitor
```

### 8. Generate Analysis Report

```bash
python script/08_analysis.py
# → result/08_analysis_report.md
# → result/08_cpu_gpu_split.png
# → result/08_tokenizer_scaling.png
# → result/08_hbm_vs_lmcache.png
```

## Test Matrix

From the blog (Section 2.3):

| Scenario    | Concurrency | Context | Purpose                           |
| ----------- | ----------- | ------- | --------------------------------- |
| single_1k   | 1           | 1,000   | Baseline: pure overhead           |
| single_8k   | 1           | 8,000   | Typical agent turn                |
| single_32k  | 1           | 32,000  | Large agent context               |
| single_100k | 1           | 100,000 | Maximum agent context             |
| conc4_8k    | 4           | 8,000   | Light multi-tenant                |
| conc16_32k  | 16          | 32,000  | Medium load                       |
| conc32_32k  | 32          | 32,000  | High load, moderate context       |
| conc32_100k | 32          | 100,000 | Stress: high load + large context |

## Expected Results (Validation Criteria)

The experiment is successfully reproduced if we observe:

1. ✅ **Tiny CPU cost for single requests** — 0.4–0.6% of E2E time
2. ✅ **10–15% CPU share at high concurrency** — 11–15% at 32 concurrent users
3. ✅ **Scheduling/queue wait dominating CPU time** — >90% of CPU overhead
4. ✅ **Little to no additional CPU overhead from LMCache** — CPU% delta < 1% between HBM-PC and LMCache

## Key Differences from Reference

| Aspect    | Reference (MI300X)        | Our Setup (A800)            |
| --------- | ------------------------- | --------------------------- |
| GPU       | 2× AMD MI300X (192 GB)    | 2× NVIDIA A800 (80 GB)      |
| Software  | ROCm 6.x                  | CUDA 12.x                   |
| Model     | MiniMax-M2.5 (230 GB FP8) | Qwen3-30B-A3B (~60 GB FP16) |
| TP Degree | 2                         | 1 (fits single GPU)         |
| KV Cache  | HBM3 / CPU DRAM           | HBM2e / CPU DRAM            |

## Reference

- Blog: [CPU-GPU Co-Design for Agentic LLM Inference](https://andyluo7.github.io/llm/amd/mi300x/vllm/lmcache/performance/2026/05/14/cpu-gpu-codesign-agentic-inference-mi300x/)
- Code: [github.com/andyluo7/cpu-gpu-codesign-agentic-inference](https://github.com/andyluo7/cpu-gpu-codesign-agentic-inference)
- LMCache: [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache)
- Model: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
