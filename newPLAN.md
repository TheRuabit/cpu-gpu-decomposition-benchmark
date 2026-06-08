Aimed at reproducing the CPU/GPU split and the HBM prefix cache vs LMCache comparison.

**Goal**
Recreate the benchmark so you can measure, for each request:

- client serialization time
- server HTTP/scheduler overhead
- GPU prefill time
- GPU decode time
- response parsing time

Then compare:

- HBM prefix cache
- LMCache CPU DRAM cache

**Tools**

- CPU: Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
- GPU: 2x NVIDIA A800-SXM4-80GB (CUDA 0, CUDA 1)

**Storage Location**

- All script file store in ./script/XX_filename.py
- All result file store in ./result/XX_filename.json
- All instruction and result summary store in ./FILENAME.md
- ./gitignore ./reference, models and all 3rd party file.

**Reminder**

- This project need to be upload on ssh server kwchen@192.168.0.150
- Download model only after the first compile of the workflow

**Plan**

1. **Lock the reproduction target**
   - Use the model family, framework, and hardware class if possible.
   - Match the blog’s matrix: single request and 4/16/32 concurrency, with 1k, 8k, 32k, and 100k context lengths.
   - Decide whether you want exact reproduction or a smaller “sanity-check” version first.

2. **Recreate the environment**
   - Build the same vLLM/ROCm stack version used in the post.
   - Set up the model weights and cache configuration.
   - Verify GPU visibility, tensor parallel setup, and prefix caching behavior.

3. **Rebuild the workload**
   - Collect or reuse agentic traces similar to the Claude Code conversations mentioned in the post.
   - Convert traces into benchmark inputs with the same context lengths and concurrency patterns.
   - Keep the request shape stable so the results are comparable.

4. **Instrument the request lifecycle**
   - Add timing around:
     - request serialization on the client
     - HTTP request arrival to first byte back
     - prefill start/end
     - decode start/end
     - SSE parsing on the client
   - Make sure the timing boundaries match the blog’s definitions, or the numbers won’t compare cleanly.

5. **Run the two cache configurations**
   - HBM prefix cache baseline.
   - LMCache DRAM configuration.
   - Use the same concurrency/context matrix for both.
   - Run multiple batches per scenario so you can average out noise.

6. **Isolate CPU hot spots**
   - Run micro-benchmarks for:
     - tokenization
     - JSON serialization
     - hash computation
     - SSE parsing
     - detokenization
   - This lets you verify whether scheduling/queue wait is really the dominant CPU cost.

7. **Analyze the results**
   - Compute CPU% vs GPU% for each scenario.
   - Check whether CPU overhead stays flat at low concurrency and rises sharply at 16–32 users.
   - Confirm whether LMCache changes CPU overhead meaningfully or not.
   - Look for superlinear growth in scheduling/queue wait.

8. **Validate the conclusion**
   - The experiment is reproduced if you see:
     - tiny CPU cost for single requests
     - 10–15% CPU share at high concurrency
     - scheduling/queue wait dominating CPU time
     - little to no additional CPU overhead from LMCache

**Recommended execution order**

1. Small pilot run with 1 concurrency and 1k/8k context.
2. Full benchmark matrix for HBM prefix cache.
3. Repeat with LMCache.
4. Micro-benchmarks.
5. Final comparison and write-up.
