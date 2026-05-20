# **Introduction**

This document details the development and optimization of an inference pipeline for the *AI Mathematical Olympiad \- Progress Prize 3* competition hosted on Kaggle. Over a three-month period, a comprehensive system was developed utilizing:

* GPT-OSS-120B with `ReasoningEffort.HIGH`
* OpenAI's Harmony Prompt template
* Tool-Integrated Reasoning (TIR) with Python execution

## **Computational Resources**

The competition provides access to a single instance with the following specifications:

* **1× H100 SXM GPU**: 80 GB HBM3 memory, 3.35 TB/s memory bandwidth, native FP8 support, MXFP4 support via Marlin backend (non-native)
* **26 CPU workers**
* **230 GB system memory**
* **No internet access**

The runtime limit for submission notebooks is constrained to **5 hours**.

## **GPT-OSS Technical Specifications**

The GPT-OSS models are autoregressive Mixture-of-Experts (MoE) transformers.

### **Architecture**
*   **Residual Stream**: 2880 dimension.
*   **Layer Normalization**: Pre-LN placement with RMSNorm.
*   **Activation**: Gated SwiGLU (unconventional implementation with clamping and a residual connection).
*   **MoE Strategy**: Both models utilize a top-4 expert selection mechanism.
    *   **GPT-OSS-120B**: 36 layers, 116.8B total parameters (5.1B active), 128 experts per MoE block.
    *   **GPT-OSS-20B**: 24 layers, 20.9B total parameters (3.6B active), 32 experts per MoE block.

### **Attention & Context**
*   **Grouped Query Attention (GQA)**: 64 query heads (dim 64) and 8 key-value heads.
*   **Patterning**: Alternating banded window (128-token bandwidth) and fully dense patterns.
*   **Context Window**: 131,072 tokens via YaRN scaling and rotary position embeddings (RoPE).
*   **Softmax Stability**: Learned bias in the denominator (attention sinks) to allow the attention mechanism to pay no attention to specific tokens.

### **Data Type & Quantization**
The models utilize the **MXFP4** (OCP Microscaling Formats) specification for MoE weights, which comprise over 90% of the parameter count. This 4.25 bits-per-parameter quantization allows the 120B model to fit within a single 80GB H100 GPU and the 20B model to run on systems with as little as 16GB memory.

### **Harmony Protocol & Tool Use**
The models use the **Harmony chat format**, implementing a strict instruction hierarchy (**System > Developer > User > Assistant > Tool**) and visibility **channels**:
*   `analysis`: For internal Chain-of-Thought (CoT) reasoning.
*   `commentary`: For function tool-calling logic and internal deliberation.
*   `final`: For the final user-facing response.

GPT-OSS natively supports **Tool-Integrated Reasoning (TIR)** with stateful Python execution in Jupyter notebook environments and web-browsing capabilities. Reasoning effort is variable and configurable (`low`, `medium`, `high`) via system prompt keywords.

## **Qwen3.5-35B-A3B Experimental Findings**

An attempt was made to adapt the GPT-OSS pipeline to work with the ChatML template for Qwen3.5-35B-A3B (utilizing `vllm==0.17.0`, the first version to officially support the model). During execution:
* The H100 VRAM allocated ~77.5 GB, yet GPU utilization remained at 0%.
* CPU utilization reached 2400% (pegging 24 cores), and system RAM consumption increased from 61 GB to over 190 GB, forcing manual termination before an OOM event.

Since the exact same pipeline was stable with GPT-OSS, this rules out OS Page Cache loading or the asynchronous execution of 16 sandboxes (with pre-imported heavy libraries like `sympy` and `numpy`) as the root cause. The observed behavior suggests that the `qwen3_5_moe` implementation path in vLLM was not mature enough for this workload at the time. Although the pipeline carried over FlashAttention 3 (FA3) from the GPT-OSS configuration, changing only the attention or MoE backend was unlikely to resolve the observed CPU and memory behavior.

## **vLLM 0.17.0 & Inference Backend Observations**

The adoption of `vllm==0.17.0` for the GPT-OSS-120B pipeline introduced new server arguments and behavioral changes that significantly impacted stability and memory constraints:

* **Model Runner V2:** When enabling the `VLLM_USE_V2_MODEL_RUNNER` flag, the model exhibited a large drop in reasoning performance and did not solve basic reference problems (e.g., the "Alice and Bob" problem). Removing the flag entirely restored the baseline mathematical performance.
* **Triton vs. Marlin MoE Backends:** Experimentation with the new `--moe-backend` argument revealed critical differences in memory footprints. Utilizing the `triton` backend (default option if Marlin is not specified) triggered a CUDA OOM error with a 46 MB shortfall even when capped at `gpu_memory_utilization=0.98`. In contrast, the `marlin` backend successfully executes at `gpu_memory_utilization=0.99` without OOM issues. Furthermore, with the Marlin backend on `vllm==0.17.0`, overall VRAM consumption decreased to 78.6 GB (compared to 78.8 GB on `vllm==0.16.0`), making it the required choice for hosting the 120B model. The concurrency remained *identical* to `vllm==0.16.0` (`Maximum concurrency for 81,920 tokens per request: 7.77x`).
* **Throughput vs Interactivity Mode:** Furthermore, `vllm==0.17.0` is the first release to introduce the `--performance-mode` server argument. To ascertain the optimal configuration for long-context tool-use workloads, metrics were parsed and aggregated across four complete test runs (two runs per mode).

#### Summary Averages (Combining Both Runs)

**Interactivity Mode (Average)**
- **Server Uptime:** ~41m 28s
- **Successful Completion Requests (200 OK):** 1590

| Metric | Mean | Max | Min | Samples |
|---|---|---|---|---|
| Prompt Throughput (tokens/s) | 155.41 | 575.70 | 8.70 | 80 |
| Generation Throughput (tokens/s) | 608.21 | 893.10 | 171.20 | 80 |
| GPU KV Cache Usage (%) | 16.15% | 43.30% | 0.40% | 80 |
| Prefix Cache Hit Rate (%) | 98.48% | 99.20% | 96.20% | 80 |

**Throughput Mode (Average)**
- **Server Uptime:** ~32m 25s
- **Successful Completion Requests (200 OK):** 1275.5

| Metric | Mean | Max | Min | Samples |
|---|---|---|---|---|
| Prompt Throughput (tokens/s) | 146.35 | 524.60 | 13.10 | 62 |
| Generation Throughput (tokens/s) | 667.08 | 855.80 | 260.10 | 62 |
| GPU KV Cache Usage (%) | 13.09% | 38.10% | 0.00% | 62 |
| Prefix Cache Hit Rate (%) | 97.85% | 98.90% | 91.20% | 62 |

#### Detailed Breakdown by Run

| Mode / Run | Server Uptime | Prompt Mean (tok/s) | Gen Mean (tok/s) | Prefix Hit Mean (%) | KV Cache Peak (%) | Successful Reqs |
|---|---|---|---|---|---|---|
| **Interactivity - Run 1** | 40m 27s | 145.37 | 627.47 | 98.46% | 43.30% | 1472 |
| **Interactivity - Run 2** | 42m 29s | 164.95 | 589.88 | 98.49% | 33.50% | 1708 |
| **Throughput - Run 1** | 29m 27s | 147.47 | 670.01 | 97.52% | 38.10% | 1145 |
| **Throughput - Run 2** | 35m 23s | 145.42 | 664.67 | 98.13% | 31.90% | 1406 |

#### Performance Conclusion

Comparing the pure throughput metrics from the aggregated vLLM server logs (specifically isolating core generation intervals from total runtimes which are susceptible to variance like the 10-minute stalls occasionally observed during interactivity-mode runs):

- **Prompt Throughput (Context Processing):** Interactivity mode averaged 155.41 tokens/s, marginally outperforming Throughput mode's 146.35 by ~9 tokens/s.
- **Generation Throughput (Token Generation):** Throughput mode averaged 667.08 tokens/s, significantly outperforming Interactivity mode's 608.21 by ~59 tokens/s.
- **Prefix Cache Hit Rate:** Both modes achieved near-perfect hit rates (Interactivity: 98.48%, Throughput: 97.85%).

**Final Recommendation for the AIMO3 Pipeline:**

For long-context tool-use workloads specifically characterized by sequences that request inference repeatedly but extend the existing context iteratively, **Throughput Mode performs better in the observed runs.** Because the overarching Prefix Cache Hit Rate is high (~98%), vLLM bypasses most of the prefill processing during identical problem-solving loops.

Consequently, the minor advantage that Interactivity mode holds for prompt processing is limited because most repeated requests benefit from KV caching. With the prompt bottleneck reduced, performance rests primarily on how fast the model can output function tool arguments and text responses. The logged statistics indicate that `throughput` mode provides ~10% faster output generation for this workload.

> *Note: Mk IV is the first series to implement both `vllm==0.17.0` and its `--performance-mode` argument, building upon Version 5 of Mk III.*

## **CUDA Environment Log Analysis (vLLM 0.17.0)**

With the upgrade to `vllm==0.17.0` (which requires `torch==2.10.0`), a comprehensive analysis was conducted to determine the optimal CUDA environment for the H100 SXM instance. Two separate configurations were tested across 4 full execution runs (2 runs using `cu128` wheels and 2 runs using `cu130` wheels), both utilizing an identical `CFG` class (parameters specific to context length, attention/compilation config and gpu utilization are listed below).

The vLLM engine logs were parsed and aggregated to compare the two environments explicitly. The resulting metrics isolate active generation phases (excluding idle `0.0 tokens/s` polling intervals):

| Metric | cu128 Run 1 | cu128 Run 2 | **cu128 Avg** | cu130 Run 1 | cu130 Run 2 | **cu130 Avg** |
|---|---|---|---|---|---|---|
| **Active Prompt Mean (tok/s)** | 173.1 | 163.5 | **168.3** | 136.6 | 149.3 | **142.9** |
| **Prompt Peak (tok/s)** | 1024.5 | 1145.2 | **1084.8** | 967.7 | 904.2 | **936.0** |
| **Active Gen Mean (tok/s)** | 635.6 | 608.2 | **621.9** | 569.7 | 550.5 | **560.1** |
| **Gen Peak (tok/s)** | 939.1 | 929.5 | **934.3** | 932.7 | 926.3 | **929.5** |
| **KV Cache Peak (%)** | 30.5% | 35.0% | **32.8%** | 48.3% | 50.6% | **49.5%** |
| **Prefix Hit Mean (%)** | 97.9% | 98.1% | **98.0%** | 97.8% | 98.4% | **98.1%** |
| **Prefix Hit Peak (%)** | 98.7% | 99.0% | **98.8%** | 99.0% | 99.3% | **99.2%** |

### **Configuration Used**

```python
attention_config = {
    'backend': 'FLASH_ATTN',
    'flash_attn_version': 3,
    'use_prefill_decode_attention': True,
    'flash_attn_max_num_splits_for_cuda_graph': 32,
    'use_cudnn_prefill': False,
    'use_trtllm_ragged_deepseek_prefill': False,
    'use_trtllm_attention': False,
    'disable_flashinfer_prefill': True,
    'disable_flashinfer_q_quantization': True
}

compilation_config = {
    'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32, 64],
    'max_cudagraph_capture_size': 64
}

kv_cache_dtype = 'fp8_e4m3'
performance = 'throughput'
moe_backend = 'marlin'

gpu_memory_utilization = 0.99
context_tokens = 81920
batched_tokens = 2048
batch_size = 64
```

**Key Findings:**
1. **Throughput Superiority (cu128):** The `cu128` environment consistently outperformed `cu130` in raw processing speeds across both isolated runs. It achieved higher average generation throughput (+61.8 tokens/s on average) and higher peak prompt processing speed (+148.8 tokens/s on average).
2. **KV Cache Dynamics:** The `cu130` environment exhibited a higher peak KV Cache utilization (averaging 49.5% vs 32.8%), despite processing fewer tokens per second. This suggests that the `cu130` wheels may be handling context window scaling or sequence chunking less efficiently than the `cu128` compiled binaries.
3. **Prefix Caching Parity:** Both environments demonstrated near-identical, high Prefix Cache hit rates (averaging ~98%), confirming that the pipeline's architectural reliance on vLLM's Automatic Prefix Caching (APC) remains robust regardless of the underlying CUDA version.

Based on these empirical metrics, the **`cu128` wheels are a better fit** for this specific pipeline on the Kaggle H100 instance, yielding faster token generation and more efficient memory management.

## **Fine-Tuning Decision**

Fine-tuning was intentionally omitted from the final competition pipeline. The submitted notebooks prioritize inference-time reliability because the competition environment provides one H100 GPU, no internet access, and a 5-hour runtime limit.

For this workload, fine-tuning has an unfavorable cost profile:

* Long-context solution traces are expensive to generate. With 81,920-token context and Tool-Integrated Reasoning, 50 problems already require approximately 5 hours of inference. A curated set of hundreds or thousands of problems would multiply that cost before any optimizer step is performed.
* Preference and reinforcement-style methods require multiple rollouts per problem and repeated sampling. On a single H100, scaling those methods to a meaningful dataset would require substantially more accelerator time than the competition setup provides.
* Fine-tuning on model-generated correct traces risks reproducing the teacher model's existing behavior rather than improving the submitted 120B system. For this pipeline, the limiting factors were inference stability, time allocation, tool reliability, and answer aggregation.
* Methods that require negative samples or preference labels introduce additional inference cost and label-quality risk. Without a reliable, decontaminated preference set, the expected return was low.

Consequently, final development focused on inference pipeline design: context length, KV cache data type, vLLM backend choices, Python tool reliability, sandbox overprovisioning, prompt minimalism, and score aggregation.

## **Experimental Approaches Not Used**

Following the establishment of a functional baseline pipeline, approximately one month was invested in experimenting with enhancements to improve the score from 39-42. The approaches below were not included in the final notebooks because they did not improve measured outcomes or added excessive runtime and reliability costs.

### **Architectural Modifications**

These represent fundamental pipeline modifications targeting the model directly. The following model-level changes were evaluated but not included in the final pipeline:

* **Reasoning Truncation:** Attempted to truncate the "middle" of the reasoning trace to save context window space.
  * **Issue 1 (Speed):** This breaks vLLM's Automatic Prefix Caching (APC) because the prefix changes for every subsequent token generation, forcing a full re-computation of the KV cache and significantly slowing down inference.
  * **Issue 2 (Accuracy):** Reasoning steps in the middle of a trace often contain critical intermediate results; removing them can reduce solution quality.
  * **Issue 3 (Necessity):** It is rare for a sequence to exceed the 81,920 token limit (happens \<5% of the time on the hardest problems), making this optimization costly relative to its expected benefit.
* **Entropy-Based Pruning (Pass@12/Pass@16 -> Stop@8)**. This experiment attempted to start with a larger population of attempts (12 or 16) and prune the 4 or 8 highest-entropy sequences at Turn 8, continuing with the best 8. The hypothesis was that selecting a subset of attempts based on confidence
(low entropy) would outperform a random set of 8.
  * **Issue 1 (Trigger Condition)**: The pruning logic required all 12 attempts to reach Turn 8 simultaneously to compare their
    entropy. If a single attempt finished early (e.g., at Turn 4), the condition len(active_attempts) == 12 was never met, and pruning never
    triggered. This forced the system to run all 12 attempts to completion, incurring a 50% computational penalty without measured benefit.
  * **Issue 2 (Consensus Threshold)**: Increasing the starting population to 12 raised the consensus threshold (typically ceil(N/2)) from 4 to 6. This
    made it significantly harder for the system to "early stop" on easier problems, causing unnecessary resource drain waiting for a consensus that
    took longer to form than in the N=8 baseline.
  * **Issue 3 (Signal Validity)**: Empirical observation suggests that low entropy does not necessarily correlate with "correctness" in CoT
    reasoning. Low-entropy incorrect attempts can be preserved while high-entropy attempts that explore useful intermediate paths are removed. The strongest result from one version (V7P5) was likely driven by the larger number of attempts (12) rather than by the pruning
    logic itself, which may not have triggered.
  * **Result**: The added overhead (+10-20 minutes on the reference set) and the sensitivity of the trigger made this approach non-viable for the 5-hour
    limit.
* **In-Context Learning** (ICL)
* **Retrieval-Augmented Generation** (RAG)
* **Heuristic Search** (Monte Carlo Tree Search \+ Process Reward Models)
* **Speculative Decoding** utilizing `nvidia/gpt-oss-120b-Eagle3-throughput` (a model developed specifically for GPT-OSS-120B by Nvidia). Inference became approximately 25% slower in the tested configuration.
* **Gumbel Noise Injection** into the model's router (inspired by the "MoEs are Stronger than You Think" paper). Monkey-patching was employed to enable vLLM to recognize the custom code; however, it was never invoked during inference. The original paper likely employed a custom implementation rather than a framework such as vLLM or TensorRT.
* **Sequential Refinement** (MEDIUM reasoning effort, 2 turns). Inspired by "The Sequential Edge" and DeepSeekMath-V2 papers, a two-turn architecture was implemented: an initial generation turn followed by a verification/refinement turn with fresh context. The refinement turn received only the problem statement and the claimed answer from the first turn's final channel (not the full chain of thought). Empirical results showed lower performance than single-turn HIGH reasoning across all metrics: Problems 1-9 required more time to solve, and Problem 10 achieved 0/k success rate (compared to occasional success with HIGH reasoning). The additional computational overhead of the second turn did not provide enough error-correction benefit to justify the increased latency. Sequential Refinement with HIGH reasoning was not evaluated, as single-turn HIGH reasoning already exhausts the available 5-hour runtime limit; employing HIGH reasoning for both turns would require a runtime budget exceeding this constraint.
* **Domain Specific Language (DSL) and Claim Prover tools**. Two additional tools were implemented alongside the Python interpreter: a DSL tool providing specialised mathematical functions not available in standard libraries, and a Prover tool for formal verification of conjectures using SymPy and Z3. Neither tool demonstrated meaningful adoption: the DSL tool was invoked by only one attempt on a single problem (3 invocations total), while the Prover tool was never utilised. This suggests that the benefit of additional tools was lower than expected, and the model exhibits strong preference for the familiar Python environment.
* **Roster of Experts & Deep-Thinking Tokens (DTR):** Inspired by recent literature on expert diversification (*MoEs are Stronger than You Think*) and internal reasoning measurement (*Think Deep, Not Just Long*), we attempted to implement dynamic expert routing (RoE) and Deep-Thinking Ratio (DTR) to establish a more reliable proxy for ground-truth convergence than Mean Entropy. Both approaches are attractive in theory, but assume the use of eager-execution frameworks (like the `transformers` library) where low-level LLM features such as individual layers, logits, and router mechanisms are natively exposed. In practice, implementing these inside throughput-focused inference engines like vLLM, SGLang, or TensorRT-LLM requires disabling CUDA graphs via the `--enforce-eager` flag. This choice alone increased the inference time per problem by a factor of 400%. Because the AIMO3 pipeline requires high token throughput to fit within the 5-hour limit, any potential qualitative benefits were outweighed by the loss of execution speed.

### **Prompt Engineering Experiments**

These approaches involved modifying model instructions. This section focuses on channels in the Harmony Template that were not utilised in the final submitted notebooks:

* **Prompt Diversity** (4 different prompts instead of 1\)
* **Different role identities**: Codeforces Grandmaster, AoPS Contributor, or combinations with IMO Gold Medalist. Only IMO-related roles maintained consistent success rates on both ground truth and short sequences. Additionally, an extended prompt (approximately 15 lines) was tested establishing a progression from AIME to USAMO to IMO in an effort to establish a more robust persona. While local performance (on the 10-problem reference set) improved (shorter sequences, higher success rate on Problem No. 10), leaderboard performance oscillated between 35 and 37\. This confirms that minimal prompts are optimal.
* **Verification Turn** with fresh context (one attempt per unique answer)
* **Specified `conversation_start_date`** in the `SystemContent` class. Despite expectations of improved stability, every notebook utilizing this scored below 38 on the leaderboard.
* **Separate `DeveloperContent` class in addition to `SystemContent`** (*) Complete utilization of the Harmony template was expected to improve stability and thus increase leaderboard scores. The measured results did not support this change. While exact prompts were not preserved, they constituted variations of:
  * *"Use Python to verify your reasoning. Treat code as a Computer Algebra System rather than a plain calculator tool."*
  * *"Use Python to solve the problem. If parameters are computationally infeasible—such as towers of powers or large factorials—never compute values directly. Instead, use modular arithmetic to simplify expressions algebraically."*
  * A separate experiment used regex-based extraction of problem constraints (dividend, modulo, bounds) for injection into the developer prompt to maintain information salience. While not evaluated on the leaderboard, performance on the reference set did not improve and was slightly lower for Problem 10.

Variations of the first two prompts for `DeveloperContent()` were tested. For every notebook submitted to the public test set, the leaderboard score was lower (35-37). In some cases, sequence lengths for **easier** problems were at least 50% longer.

(*) A caveat for these experiments is that the DEVELOPER channel changes were accompanied by prompt changes. Specifically, the user prompt did not mention the libraries (``Use `math`, `numpy`, `sympy` to solve the problem.`` or variations of this), which previously helped keep those libraries salient in the model's recent context. The developer prompt also added instructions with limited measured signal, and the user prompt appended after the problem text was missing in the experiments that included the developer channel.

## **The Sandbox Overprovision Phenomenon**

The inference pipeline employs a pool of stateful Jupyter kernel environments (sandboxes) managed via a thread-safe FIFO queue. Sandboxes are created at startup and persist for the entire notebook execution (all 50 problems), being reset between problems rather than recreated.

### **Sandbox Acquisition Mechanism**

Each attempt acquires a sandbox via blocking queue retrieval with 3-second timeout:
```python
sandbox = self.sandbox_pool.get(timeout=3)
```

The sandbox is retained for the attempt's entire lifetime (up to 192 turns with multiple Python executions), enabling stateful computation across tool calls. After completion, the sandbox is reset and returned to the pool.

### **The Empirical Paradox**

**Configuration:** `attempts=8` (parallel solution attempts per problem), `workers=N` (number of sandboxes created)

**Theoretical Expectation:** With 8 parallel attempts, only 8 sandboxes are necessary. The remaining `N-8` sandboxes sit idle throughout execution.

**Empirical Observation:** Every notebook scoring 40-42 on the leaderboard uses `workers = 16` (2× overprovision). Configurations using `workers=8` (1:1 ratio) score mid-to-high 30s.

### **Reference Set Testing**

On the 10-problem reference subset, `workers=8` versus `workers=16` configurations show **no measurable performance difference**. Problems are solved in identical time with identical success rates. This suggests the effect emerges only over the full 50-problem test set.

### **Hypothesized Mechanisms**

Several non-mutually-exclusive mechanisms may explain this phenomenon:

1. **Thread Pool Scheduling Efficiency:** The system uses `ThreadPoolExecutor(max_workers=workers)`. With `workers=16` and `sandboxes=16`, the executor maintains balanced resource allocation for both compute threads and sandbox availability. With `workers=16` and `sandboxes=8`, potential thread-sandbox contention may arise: async operations (streaming callbacks, tool execution, result aggregation) compete for thread capacity while 8 threads remain blocked waiting for sandbox availability. The 2:1 ratio provides computational "slack" for better handling of concurrent async operations during inference.

2. **Queue Timing Edge Cases:** Although blocking should theoretically never occur with `attempts < workers`, timing race conditions exist: sandboxes may be mid-reset when new attempts launch, or `put()` operations may be delayed by thread scheduling. The 2× overprovision ensures the queue always contains available sandboxes, eliminating timeout exceptions and reducing acquisition latency variance.

3. **Statistical Accumulation:** The 10-problem reference set has insufficient statistical power to detect small per-problem effects. If each problem experiences a 2% performance degradation with 1:1 configuration, this manifests as ~0.2 problems on the reference set (undetectable) but ~1 problem on the 50-problem test set (observable as 40 → 39 or 42 → 41).

### **Practical Implications**

The 2× overprovision represents an empirically-validated system configuration that prioritizes robustness and performance stability over theoretical resource efficiency. The idle sandboxes consume system memory but provide measurable improvements in leaderboard performance. This configuration has proven stable across multiple notebook versions and represents a design pattern for competitive inference pipelines using stateful tool execution environments.

## **Key Technical Findings**

After the late Mk II previews did not match Mk I baseline performance, the following technical constraints were identified:

1. **Problem Timeouts:** A `base_problem_timeout` of 240 (seconds) and a `high_problem_timeout` of 900 or 960 (seconds) is the **optimal** configuration, since all 3 notebooks that used this configuration (Mk I V16, Mk III V5, Mk IV V1) scored $\ge 40 (41, 40, 42 respectively).
2. **Reference set mismatch:** Strong or neutral performance on the 10-problem reference set does not guarantee the same behavior on the 50-problem test set. Examples: BF16 KV Cache, Pass@12/Stop@6 Inference.
3. **Pass@K:** When performing inference on the reference set, pass@8 (and early stop at Parity 4\) is the optimal strategy that provides adequate coverage without aggregating erroneous answers. Pass@12 is feasible (if the context is \~65,000 tokens); however, it does not increase the likelihood of more sequences converging to the ground truth. Additionally, more time was needed for the 10 problems to be solved. If the test set contains a higher proportion of hard problems (that finish in 3-4 minutes rather than 1-2 minutes, as is the case with 40% of the reference set), many problems likely reach TLE and are never attempted. This likely contributed to sub-35 LB scores.
4. **KV Cache data type:** When setting the `--kv-cache-dtype` to `auto` (defaults to BF16), the concurrency ceiling at max context (65k tokens) drops to 4.92x (vs 9.62x for FP8 E4M3 at the same context). With the reference set, the reduced concurrency was **not** a problem, because most sequences finished before 30,000 tokens were reached (for problem no. 10, only 3-4 sequences surpassed 30k tokens and stopped at 60k). However, the LB score of 33 indicated that many problems likely need 40,000-60,000 tokens.
5. **Maximum context length:** When using 65,536 token context length, problem no. 10 produced \~2 (sometimes 3\) sequences that always returned NA; these sequences always stopped at 58k-62k tokens. Since the python output does not count into the `sequence length` column of the log dataframe, I assume these sequences reached 65k tokens and were terminated at the context limit. Increasing the context to 81,920 tokens eliminated these issues (with sequences going up to \~78,000 tokens). 81,920 token context length, coupled with FP8 E4M3 KV Cache, allows for a concurrency at max context of 7.75x (effectively all 8 sequences, since there is 0% chance all 8 go up to \~80,000 tokens).
6. **Sampling Parameters:** The default values of 1.0 for temperature and 0.02 for min\_p (top\_p and top\_k are not utilized) are optimal. Having tried different values for *temperature*, the runs were less stable (vis-a-vis a temperature of 1.0)
7. **Early Stopping:** For Pass@8, it is harder for 6 sequences to agree than for only 4 (before all 8 finish). Therefore, Stop@6 demands more solving time per problem.
8. **USER channel instruction:** This refers to the user prompt appended after the problem statement; performance on the reference set is better if this prompt includes the *Python* related instructions rather than *problem constraints* (`\boxed{}` constraint, \[0, 99999\] range).
9. **Prompt structure:** When I use succinct prompts (similar to preview-02-03 or the IMO Gold Medalist persona), the LB is stable (in the [36, 42] range). However, when I use the public notebook's prompts (which are overly verbose), the score drops to 33. The verbose prompts behave as high-variance configurations, occasionally scoring high but more often reducing stability. My pipeline prioritizes stability, targeting a robust floor score while maintaining a strong probability of crossing 40+. (See *Leaderboard Engineering* below for a detailed statistical breakdown).
10. **Number of Workers:** Every notebook that scores 40-42 uses **2x workers** relative to the number of attempts (e.g., 16 workers for 8 attempts). Notebooks using a 1:1 ratio (like Version 21 and Preview-02-05) exhibit lower scores (mid-30s). This suggests that a sandbox issue in a 1:1 setup can remove an attempt, whereas a redundant pool handles interruptions gracefully. The port-retry logic in Mk II further improves this stability.
11. **Persona Selection:** The comparison between "IMO Gold Medalist" and "World-class Competitor" favors **Gold Medalist** for stability:
  * Both personas have a maximum public test set score of 42/50.
  * *IMO Gold Medalist* range is [39, 42].
  * *world-class IMO competitor* range is [37, 42] (the 37 is because of Mk II V2).
  * Scores >= 40 are not reproducible with the competitor persona.
  * Due to the proximity in the scores of submitted notebooks, if we judge by natural flow and semantic weight, *IMO Gold Medalist* is more stable.
12. **Prefix Caching:** Disabling vLLM prefix caching (observed in Versions 13 and 14\) correlated with a drop in score to 37, compared to scores of 39-42 in versions where it was enabled. This suggests that the throughput or latency benefits of prefix caching are material to the pipeline's success within the 5-hour limit.
13. **Public Test set - score floor:** Even though this is not "technical", it is worth putting it here. When a notebook scores 36 or 37 (when the typical range is 38-42), the run should be treated as a lower-tail stochastic outcome and handled carefully in conclusions/inferences. **Run-to-run variance is a material factor in AIMO3**.
14. **Jupyter Timeout:** Having added the `debug_mode` flag in `version-13` (and its 5 previews), *timeout errors* were the most common by far; my original thought was that doubling the `jupyter_timeout` from 6 to 12 (seconds) would eliminate most timeout errors. However, this was **not** the case, as there was no decrease in timeout errors for `jupyter_timeout = 12`; I will assume this is not because the execution window is not sufficient (i.e. code needs 10 seconds but is terminated at 6 seconds) but rather because of code paths that do not produce a result or an error.

## **Leaderboard Engineering**

### Total Versions and Previews
Across all development phases (Mk I, Mk II, Mk III, Mk IV and their respective previews), a total of **68** notebook versions were created:

| Series | Count |
| :--- | :--- |
| Mk I | 21 |
| Mk II Previews | 7 |
| Mk II | 17 |
| Mk III Previews | 9 |
| Mk III | 6 |
| Mk IV | 8 |
| **Total** | **68** |

### Score Distribution
The occurrences of each score on the public leaderboard across all 68 notebooks. The percentages exclude the **5** notebooks that scored NA (making the evaluated total **63** notebooks):

| Score | Occurrences | Percentage |
| :---: | :---: | :---: |
| 42 | 3 | 4.76% |
| 41 | 6 | 9.52% |
| 40 | 6 | 9.52% |
| 39 | 11 | 17.46% |
| 38 | 13 | 20.63% |
| 37 | 9 | 14.29% |
| 36 | 4 | 6.35% |
| 35 | 2 | 3.17% |
| 34 | 6 | 9.52% |
| 33 | 2 | 3.17% |
| 32 | 1 | 1.59% |
| NA | 5 | - |
