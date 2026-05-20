## Preview-02-23

**Mk III Baseline — vLLM 0.15.1 Migration and Prompt Architecture Consolidation.**

The initial Mk III release, forked from Mk II Version 18 with a vLLM runtime upgrade and significant architectural simplifications:

* **vLLM Upgrade:** Migrated from vLLM 0.11.2 (Mk II) to **vLLM 0.15.1**. Maximum concurrency for 98,304 tokens per request: **7.43×** (up from 6.51× on vLLM 0.11.2).
* **Environment variables:** Removed `VLLM_FLASH_ATTN_VERSION='3'` and `VLLM_ATTENTION_BACKEND='FLASH_ATTN'` (adopted in Mk II from Preview-02-05). Retained `TRANSFORMERS_NO_TF='1'`, `TRANSFORMERS_NO_FLAX='1'`, `TIKTOKEN_ENCODINGS_BASE` (local tiktoken cache), and `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'`. Retained `VLLM_DO_NOT_TRACK`, `VLLM_NO_USAGE_STATS`, `VLLM_LOG_STATS_INTERVAL=60`, `CUDA_VISIBLE_DEVICES='0'`, `TOKENIZERS_PARALLELISM='false'`, `TRITON_PTXAS_PATH`.
* **Setup:** `set_env()` installs only `vllm` (same as Mk II). Uninstalls `keras`, `matplotlib`, `scikit-learn`, `tensorflow`.
* **Prompt architecture — merged back to single-role:**
  * Mk II V18 split the system prompt into `system_prompt` (persona only) + `developer_prompt` (task constraints) with a three-message template `[system, developer, user]`.
  * Mk III V1 merges them back into a unified `system_prompt` with a two-message template `[system, user]`. `DeveloperContent` and `developer_prompt` are removed entirely (imitating every version of the Mk II series before Version 18)
* **System Prompt:** `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'`
* **Tool Prompt:** `'Use this tool to execute Python code. The environment is a stateful Jupyter notebook. The tool returns the execution output or times out after 6 seconds.'` (inherited from Mk II V18).
* **User Prompt:** ``'Use `math` and `sympy` to solve the problem, supported by `collections`, `itertools`, `fractions` and `decimal`.'`` (inherited from Mk II V1).
* **User input composition:** `user_prompt = f'{problem} {self.cfg.user_prompt}'`.
* **Harmony metadata removed:** Removed `conversation_start_date` and `knowledge_cutoff` fields (introduced in Mk II V18). `get_system_content()` and `apply_chat_template()` no longer accept these parameters.
* **`solve_problem` output format:** Changed from `f'({id_value}) {problem}'` to `f'{id_value} | {problem}'`.
* **Sandbox architecture:** Identical to Mk II V15+ (port management, kernel retry, output truncation, `_kill_descendants`, enhanced `reset()`). `AIMO3Sandbox.__init__` takes 11 parameters (adds `custom_errors` — was passed separately in Mk II).
* **Sandbox imports:** `sys`, `json`, `math`, `numpy`, `scipy`, `sympy`, `mpmath` (`dps=64`), `decimal`, `fractions`, `functools`, `itertools`, `collections`, `numpy as np`, `sympy as sp`, `Fraction`, `Decimal`, `getcontext` (with `getcontext().prec = 64`). Identical to Mk II V13+.
* **`_format_error()`:** Identical to Mk II V15's implementation — dynamic `self._custom_errors` dictionary with substring and regex-tuple triggers, `ipykernel_` → `Cell` sanitisation, and `<cell line:>` regex cleanup scoped to kernel frames.
* **`custom_errors` dictionary:** Same three entries as Mk II V15: 4300-digit `ValueError` guidance, `sympy.valuation` → `sympy.multiplicity` correction, and `NameError` regex trap. The `ModuleNotFoundError` intercept (added in Mk II V18) is removed.
* **`AIMO3Tool`:** Takes `tool_prompt` and `sandbox`. No `zmq_attempts`/`zmq_timeout`. Unchanged from Mk II.
* **Answer detection:** Only `boxed_pattern` regex (`r'\\boxed\s*\{\s*([0-9,_]+)\s*\}'`). No `text_pattern`. `rfind('}')` approach in `final` channel handler. Unchanged from Mk II.
* **Streaming:** `_stream_completion()` returns `tuple[list[int], list[str], float, int]` — inline entropy accumulation (inherited from Mk II V3+). No real-time answer detection.
* **Scoring/voting:** IEW weight formula: `weight = 1 / max(entropy, entropy_floor)`, where `entropy` is the arithmetic mean of per-token Shannon entropies across all generation turns:

$$H = \frac{1}{N} \sum_{t=1}^{N} H_t, \qquad w = \frac{1}{\max(H,\, \varepsilon_{\text{floor}})}$$

  No Python-corroborated boosting (removed in Mk II V10, retained as tracking only). `from_python` flag is still computed and logged.
* **Preload weights:** `ThreadPoolExecutor()` (default workers).
* **Kernel initialization:** `ThreadPoolExecutor(max_workers=self.cfg.workers)`.
* **`_process_attempt`:** Uses `hashlib.sha256`-based deterministic seeding (inherited from Mk II V3+). Signature simplified from Mk II V18 — removed `system_prompt`, `developer_prompt`, `conversation_start_date`, `knowledge_cutoff` parameters; accepts `user_prompt`, `attempt_index`, `stop_event`, `deadline`.
* **`_handle_tool_call`:** Returns `tuple[int, list[Message], str]` (inherited from Mk II V9+). Error detection: `'Execution timed out' in response_text or 'Error:' in response_text`.
* **Config:**

| Parameter | Value |
| :--- | :--- |
| `kv_cache_dtype` | `fp8_e4m3` |
| `high_problem_timeout` | 996 |
| `base_problem_timeout` | 276 |
| `notebook_limit` | 17400 |
| `server_timeout` | 180 |
| `session_timeout` | 1020 |
| `jupyter_timeout` | 6 |
| `sandbox_timeout` | 3 |
| `backoff_delay` | 0.5 |
| `iopub_timeout` | 1.0 |
| `kernel_attempts` | 3 |
| `port_increment` | 10 |
| `port_attempts` | 10 |
| `port_timeout` | 300 |
| `max_port` | 65535 |
| `min_port` | 50000 |
| `stream_interval` | 200 |
| `context_tokens` | 98304 |
| `batched_tokens` | 2048 |
| `entropy_floor` | 0.1 |
| `buffer_tokens` | 1024 |
| `problem_count` | 50 |
| `output_chars` | 2048 |
| `capture_size` | 64 |
| `top_logprobs` | 5 |
| `batch_size` | 64 |
| `early_stop` | 4 |
| `attempts` | 8 |
| `workers` | 16 |
| `turns` | 192 |
| `seed` | 42 |
| `gpu_memory_utilization` | 0.99 |
| `temperature` | 1.0 |
| `min_p` | 0.02 |

---

## Preview-02-24

**Explicit attention and compilation configuration.**

* **New import:** `import json`.
* **New config parameter `attention_config`:** Dictionary passed to vLLM's `--attention-config` flag, replacing the environment variables removed in Version 1 with explicit engine-level control:
  * `'backend': 'FLASH_ATTN'`, `'flash_attn_version': 3` — restores FA3 selection (previously set via `VLLM_ATTENTION_BACKEND` and `VLLM_FLASH_ATTN_VERSION` environment variables in Mk II).
  * `'use_prefill_decode_attention': True` — enables split prefill/decode attention scheduling.
  * `'flash_attn_max_num_splits_for_cuda_graph': 64` — caps Flash Attention CUDA graph splits.
  * `'use_cudnn_prefill': False` — disables cuDNN-based prefill.
  * `'use_trtllm_ragged_deepseek_prefill': True` — enables TensorRT-LLM ragged prefill for DeepSeek architecture.
  * `'use_trtllm_attention': False` — disables full TensorRT-LLM attention replacement.
  * `'disable_flashinfer_prefill': True` — disables FlashInfer prefill path.
  * `'disable_flashinfer_q_quantization': True` — disables FlashInfer query quantization.
* **New config parameter `compilation_config`:** Dictionary passed to vLLM's `--compilation-config` flag, replacing the scalar `capture_size` parameter:
  * `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32, 64]` — explicit list of batch sizes to capture CUDA graphs for (previously inferred by vLLM up to `--max-cudagraph-capture-size`).
  * `'cudagraph_specialize_lora': False` — disables LoRA-specialized CUDA graphs.
  * `'max_cudagraph_capture_size': 64` — equivalent to the removed `capture_size` parameter.
* **Removed config parameter:** `capture_size` (was 64) — its function is subsumed by `compilation_config['max_cudagraph_capture_size']`.
* **vLLM server flags:** Replaced `--max-cudagraph-capture-size` (set to `capture_size`) with `--attention-config` (JSON-serialized `attention_config` dict) and `--compilation-config` (JSON-serialized `compilation_config` dict).
* Everything else is identical to Preview-02-23.

---

## Preview-02-25

**Attention and compilation configuration corrections from GPT-OSS model card analysis.**

* **`attention_config['use_trtllm_ragged_deepseek_prefill']`: `True` → `False`.** The GPT-OSS-120B model card (Section 2.2) confirms that GPT-OSS attention is an OpenAI-lineage alternating banded-window / fully-dense pattern derived from GPT-3 sparse transformers (Child et al. 2019). It is architecturally unrelated to DeepSeek's Multi-head Latent Attention (MLA), which uses compressed low-rank KV projections that produce ragged tensor layouts during prefill. The TRT-LLM ragged DeepSeek prefill kernel targets those non-standard shapes; applying it to GPT-OSS's standard GQA (8 KV heads, dim 64) either silently falls back to a generic path or selects an incompatible code path.
* **`compilation_config['cudagraph_copy_inputs']`: `False` → `True`.** GPT-OSS-120B uses top-4 dynamic expert routing — the router dispatches different tokens to different expert buffers on every forward pass. With `copy_inputs=False`, the CUDA graph replay can encounter input tensor aliasing bugs where the graph operates on stale or overwritten memory if the runtime reuses underlying storage between decode steps. Explicitly copying the small input tensors (token IDs and positions) eliminates this risk at negligible overhead.
* Everything else is identical to Preview-02-24.

---

## Preview-02-26

**EMA entropy scoring — incremental per-token entropy accumulation.**

* **New config parameter `alpha`:** `0.9999`. Controls the decay rate of the Exponential Moving Average (EMA) used to score the entropy of each generated sequence. At `α = 0.9999` the EMA half-life is ≈ 6,931 tokens, making it appropriate for the very long sequences (up to 20,000+ tokens) produced by GPT-OSS-120B — recent, confident reasoning near the end of the derivation dominates the score while early exploration is up-weighted only modestly.
* **`_stream_completion()` — EMA entropy accumulation:**
  * Signature change: accepts a new `ema_entropy: float` parameter (the running EMA state carried in from the caller) and the return type narrows from `tuple[list[int], list[str], float, int]` to `tuple[list[int], list[str], float]`.
  * Per-token update: `ema_entropy = cfg.alpha * ema_entropy + (1 - cfg.alpha) * token_entropy` replaces the previous `cumulative_entropy += token_entropy` / `entropy_token_count += 1` accumulation. The two local variables `cumulative_entropy` and `entropy_token_count` are removed.
  * Early-exit path returns the current `ema_entropy` as-is (previously returned `0.0, 0`).
* **`_process_attempt()` — EMA state threading:**
  * `total_entropy: float` and `total_entropy_tokens: int` are replaced by a single `ema_entropy: float = 0.0` that is passed into and returned from every `_stream_completion` call across all turns, so the EMA is continuous across tool-call round-trips.
  * `response_tokens` accumulation no longer includes `chunk_tokens` (the count variable no longer exists).
  * Final score: `final_entropy = ema_entropy if ema_entropy > 0.0 else float('inf')` replaces the conditional `total_entropy / total_entropy_tokens` mean. The logged key `'Entropy'` receives `final_entropy`.
* **IEW formula comparison:**

  | | Version 1–3 (arithmetic mean) | Version 4 (EMA) |
  | :--- | :--- | :--- |
  | **Entropy estimate** | $H = \frac{1}{N}\sum_{t=1}^{N} H_t$ | $\hat{H}_t = \alpha\,\hat{H}_{t-1} + (1-\alpha)\,H_t$ |
  | **IEW weight** | $w = \dfrac{1}{\max(H,\,\varepsilon_{\text{floor}})}$ | $w = \dfrac{1}{\max(\hat{H}_N,\,\varepsilon_{\text{floor}})}$ |

  Where $H_t = -\sum_v p_v \log_2 p_v$ is the per-token Shannon entropy over the top-$k$ logprob vocabulary, $N$ is the total number of generated tokens, and $\varepsilon_{\text{floor}} = 0.1$ is the entropy floor that prevents division-by-near-zero.

---

## Preview-02-27

**vLLM 0.16.0 Migration, Persona ablation, EMA revert, CUDA graph and sandbox cleanup.**

*   **vLLM Upgrade:** Migrated to **vLLM 0.16.0**. Maximum concurrency for 98,304 tokens per request: **6.51×**.
*   **Pip packages — expanded uninstall list:** Added `tensorflow-io`, `albumentations`, `lightgbm`, `catboost`, `xgboost` to the `pip uninstall` list alongside the existing `keras`, `matplotlib`, `scikit-learn`, `tensorflow`.
*   **Utils path updated:** `set_env(wheels_path=...)` changed from `/kaggle/input/aimo-3-utils-mk-ii/wheels` to `/kaggle/input/notebooks/andreasbis/aimo-3-utils-mk-iii/wheels`.
*   **Environment variables:**
    *   Added: `VLLM_MXFP4_USE_MARLIN='1'`.
    *   `TIKTOKEN_ENCODINGS_BASE` path updated from `/kaggle/input/aimo-3-utils-mk-ii/tiktoken_encodings` to `/kaggle/input/notebooks/andreasbis/aimo-3-utils-mk-iii/tiktoken_encodings`.
*   **System prompt — persona prefix dropped:**
    *   From: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'`
    *   To: `'The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'`
    *   The persona identity is removed entirely; the system prompt now carries only task constraints.
*   **Wording change:** `'Return the final answer'` → `'Place your final answer'` (in system prompt).
*   **`get_system_content()` signature simplified:** Removed `conversation_start_date` and `knowledge_cutoff` parameters. No longer calls `.with_conversation_start_date()` or `.with_knowledge_cutoff()` on the `SystemContent` builder chain.
*   **`apply_chat_template()` — reverted to two-message format:** Removed `developer_prompt`, `conversation_start_date`, `knowledge_cutoff` parameters.
*   **Custom errors — ModuleNotFoundError intercept removed:** Removed the regex trap for `ModuleNotFoundError`. `custom_errors` now has three entries (down from four).
*   **Entropy scoring — reverted from EMA to arithmetic mean:**
    *   Removed config parameter `alpha` (was `0.9999` in Preview-02-26).
    *   `_stream_completion()` signature reverted: removed `ema_entropy` parameter, return type back to `tuple[list[int], list[str], float, int]`. Accumulation reverted to `cumulative_entropy += token_entropy` / `entropy_token_count += 1` (the two local variables removed in Preview-02-26 are restored).
    *   `_process_attempt()` uses `total_entropy: float` and `total_entropy_tokens: int` instead of the single `ema_entropy: float`. `response_tokens` accumulation re-includes `chunk_tokens`.
    *   Final score: `mean_entropy = total_entropy / total_entropy_tokens` (conditional, `inf` if zero). The logged key `'Entropy'` receives `mean_entropy`.
    *   Net effect: reverts the IEW weight formula from the EMA variant (Preview-02-26) back to the arithmetic mean variant (Preview-02-23 through Preview-02-25).
*   **`compilation_config` — removed `cudagraph_copy_inputs`:** The `'cudagraph_copy_inputs': True` entry (added in Preview-02-25) is removed. The remaining keys (`cudagraph_capture_sizes`, `cudagraph_specialize_lora`, `max_cudagraph_capture_size`) are unchanged.
*   **Model path updated:** `/kaggle/input/gpt-oss-120b/transformers/default/1` → `/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1`.
*   **New config parameter `load_format`:** `'safetensors'` — passed to the new `--load-format` vLLM server flag.
*   **Sandbox kernel initialization — removed `sys` and `json` imports:** Removed `import sys` and `import json` from both the initial kernel startup block and the `reset()` method.
*   **Error handling in `AIMO3Tool` — format change:** `TimeoutError` output changed from `f'[ERROR] {exc}'` to `str(exc)` (drops the `[ERROR]` prefix).
*   **Preload weights:** `ThreadPoolExecutor()` changed to `ThreadPoolExecutor(max_workers=self.cfg.workers)` — now uses explicit worker count for weight preloading.
*   **vLLM server flags:**
    *   Added: `--load-format`, `--enable-chunked-prefill`.
    *   Reordered: `--enable-prefix-caching` moved after `--compilation-config`; `--async-scheduling` and `--disable-log-stats` moved to end.
*   **`solve_problem` print format changed:** From: `f'\n({id_value}) {problem}\n'` To: `f'\n{id_value} | {problem}\n'`.
*   **`_process_attempt` signature simplified:** Removed `system_prompt`, `developer_prompt`, `conversation_start_date`, `knowledge_cutoff` parameters. Now reads `system_prompt` directly from `self.cfg.system_prompt` inside the method body.
*   **`solve_problem` — task dispatch simplified:** Removed the `tasks` list pre-construction that packed `(system_prompt, developer_prompt, conversation_start_date, knowledge_cutoff, attempt_index)` tuples. Loop now iterates `for attempt_index in range(self.cfg.attempts)` directly, submitting `self._process_attempt` with only `user_prompt` and `attempt_index` (plus shared args).
*   Everything else is identical to Preview-02-26.

---

## Preview-02-28

**Sandbox CPU isolation and kernel priority reduction.**

* **Sandbox environment variables — thread pinning:**
  * Added `OPENBLAS_NUM_THREADS='1'`, `NUMEXPR_NUM_THREADS='1'`, `MKL_NUM_THREADS='1'`, `OMP_NUM_THREADS='1'` to the kernel subprocess environment. Restricts all linear algebra backends (OpenBLAS, NumExpr, MKL, OpenMP) to a single thread per kernel, preventing thread explosion across 16 concurrent sandbox workers.
* **Sandbox kernel initialization — process deprioritisation:**
  * Added `import os` and `os.nice(10)` to the initial kernel startup block. Lowers the scheduling priority of each sandbox kernel process so that vLLM inference threads receive preferential CPU time. Applied only at kernel creation (not in `reset()`), since the nice value persists across IPython resets.
* Everything else is identical to Preview-02-27.

---

## Preview-03-01

**System prompt persona hint, error-capped sandbox loop.**

* **System prompt — persona hint restored:**
  * Prepended `'Solve this International Mathematical Olympiad (IMO) problem. '` to `system_prompt`. The full prompt is now `'Solve this International Mathematical Olympiad (IMO) problem. The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'` — reintroduces a task-identity cue without reverting to the Mk II persona format.
* **New config parameter `max_errors`:** `16` — caps the number of Python execution errors tolerated per problem before the sequence is terminated early.
* **`_process_attempt` — early exit on excessive errors:**
  * Added `if python_errors > self.cfg.max_errors: break` immediately after `python_errors += errors`. When the running error count exceeds `max_errors`, the sequence is terminated, preventing further wasted inference on a failed attempt.
* Everything else is identical to Preview-02-28.

---

## Preview-03-02

**System/developer role separation, Harmony metadata restoration, tool prompt refinement.**

* **Prompt architecture — system/developer role separation (restored and corrected):**
  * The Mk III preview series (Preview-02-23 through Preview-03-01) used a unified `system_prompt` with a two-message template `[system, user]`.
  * Preview-03-02 splits the prompt into `system_prompt` (persona only) and `developer_prompt` (task constraints) with a three-message template `[system, developer, user]`, matching the intended Harmony instruction hierarchy. This corrects the confounded Mk II V18 experiment: the prior failure of the developer channel was due to simultaneous removal of the `user_prompt` library instructions and use of uninformative developer content — not the channel split itself.
  * `system_prompt` now contains only the persona identity: `'You are an International Mathematical Olympiad (IMO) Gold Medalist.'`
  * New field `developer_prompt` carries the task constraints: `'The answer to the problem is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'` Wording change from Mk II V18: `'The answer is'` → `'The answer to the problem is'` (three words added to reduce ambiguity).
* **Harmony protocol — `DeveloperContent` class:**
  * `DeveloperContent` is now imported from `openai_harmony` alongside the existing imports.
  * `AIMO3Template` gains a new `get_developer_content(developer_prompt)` method returning `DeveloperContent.new().with_instructions(developer_prompt)`.
  * `apply_chat_template()` constructs three messages: `[system_message, developer_message, user_message]`. `developer_message` is created via `Message.from_role_and_content(Role.DEVELOPER, developer_content)`.
  * `apply_chat_template()` signature: `(system_prompt, conversation_start_date, knowledge_cutoff, developer_prompt, user_prompt, tool_config)`.
* **Harmony protocol — system content metadata (restored):**
  * New config fields: `conversation_start_date = '2025-11-20'` and `knowledge_cutoff = '2024-06'`.
  * `get_system_content()` calls `.with_conversation_start_date(conversation_start_date)` and `.with_knowledge_cutoff(knowledge_cutoff)` on the `SystemContent` builder chain. These were removed in Preview-02-27; their restoration here is safe because the prior notebooks that scored below 38 with these fields also lacked the `user_prompt` library instructions and used uninformative developer content — the dates were not the causal factor.
  * `get_system_content()` signature: `(system_prompt, conversation_start_date, knowledge_cutoff, tool_config)`.
* **Tool prompt — wording change:**
  * From: `'The environment is a stateful Jupyter notebook.'`
  * To: `'The tool executes code in a stateful Jupyter notebook environment.'`
* **`AIMO3Tool` — lock removed:** `self._execution_lock = threading.Lock()` and the corresponding `with self._execution_lock:` block in `process_sync_plus` are removed. Each attempt instantiates its own local `AIMO3Tool` wrapping its own dedicated sandbox, so the lock was never contended and is dead code.
* **`_process_attempt` — template call updated:** The `apply_chat_template()` call inside `_process_attempt` now passes `self.cfg.conversation_start_date`, `self.cfg.knowledge_cutoff`, and `self.cfg.developer_prompt` as additional arguments (matching the expanded signature above).
* **Local gateway path updated:** `'/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv'` → `'/kaggle/input/competitions/ai-mathematical-olympiad-progress-prize-3/reference.csv'`.
* Everything else is identical to Preview-03-01.

---

## Preview-03-03

**Developer prompt simplification, CUDA graph split reduction, error tolerance increase.**

* **`developer_prompt` — wording simplification:**
  * From: `'The answer to the problem is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'`
  * To: `'The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'`
  * Reverts the three-word addition (`'to the problem'`) introduced in Preview-03-02 back to the original wording used in the system prompt through Preview-02-23 to Preview-03-01.
* **`attention_config['flash_attn_max_num_splits_for_cuda_graph']`: `64` → `32`.** Halves the maximum number of Flash Attention splits captured in CUDA graphs, reducing graph memory overhead.
* **`max_errors`: `16` → `32`.** Doubles the per-problem Python execution error tolerance before early termination, allowing more retry opportunities for problems that produce transient errors.
* Everything else is identical to Preview-03-02.

---

## Mk III Public Leaderboard Performance

| Version | LB Score |
| :--- | :--- |
| Preview-02-23 | 39 |
| Preview-02-24 | 38 |
| Preview-02-25 | 36 |
| Preview-02-26 | NA |
| Preview-02-27 | 35 |
| Preview-02-28 | 39 |
| Preview-03-01 | 34 |
| Preview-03-02 | 37 |
| Preview-03-03 | 37 |
