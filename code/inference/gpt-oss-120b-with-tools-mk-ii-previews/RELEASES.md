## Preview-01-29

**Baseline Mk II Architecture.**

The initial Mk II preview, forked from the Mk I lineage with significant structural differences:

* **Setup:** Unified `set_env()` that combines pip uninstall and pip install in one function. Installs only `vllm` (no `unsloth`, `trl`, or `openai_harmony` — imported from the environment).
* **Environment variables:** Adds `VLLM_DO_NOT_TRACK`, `VLLM_NO_USAGE_STATS`, `VLLM_LOG_STATS_INTERVAL=60` (absent in Mk I).
* **System Prompt:** ``'You are an International Mathematical Olympiad (IMO) Gold Medalist. You treat every problem as a computational challenge to be solved with Python; you rely on `math` and `sympy` to verify conjectures and perform precise arithmetic; you test small inputs, write scripts to validate candidate functions, and apply modular arithmetic to prune the search space.'``
* **Tool Prompt:** ``'Use this tool to execute Python code. The environment is a stateful Jupyter notebook. Use `print()` to output results.'``
* **User Prompt:** `'The answer is a non-negative integer between 0 and 99999. Place the final answer inside \\boxed{}.'`
* **User input composition:** `user_text = f'{problem} {self.cfg.user_prompt}'` — user prompt is appended to the problem text.
* **`solve_problem` signature:** `solve_problem(self, problem: str)`. Prints `'Problem: {problem}'`.
* **Sandbox:** `AIMO3Sandbox.__init__` takes `timeout`, `zmq_attempts`, `zmq_timeout`. Has ZMQ retry logic with configurable attempts and sleep between retries.
* **Sandbox imports:** `math`, `sympy`, `mpmath` (`dps=64`), `itertools`, `collections`. No `numpy`.
* **`AIMO3Tool`:** Takes `zmq_attempts`/`zmq_timeout` parameters. No `close()` or `__del__`.
* **Answer detection:** Two pre-compiled regex patterns: `boxed_pattern` (`\\boxed{}`) and `text_pattern` (greedy semantic detection: `(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:|equals)?\s*([0-9,]+)`).
* **Streaming:** `_stream_completion()` returns `tuple[list[int], list[str], int | None]` — includes real-time answer detection during streaming. Checks `'}' in new_text` and scans the last `search_tokens` chunks during the stream.
* **`_handle_tool_call()`:** Multi-line signature (parameters each on their own line).
* **`_process_attempt`:** First parameter is `user_text`. Returns timing info (`'Time': time_duration`). Uses `hash((self.cfg.seed, attempt_index, 'salt')) % (2**31)` for attempt seed.
* **Scoring/voting:** Weight formula: `max(calls - errors, 1) / max(entropy, entropy_floor)`. Uses Python call/error counts as a quality signal.
* **Preload weights:** `ThreadPoolExecutor(max_workers=self.cfg.workers)`.
* **`early_stop` logic:** Simplified — just `if counts and counts[0][1] >= self.cfg.early_stop:` followed by `stop_event.set()` then `break`. No future cancellation.
* **Config:**

| Parameter | Value |
| :--- | :--- |
| `kv_cache_dtype` | `fp8_e4m3` |
| `high_problem_timeout` | 840 |
| `base_problem_timeout` | 276 |
| `session_timeout` | 900 |
| `jupyter_timeout` | 6 |
| `sandbox_timeout` | 3 |
| `zmq_attempts` | 3 |
| `zmq_timeout` | 1 |
| `context_tokens` | 65536 |
| `entropy_floor` | 0.1 |
| `buffer_tokens` | 512 |
| `search_tokens` | 32 |
| `problem_count` | 50 |
| `capture_size` | 256 |
| `top_logprobs` | 5 |
| `batch_size` | 256 |
| `early_stop` | 6 |
| `attempts` | 8 |
| `workers` | 8 |
| `seed` | 42 |
| `gpu_memory_utilization` | 0.97 |
| `temperature` | 1.0 |
| `min_p` | 0.02 |

* **vLLM Server flags:** `--async-scheduling`, `--disable-log-stats`, `--enable-prefix-caching`, `--max-cudagraph-capture-size` (set to `capture_size`). No `--max-num-batched-tokens`.

---

## Preview-01-30

**Prompt architecture reorganization.**

* **System prompt:** Condensed from the verbose computational-strategy persona to a shorter version. Extracted the answer format requirements from `user_prompt` and merged them into `system_prompt`:
  * From: ``'You are an International Mathematical Olympiad (IMO) Gold Medalist. You treat every problem as a computational challenge to be solved with Python; you rely on `math` and `sympy` to verify conjectures and perform precise arithmetic; you test small inputs, write scripts to validate candidate functions, and apply modular arithmetic to prune the search space.'``
  * To: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is a non-negative integer between 0 and 99999. Place the final answer inside \\boxed{}.'`
* **User prompt:** Replaced answer-format instructions with a library-usage directive:
  * From: `'The answer is a non-negative integer between 0 and 99999. Place the final answer inside \\boxed{}.'`
  * To: ``'Use `math` and `sympy` to solve the problem.'``
* Everything else is identical to Preview-01-29.

---

## Preview-01-31

**Aggressive prompt minimization.**

* **System prompt:** Stripped all persona-related content. Removed `'You are an International Mathematical Olympiad (IMO) Gold Medalist.'` prefix. Changed `'non-negative integer'` to `'integer'`:
  * From: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is a non-negative integer between 0 and 99999. Place the final answer inside \\boxed{}.'`
  * To: `'The answer is an integer between 0 and 99999. Place the final answer inside \\boxed{}.'`
* **User prompt:** Eliminated entirely — `user_prompt` field removed from `CFG`.
* **User input composition:** Removed `user_text = f'{problem} {self.cfg.user_prompt}'`. Now passes bare `problem` text directly.
* **`_process_attempt`:** First parameter renamed from `user_text` back to `problem`.
* **`solve_problem()`:** Passes `problem` instead of `user_text` to `_process_attempt`.
* Everything else is identical to Preview-01-30.

---

## Preview-02-01

**Resource scaling, scoring simplification, and V11 prompt restoration.**

* **System prompt:** Reverted to the canonical V1/V11 Mk I system prompt:
  * `'You are a world-class International Mathematical Olympiad (IMO) competitor. The final answer must be a non-negative integer between 0 and 99999. You must place the final integer answer inside \\boxed{}.'`
* **Tool prompt:** Changed ``'Use `print()` to output results.'`` to `'You must use print() to output results.'` (imperative, removed backticks around `print()`).
* **User prompt:** Re-introduced `user_prompt` field: ``'You have access to `math`, `numpy` and `sympy` to solve the problem.'``
* **User input composition:** Re-introduced `user_text = f'{problem} {self.cfg.user_prompt}'`.
* **`_process_attempt`:** First parameter renamed from `problem` back to `user_text`.
* **`solve_problem` signature:** Changed from `solve_problem(self, problem: str)` to `solve_problem(self, id_value: int, problem: str)`. Prints `f'({id_value}) {problem}'` instead of `f'Problem: {problem}'`. The `predict()` function now passes `id_value` to `solve_problem()`.
* **Sandbox imports:** Added `import numpy` (inserted before `sympy` in the import order — both `__init__` and `reset()`).
* **Config changes:**
  * `capture_size`: 256 → **64**.
  * `batch_size`: 256 → **64**.
  * `attempts`: 8 → **12**.
  * `workers`: 8 → **12**.
  * `gpu_memory_utilization`: 0.97 → **0.99**.
* **New config field:** `batched_tokens = 1024`.
* **vLLM server command:** Added `--max-num-batched-tokens` (set to `batched_tokens`).
* **`_handle_tool_call()` signature:** Collapsed from multi-line to single-line parameter list. No logic change.
* **Preload weights:** Changed from `ThreadPoolExecutor(max_workers=self.cfg.workers)` to `ThreadPoolExecutor()` (default max workers).

---

## Preview-02-02

**BF16 KV cache experiment, output truncation, and concurrency reduction.**

* **System prompt:** Softened wording:
  * From: `'...The final answer must be a non-negative integer between 0 and 99999. You must place the final integer answer inside \\boxed{}.'`
  * To: `'...The answer is an integer between 0 and 99999. Place the final answer inside \\boxed{}.'`
* **Tool prompt:** Reverted to the 01-29 wording: ``'Use `print()` to output results.'`` (restored backticks, removed "You must").
* **KV cache:** Changed `kv_cache_dtype` from `'fp8_e4m3'` to `'auto'` (effectively BF16 on the target hardware).
* **New config field:** `output_chars = 2048` — sandbox output truncation limit.
* **Config changes:**
  * `early_stop`: 6 → **4**.
  * `attempts`: 12 → **8**.
  * `workers`: 12 → **8**.
* **New method `AIMO3Sandbox._truncate_output()`:** If output exceeds `output_chars`, keeps the first and last `output_chars // 2` characters with a `'... [Truncated N characters] ...'` message in between. Applied to both `stdout` and `stderr` in `execute()`.
* **`AIMO3Sandbox.__init__`:** Added `output_chars` parameter.
* **`AIMO3Tool.__init__`:** Added `output_chars` parameter. Passed through to sandbox creation.
* **`_initialize_kernels`** and **`_process_attempt`:** Pass `output_chars` when constructing sandboxes and tools.
* **`_select_answer()` scoring formula simplified:** Removed the Python call/error quality signal.
  * From: `weight = max(calls - errors, 1) / max(entropy, self.cfg.entropy_floor)`
  * To: `weight = 1 / max(entropy, self.cfg.entropy_floor)`
* **Preload weights:** Already uses `ThreadPoolExecutor()` (from 02-01).
* **Minor whitespace:** Trailing whitespace removed after `_scan_for_answer`, and a blank line added in `_compute_mean_entropy` loop.

---

## Preview-02-03

**FP8 reversion, context expansion, and streaming architecture overhaul.**

* **KV cache:** Reverted `kv_cache_dtype` from `'auto'` back to `'fp8_e4m3'`.
* **Config changes:**
  * `high_problem_timeout`: 840 → **996**.
  * `session_timeout`: 900 → **1020**.
  * `context_tokens`: 65536 → **81920**.
  * `batched_tokens`: 1024 → **2048**.
  * `buffer_tokens`: 512 → **1024**.
  * `output_chars`: 2048 → **4096**.
* **Removed config field:** `search_tokens` deleted entirely.
* **Semantic detection eliminated:** Removed `text_pattern` regex (`(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:|equals)?\s*([0-9,]+)`) from `__init__`. Only `boxed_pattern` remains.
* **`_scan_for_answer()`:** Removed the `text_pattern` fallback branch. Only `\\boxed{}` matching survives.
* **Streaming overhaul — `_stream_completion()` return type changed:**
  * From: `tuple[list[int], list[str], int | None]`
  * To: `tuple[list[int], list[str]]`
  * **Removed real-time answer detection during streaming:** The `if '}' in new_text:` block that checked for `\\boxed{}` during streaming was deleted. Streams now run to natural completion without mid-stream answer scanning.
  * `final_answer` local variable removed from the method.
  * If `max_tokens < buffer_tokens`, returns `([], [])` instead of `([], [], None)`.
* **`_process_attempt` updated:**
  * Unpacks `_stream_completion` result as `token_buffer, text_chunks` (no third element).
  * Removed the `if answer is not None: final_answer = answer; break` block after streaming.
* **Answer extraction refined — `rfind` approach:**
  * In the `last_message.channel == 'final'` block, instead of `final_answer = self._scan_for_answer(answer_text)`, now uses:
    ```python
    last_brace_index = answer_text.rfind('}')
    if last_brace_index != -1:
        search_text = answer_text[:last_brace_index + 1]
        final_answer = self._scan_for_answer(search_text)
    ```
  * This targets the last occurrence of `}`, preventing false matches from intermediate `\\boxed{}` tokens in multi-step reasoning.
* **Sandbox warning message:** Changed `'[WARN] No output. Use print() to see results.'` to ``'[WARNING] No output. Use `print()` to output results.'``
* **Minor whitespace:** Trailing spaces on some lines within `_process_attempt`.

---

## Preview-02-05

> **Note:** Preview-02-04 does not exist.

**"Elite Problem Solver" prompt adoption and Flash Attention 3.**

* **New environment variables:**
  * `VLLM_FLASH_ATTN_VERSION = '3'` — enables Flash Attention v3.
  * `VLLM_ATTENTION_BACKEND = 'FLASH_ATTN'` — forces Flash Attention backend.

* **System prompt — complete rewrite to "Elite Problem Solver" framework:**
  * From: `'You are a world-class International Mathematical Olympiad (IMO) competitor. The answer is an integer between 0 and 99999. Place the final answer inside \\boxed{}.'`
  * To: A structured multi-section prompt with sections: `# Problem-Solving Approach` (5-step UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY framework), `# Mathematical Reasoning Principles` (6 bullet points), `# Verification Requirements` (4 bullet points), and `# Output Format`. Opens with: `'You are an elite mathematical problem solver with expertise at the International Mathematical Olympiad (IMO) level.'` Closes with: `'Think step-by-step and show your complete reasoning process. Quality of reasoning is as important as the final answer.'`

* **Tool prompt — expanded with use-case enumeration:**
  * From: 3-line plain description.
  * To: Multi-section description enumerating 5 use cases (complex calculations, numerical verification, generating examples, visualizing structure, brute-force verification), plus guidelines: `'Code should support your mathematical reasoning, not replace it. Explain what you're computing and why before running code.'`

* **User prompt — expanded with library documentation:**
  * From: ``'You have access to `math`, `numpy` and `sympy` to solve the problem.'``
  * To: Structured multi-section prompt with `# Symbolic Computation (sympy)` (6 bullet points), `# Numerical Computation (numpy)` (4 bullet points), `# Mathematical Functions (math)` (3 bullet points), and `Best Practices` (5 bullet points). Includes: `'Combine symbolic and numerical approaches: derive symbolically, verify numerically.'`

* **Config changes:**
  * `output_chars`: 4096 → **2048** (reverted to 02-02 value).
* Everything else is identical to Preview-02-03.

---

## Mk II Public Leaderboard Performance

| Version | LB Score |
| :--- | :--- |
| Preview-01-29 | 38 |
| Preview-01-30 | 38 |
| Preview-01-31 | 34 |
| Preview-02-01 | 34 |
| Preview-02-02 | 33 |
| Preview-02-03 | 38 |
| Preview-02-05 | 33 |
