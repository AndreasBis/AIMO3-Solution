## Version 1

**Stable Baseline — Final Mk II Architecture.**

Integrates all improvements from the Mk II preview series into a production-grade notebook. Key architectural differences from the Mk II previews:

* **Environment variables:** Includes `VLLM_FLASH_ATTN_VERSION='3'` and `VLLM_ATTENTION_BACKEND='FLASH_ATTN'` (adopted from Preview-02-05).
* **System Prompt:** `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'`
* **Tool Prompt:** ``'Use this tool to execute Python code. The environment is a stateful Jupyter notebook. Use `print()` to output results.'``
* **User Prompt:** ``'Use `math` and `sympy` to solve the problem, supported by `collections`, `itertools`, `fractions` and `decimal`.'``
* **User input composition:** `user_text = f'{problem} {self.cfg.user_prompt}'`.
* **`solve_problem` signature:** `solve_problem(self, id_value: int, problem: str)`. Prints `f'({id_value}) {problem}'`.
* **Sandbox architecture — complete rewrite from previews:**
  * `AIMO3Sandbox.__init__` takes 10 parameters: `jupyter_timeout`, `output_chars`, `kernel_attempts`, `port_increment`, `port_attempts`, `port_timeout`, `max_port`, `min_port`, `backoff_delay`, `iopub_timeout`.
  * **Port management:** Class-level `_port_lock`, `_blacklisted_ports`, `_last_blacklist_clear`, `_next_port`. Thread-safe port allocation via `_get_next_ports(count)` with `_is_port_available()` and `_blacklist_ports()` for port-in-use recovery.
  * **Kernel retry logic:** Exponential backoff over `kernel_attempts` retries. On `'Address already in use'`, blacklists the conflicting ports.
  * **Output truncation:** Same `_truncate_output()` as Preview-02-02+.
  * **Warning message:** ``'[WARNING] No output. Use `print()` to output results.'``
* **Sandbox imports:** `math`, `sympy`, `mpmath` (`dps=64`), `decimal`, `fractions`, `itertools`, `collections`, `Fraction`, `Decimal`, `getcontext` (with `getcontext().prec = 64`). No `numpy`.
* **`AIMO3Tool` — simplified constructor:** Takes only `tool_prompt` and `sandbox`. No `zmq_attempts`/`zmq_timeout`/`output_chars` — the sandbox handles all of that internally.
* **Answer detection:** Only `boxed_pattern` regex (`\\boxed{}`). No `text_pattern`. Pattern matches `[0-9,_]+` (adds underscore support: `r'\\boxed\s*\{\s*([0-9,_]+)\s*\}'`). Value cleaning uses `re.sub(r'[,_]', '', ...)`.
* **Streaming:** `_stream_completion()` returns `tuple[list[int], list[str]]`. No real-time answer detection (inherited from Preview-02-03).
* **Answer extraction:** Uses `rfind('}')` approach in the `final` channel handler (inherited from Preview-02-03).
* **Scoring/voting:** Weight formula: `weight = 1 / max(entropy, entropy_floor)`. No call/error quality signal (inherited from Preview-02-02).
* **Preload weights:** `ThreadPoolExecutor()` (default workers).
* **Kernel initialization:** `ThreadPoolExecutor(max_workers=self.cfg.workers)` — uses explicit worker count for kernel startup.
* **`_process_attempt`:** Uses `hash((self.cfg.seed, attempt_index, 'salt')) % (2**31)` for seeding.
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
| `context_tokens` | 81920 |
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
| `turns` | 128 |
| `seed` | 42 |
| `gpu_memory_utilization` | 0.99 |
| `temperature` | 1.0 |
| `min_p` | 0.02 |

* **vLLM Server flags:** `--async-scheduling`, `--disable-log-stats`, `--enable-prefix-caching`, `--max-cudagraph-capture-size` (set to `capture_size`), `--max-num-batched-tokens` (set to `batched_tokens`). No `--max-logprobs`.

---

## Version 2

**Persona Ablation.**

* **System prompt:** Changed persona from "IMO Gold Medalist" to "world-class competitor":
  * From: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. ...'`
  * To: `'You are a world-class International Mathematical Olympiad (IMO) competitor. ...'`
* Everything else is identical to Version 1.

---

## Version 3

**Deterministic seeding, inline entropy, and granular token tracking.**

* **New import:** `import hashlib`.
* **System prompt:** Reverted persona back to "IMO Gold Medalist":
  * From: `'You are a world-class International Mathematical Olympiad (IMO) competitor. ...'`
  * To: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. ...'`
* **Config changes:**
  * `capture_size`: 64 → **32**.
  * `batch_size`: 64 → **32**.
* **vLLM server command:** Added `--max-logprobs` (set to `top_logprobs`).
* **Deterministic seeding:** Replaced `hash()` with `hashlib.sha256` for attempt seed generation:
  * From: `attempt_seed = hash((self.cfg.seed, attempt_index, 'salt')) % (2**31)`
  * To:
    ```python
    string = f'{self.cfg.seed}-{attempt_index}'
    signature = hashlib.sha256(string.encode()).hexdigest()
    attempt_seed = int(signature, 16) % (2**31)
    ```
* **Entropy computation — inlined into streaming loop:**
  * Removed `_compute_mean_entropy()` method entirely.
  * Removed `logprobs_buffer` parameter from `_stream_completion()`.
  * `_stream_completion()` return type changed from `tuple[list[int], list[str]]` to `tuple[list[int], list[str], float, int]` — returns `cumulative_entropy` and `entropy_token_count` in addition to token/text data.
  * Entropy is now calculated incrementally inside the streaming loop: each chunk's `top_logprobs` is iterated, per-token entropy is computed inline (`token_entropy = -Σ prob * log2(prob)`), and accumulated into `cumulative_entropy` / `entropy_token_count`.
  * If no tokens, returns `([], [], 0.0, 0)` instead of `([], [])`.
  * In `_process_attempt`, `total_entropy` and `total_entropy_tokens` accumulate across turns, and `mean_entropy = total_entropy / total_entropy_tokens` (or `inf` if zero).
  * `token_buffer` and `text_chunks` are initialized **before** the `try` block (previously inside it).
* **Granular token tracking in `_process_attempt`:**
  * Replaced single `total_tokens` counter with three: `input_tokens`, `python_tokens`, `response_tokens`.
  * `input_tokens`: Captured once before the loop via `len(initial_prompt_ids)` (initial prompt rendering).
  * `python_tokens`: Accumulated per tool response by encoding each tool response text via `len(encoding.encode(text))`.
  * `response_tokens`: Accumulated per streaming turn (replaces old `total_tokens`).
  * `total_tokens = input_tokens + python_tokens + response_tokens` (computed after the loop).
  * Result dict expanded from 7 to 10 keys:
    * Added: `'Input Tokens'`, `'Python Tokens'`, `'Total Tokens'`.
    * Renamed: `'Response Length'` now tracks `response_tokens` only (previously tracked all tokens).
    * Reordered: `'Answer'` moved to last position.
* Everything else is identical to Version 2.

---

## Version 4

**Throughput scaling — batch and capture size restoration.**

* **Config changes:**
  * `capture_size`: 32 → **64** (reverted to Version 1 value).
  * `batch_size`: 32 → **64** (reverted to Version 1 value).
* Everything else is identical to Version 3.

---

## Version 5

**Timeout sensitivity test.**

* **Config changes:**
  * `high_problem_timeout`: 996 → **876** (120-second reduction to test impact).
* Everything else is identical to Version 4.

---

## Version 6

**Deep reasoning, timeout restoration, and robust sandbox cleanup.**

* **New import:** `import signal`.
* **Config changes:**
  * `high_problem_timeout`: 876 → **996** (reverted to Version 4 value).
  * `turns`: 128 → **192** (50% increase in maximum tool-use turns).
* **New method `AIMO3Sandbox._kill_descendants(pid)`:** Recursive process killer that inspects `/proc` to identify child processes via `ppid` field in `/proc/{entry}/stat`. Recursively kills all descendants with `SIGKILL` before killing the direct children. Suppresses `FileNotFoundError`, `ProcessLookupError`, `PermissionError`, `ValueError`, `IndexError`.
* **`AIMO3Sandbox.reset()` — enhanced with process cleanup:** Before executing the `%reset -f` code, now:
  1. Checks `self._km.is_alive()`.
  2. Calls `self._km.interrupt_kernel()`.
  3. Calls `self._kill_descendants(self._km.kernel.pid)` to terminate any zombie child processes spawned by the kernel.
* **`close()` method repositioned:** Moved from before `reset()` to after `reset()` in the class definition (no logic change).
* Everything else is identical to Version 5.

---

## Version 7

**Reasoning depth.**

* **Config changes:**
  * `turns`: 192 → **128** (reverted to Version 4 value).
* Everything else is identical to Version 6.

---

## Version 8

**Memory optimization and reasoning depth restoration.**

* **New environment variable:** Added `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'` to reduce CUDA memory fragmentation.
* **Config changes:**
  * `turns`: 128 → **192** (reverted to Version 6 value).
* Everything else is identical to Version 7.

---

## Version 9

**Python-corroborated answer boosting.**

* **New config parameter:** `weight`: **2.0** — multiplier for answers whose final value also appeared in Python tool output.
* **`_handle_tool_call` return type changed:** Now returns `tuple[int, list[Message], str]` (adds `response_text` as third element) instead of `tuple[int, list[Message]]`.
* **Python output tracking in `_process_attempt`:**
  * New variable `python_outputs = set()` — collects integer values found in Python tool responses.
  * After each tool call, all integers matching `\b([0-9]+)\b` in the tool response text are parsed; values in `[0, 99999]` are added to `python_outputs`.
  * New variable `from_python = final_answer is not None and final_answer in python_outputs` — boolean flag indicating whether the final answer was corroborated by Python output.
  * Result dict expanded from 10 to 11 keys: added `'From Python'` (placed before `'Answer'`).
* **`_select_answer` — weighted boosting for Python-corroborated answers:**
  * Now reads `from_python` from each result.
  * If `from_python` is `True`: `weight = self.cfg.weight / max(entropy, self.cfg.entropy_floor)` (boosted by `cfg.weight = 2.0`).
  * If `from_python` is `False`: `weight = 1 / max(entropy, self.cfg.entropy_floor)` (unchanged from Version 8).
* Everything else is identical to Version 8.

---

## Version 10

**Sandbox scaling, deterministic execution, and Python output pattern refinement.**

* **New environment variables:**
  * `PYTHONHASHSEED='0'` — deterministic hash seed for reproducible dict/set iteration order.
  * `OMP_NUM_THREADS='1'` — limits OpenMP thread spawning per worker to prevent thread explosion with 24 parallel workers.
* **Config changes:**
  * `workers`: 16 → **24** (50% increase in sandbox pool size).
  * Removed `weight` parameter (was 2.0 in Version 9).
* **Sandbox kernel initialization — execution limits:**
  * Added `import sys` to kernel startup imports.
  * Added `sys.setrecursionlimit(4096)` — raises recursion limit from Python default (1000) to allow deeper recursive algorithms.
  * Added `sys.set_int_max_str_digits(64)` — limits integer-to-string conversion to 64 digits, matching `mpmath.mp.dps` and `getcontext().prec` precision ceiling. Prevents verbose output from large intermediate values.
  * These limits apply to both initial kernel startup and `reset()` method.
* **Python output tracking — pattern refinement:**
  * Regex pattern changed from `r'\b([0-9]+)\b'` to `r'\b([0-9,_]+)\b'` — now captures integers with comma separators (e.g., `1,234`) and underscore separators (e.g., `12_345`), matching the `boxed_pattern` character class.
  * The `python_outputs` set and `'From Python'` result field remain (inherited from Version 9).
* **`_select_answer` — reverted to entropy-only weighting:**
  * Removed Python-corroborated answer boosting logic from Version 9.
  * All answers now use uniform weight formula: `weight = 1 / max(entropy, self.cfg.entropy_floor)` (reverted to Version 8 behavior).
  * The `from_python` flag is still computed and logged but no longer influences scoring.
* Everything else is identical to Version 9.

---

## Version 11

**Sandbox scaling.**

* **Config changes:**
  *  `workers`: 24 → **16** (reverted to pre Version 10 value).
* Everything else is identical to Version 10.

---

## Version 12

**Sandbox and environment cleanup — removal of Version 10 hardening.**

* **Removed environment variables:**
  * `PYTHONHASHSEED='0'` — deterministic hash seed removed; unnecessary for this pipeline.
  * `OMP_NUM_THREADS='1'` — OpenMP thread restriction removed. Originally added to prevent thread explosion with 24 workers (Version 10), but with `workers=16` this restriction throttled vLLM's CPU-side operations (tokenization, scheduling, KV cache management), reducing inference throughput globally.
* **Sandbox kernel initialization — execution limits removed:**
  * Removed `import sys` from kernel startup and `reset()` imports.
  * Removed `sys.setrecursionlimit(4096)` — reverts to Python default (1000).
  * Removed `sys.set_int_max_str_digits(64)` — the 64-digit cap on integer-to-string conversion caused `ValueError` exceptions when the model printed intermediate values exceeding 64 digits (e.g., `50!` has 65 digits, `63!` has 88 digits). These errors consumed turns on debugging rather than solving, and provided zero mathematical information compared to the raw integer output.
  * These removals apply to both initial kernel startup and `reset()` method.
* Everything else is identical to Version 11.

---

## Version 13

**Expanded sandbox imports, traceback noise reduction, debug mode, and message format cleanup.**

* **New config parameter:** `debug_mode`: **`False`** — when set to `True`, prints the full Python error text to stdout whenever a tool call produces an error, annotated with the attempt index.
* **Sandbox kernel initialization — expanded imports:**
  * Added `import sys`, `import json`, `import numpy`, `import scipy`, `import functools`, `import numpy as np`, `import sympy as sp` to both the initial kernel startup block and the `reset()` method.
  * Sandbox now exposes `numpy` and `scipy` to the model in addition to the existing `sympy`/`mpmath`/`decimal`/`fractions` stack.
* **Traceback filtering — enhanced noise reduction:**
  * The `_truncate_output()` traceback cleanup now additionally skips frames matching any of the following patterns (previously only `File "..." ipython-input` was filtered):
    * Frames containing `/tmp/ipykernel_` (kernel temp file paths).
    * Frames containing `<cell line:` (Jupyter cell-line annotations).
    * Frames containing `Traceback (most recent call last)` (redundant header line).
    * Frames containing `-----` (IPython separator lines).
    * Frames containing `/dist-packages/` (third-party library internals).
    * Frames containing `/usr/local/lib` or `/usr/lib` (system library internals).
  * Net effect: tool responses now surface only the application-level error line, stripping all kernel, IPython, and library frame noise.
* **Error detection — pattern change in `_handle_tool_call`:**
  * From: `response_text.startswith('[ERROR]') or 'Traceback' in response_text or 'Error:' in response_text`
  * To: `'Execution timed out' in response_text or 'Error:' in response_text`
  * `'Traceback'` removed as a trigger — with the enhanced traceback filtering above, raw traceback headers no longer appear in tool responses. `'[ERROR]'` prefix check removed to match the updated message format (see below).
* **Message format cleanup — bracket prefixes removed:**
  * Timeout message: `'[ERROR] Execution timed out after {effective_timeout} seconds'` → `'Execution timed out after {effective_timeout} seconds.'`
  * No-output warning: ``'[WARNING] No output. Use `print()` to output results.'`` → ``'No output. Use `print()` to output results.'``
* **`_handle_tool_call` signature:** Added `attempt_index: int` parameter, forwarded from `_process_attempt`. Used exclusively for the `debug_mode` print.
* Everything else is identical to Version 12.

---

## Version 14

**Intelligent traceback sanitisation and 4300-digit conversion interception.**

* **Traceback filtering — context preservation:**
  * IPython separator dash filter updated from `-----` to `----------`.
  * Kernel temp file paths (`/tmp/ipykernel_...`) are no longer completely dropped; they are instead sanitised and renamed to `Cell` via regex.
  * Multi-line regex (`r'^.*<cell line: \d+>\(\)\s*\n?'`) replaces the naive `<cell line:` substring match to cleanly prune notebook line stubs.
  * These changes ensure the stack trace pointing back to the model's generated code remains visible (restoring error locality), rather than being over-pruned.
* **Automatic integer string conversion guidance:**
  * Intercepts the `ValueError` exception triggered when Python's integer-to-string conversion limit (4300 digits) is exceeded.
  * Replaces the static error text with an explicit mathematical workaround to guide the model: ``'ValueError: Exceeds the limit (4300 digits) for integer string conversion. Use `d = int(mpmath.log10(n)) + 1; d += (10**d <= n) - (10**(d-1) > n)` to calculate the exact number of digits.'``
* Everything else is identical to Version 13.

---

## Version 15

**Dynamic Regex Error Trapping and Conceptual Mathematical Guidance.**

* **Sandbox architecture — configurable error intercepts:**
  * Introduced `CFG.custom_errors` dictionary to centralize and configure specific error interventions. This decouples error-specific guidance from the core parsing logic, allowing scalable interception via both substring matches and Regex patterns.
* **`AIMO3Sandbox._format_error` refactor:**
  * Iterates through `self._custom_errors.items()`. Supports `('regex', r'pattern')` tuples for dynamic trapping (using `re.sub` for string replacement) along with standard substring intercepts.
* **Error guidance pivot — from rigid code injection to conceptual steering:**
  * **4300-digit conversion intercept:** Replaced the exact code snippet (`d = int(...)`) from Version 14 with conceptual mathematical guidance: `'ValueError: Exceeds the limit (4300 digits) for integer string conversion. Use logarithms to count digits or modular arithmetic to check equality.'`
  * **SymPy valuation intercept:** Added a trap for `AttributeError: module 'sympy' has no attribute 'valuation'`, correcting the model to use `sympy.multiplicity(p, n)`.
  * **Undefined variables intercept:** Added a Regex trap for `NameError` (`r'^NameError: name \'(.+?)\' is not defined'`), dynamically capturing the exact variable name and instructing the model to ensure it defines every variable and helper function before use.
* **Traceback filtering — `<cell line:>` regex scope narrowed:**
  * The `re.sub(r'^.*<cell line: \d+>\(\)\s*\n?', ...)` cleanup was moved inside the `if '/tmp/ipykernel_'` guard. Previously ran on every frame; now only runs on frames containing `/tmp/ipykernel_`.
* Everything else is identical to Version 14.

---

## Version 17

> **Note:** Version 16 does not exist.

**Code quality improvements.**

* **Code formatting & type hinting:**
  * Added missing `-> None` and object type hints (`CFG`, `AIMO3Sandbox`) across several classes and methods.
  * Cleaned up unnecessary trailing whitespace.
* Everything else is identical to Version 15.

---

## Version 18

**Prompt architecture split, Harmony metadata, expanded error interception, and context extension.**

* **Prompt architecture — system/developer role separation:**
  * The `system_prompt` was split into two fields: `system_prompt` and `developer_prompt`.
  * `system_prompt` now contains only the persona identity:
    * From: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'`
    * To: `'You are an International Mathematical Olympiad (IMO) Gold Medalist.'`
  * New field `developer_prompt` carries the task constraints:
    * `'The answer is an integer between 0 and 99999. Return the final answer inside \\boxed{}.'`
  * Wording change: `'Place your final answer'` → `'Return the final answer'`.
* **Harmony protocol — Developer role message:**
  * `apply_chat_template()` now constructs three messages instead of two: `[system_message, developer_message, user_message]`.
  * `developer_message` is created via `Message.from_role_and_content(Role.DEVELOPER, developer_prompt)`.
* **Harmony protocol — system content metadata:**
  * New config fields: `conversation_start_date = '2025-11-20'` and `knowledge_cutoff = '2024-06'`.
  * `get_system_content()` now calls `.with_conversation_start_date(conversation_start_date)` and `.with_knowledge_cutoff(knowledge_cutoff)` on the `SystemContent` builder chain.
* **Tool prompt — timeout disclosure:**
  * From: ``'Use this tool to execute Python code. The environment is a stateful Jupyter notebook. Use `print()` to output results.'``
  * To: `'Use this tool to execute Python code. The environment is a stateful Jupyter notebook. The tool returns the execution output or times out after 6 seconds.'`
* **Custom errors — ModuleNotFoundError interception:**
  * Added a fourth entry to `CFG.custom_errors`: a regex trap for `ModuleNotFoundError` (`r'^ModuleNotFoundError: No module named \'(.+?)\''`). Dynamically captures the missing module name and responds with: ``'ModuleNotFoundError: \'\\1\' is not available. Use `math` and `sympy` to solve the problem, supported by `collections`, `itertools`, `fractions` and `decimal`.'``
* **Config changes:**
  * `context_tokens`: 81920 → **98304** (20% increase — extends maximum sequence length from 80K to 96K tokens to prevent truncation on the hardest problems). Maximum concurrency for 98,304 tokens per request: 6.51x (from 7.77x).
* **Plumbing changes:**
  * `_process_attempt` signature: removed `user_text` parameter, added `system_prompt`, `developer_prompt`, `conversation_start_date`, `knowledge_cutoff`, `user_prompt`. Parameter renamed from `user_text` to `user_prompt` throughout.
  * `get_system_content()` signature: added `conversation_start_date` and `knowledge_cutoff` parameters.
  * `apply_chat_template()` signature: added `developer_prompt`, `conversation_start_date`, `knowledge_cutoff` parameters.
  * `solve_problem()`: task tuples expanded from `(system_prompt, attempt_index)` to `(system_prompt, developer_prompt, conversation_start_date, knowledge_cutoff, attempt_index)`. Local variable renamed from `user_text` to `user_prompt`.
  * `AIMO3Tool` constructor call: collapsed from multi-line to single-line (cosmetic).
* Everything else is identical to Version 17.

---

## Mk II Public Leaderboard Performance

| Version | LB Score |
| :--- | :--- |
| Version 1 | 40 |
| Version 2 | 37 |
| Version 3 | 34 |
| Version 4 | 41 |
| Version 5 | 38 |
| Version 6 | 39 |
| Version 7 | 38 |
| Version 8 | 42 |
| Version 9 | 38 |
| Version 10 | 34 |
| Version 11 | 34 |
| Version 12 | 36 |
| Version 13 | 38 |
| Version 14 | 37 |
| Version 15 | 36 |
| Version 17 | 38 |
| Version 18 | NA |
