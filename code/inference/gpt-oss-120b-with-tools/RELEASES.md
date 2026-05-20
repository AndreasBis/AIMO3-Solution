## Version 1

**Baseline Architecture.**

The initial submission, establishing all core infrastructure:

* **vLLM Server:** `--enable-prefix-caching` enabled. No `--async-scheduling`, no `--disable-log-stats`.
* **System Prompt:** `'You are a world-class International Mathematical Olympiad (IMO) competitor. The final answer must be a non-negative integer between 0 and 99999. You must place the final integer answer inside \\boxed{}.'`
* **User Prompt:** Problem text only (no preference prompt).
* **Sandbox Imports:** `math`, `sympy`, `itertools`, `collections`, `numpy` (as `np`).
* **Sandbox Reset:** Three-step reset: `%reset -f`, then `import gc; gc.collect()`, then re-import all libraries.
* **Answer Detection:** Single `\\boxed{}` regex pattern only.
* **Voting System:** Frequency-based — ties broken by Python call count. Columns: `Answer`, `Votes`, `Calls`.
* **`AIMO3Tool`:** Has a `close()` method and `__del__` destructor.
* **`_process_attempt` return:** `Attempt`, `Answer`, `Python Calls`, `Python Errors`, `Response Length`. Calls `local_tool.close()` in `finally` block.
* **Executor shutdown:** `executor.shutdown(wait=False, cancel_futures=True)`.
* **GC around `solve_problem`:** None.
* **`_process_attempt` seed:** `int(math.pow(self.cfg.seed + attempt_index, 2))`.
* **Config:**

| Parameter | Value |
| :--- | :--- |
| `high_problem_timeout` | 900 |
| `base_problem_timeout` | 300 |
| `session_timeout` | 960 |
| `jupyter_timeout` | 10 |
| `sandbox_timeout` | 5 |
| `context_tokens` | 65536 |
| `search_tokens` | 1024 |
| `buffer_tokens` | 512 |
| `batch_size` | 256 |
| `early_stop` | 4 |
| `attempts` | 8 |
| `workers` | 16 |
| `turns` | 128 |
| `seed` | 42 |
| `gpu_memory_utilization` | 0.96 |
| `temperature` | 1.0 |
| `min_p` | 0.02 |

---

## Version 2

**vLLM server optimization.**

* **vLLM Server:** Added `--async-scheduling` flag to the server command to enable asynchronous request scheduling.
* Everything else is identical to Version 1.

---

## Version 3

**Introduction of preference-based prompting.**

* **New field:** ``preference_prompt = 'Use `math`, `numpy` and `sympy` to solve the problem.'`` added to `CFG`.
* **User input composition:** Changed from passing bare `problem` to `user_input = f'{problem} {self.cfg.preference_prompt}'`. The `user_input` is what now gets sent to `_process_attempt` instead of `problem`.
* Everything else is identical to Version 2.

---

## Version 4

**Experimental prompt variation.**

* **`preference_prompt`** changed from ``'Use `math`, `numpy` and `sympy` to solve the problem.'`` to ``'Translate the problem into a different mathematical language and solve using `math` and `sympy`.'``
* Everything else is identical to Version 3.

---

## Version 5

**Performance optimization and prompt reversion.**

* **`preference_prompt`** reverted to ``'Use `math` and `sympy` to solve the problem.'`` (dropped `numpy`).
* **Timeout reductions:**
  * `jupyter_timeout`: 10 → **6**.
  * `sandbox_timeout`: 5 → **3**.
* **`search_tokens`**: 1024 → **64**.
  * The field order also changed: `search_tokens` moved to after `buffer_tokens` (was before it).
* **Sandbox imports:** Removed `import numpy as np` from both `__init__` and `reset()`.
* **Sandbox `reset()` consolidated:** Previously three separate `execute()` calls (`%reset -f`, `gc.collect()`, imports). Now combined into a single `execute()` call starting with `%reset -f\n` followed by the library imports. The `gc.collect()` call was removed from reset.
* **`AIMO3Tool`:** Removed the `close()` method and `__del__` destructor entirely.
* **`_process_attempt` cleanup:** Removed the `local_tool.close()` call from the `finally` block (since `close()` no longer exists).
* **vLLM Server:** Added `--disable-log-stats` flag.
* **Executor shutdown:** Changed from `executor.shutdown(wait=False, cancel_futures=True)` to `stop_event.set()` before `executor.shutdown(wait=True, cancel_futures=True)`. This means the executor now waits for running futures to complete.
* **GC around `solve_problem`:** Added `gc.disable()` before and `gc.enable(); gc.collect()` after the `solver.solve_problem()` call in the `predict()` function.

---

## Version 6

**Major architectural overhaul: Entropy-weighted voting system.**

* **`preference_prompt`** changed from ``'Use `math` and `sympy` to solve the problem.'`` back to ``'Use `math`, `numpy` and `sympy` to solve the problem.'`` (re-added `numpy`).
* **New config field:** `top_logprobs = 5`.
* **Sandbox imports:** Re-added `import numpy` (as `numpy`, not `np`) to both `__init__` and `reset()`.
* **New method `_compute_mean_entropy()`:** Computes per-token Shannon entropy from top log-probabilities and returns the mean entropy across all tokens. Returns `float('inf')` if the buffer is empty.
* **Streaming changes:** Added `logprobs=self.cfg.top_logprobs` to the `completions.create()` call. Added `logprobs_buffer = []` to accumulate log-probabilities during streaming. During chunk processing, `chunk.choices[0].logprobs.top_logprobs` is extracted and appended to `logprobs_buffer`.
* **`_process_attempt` return:** Added `'Entropy': float('inf')` to the early-exit dict and `'Entropy': mean_entropy` to the normal return dict. Entropy is computed via `self._compute_mean_entropy(logprobs_buffer)` after the attempt completes.
* **`_select_answer()` rewritten (entropy-weighted voting):**
  * Old: Frequency-based voting, ties broken by Python call count. Sorted by `(votes, calls)`. Columns: `Answer`, `Votes`, `Calls`.
  * New: Entropy-weighted scoring. Each answer gets weight `1.0 / max(entropy, 1e-9)`. Answers ranked by total weight (`score`). Columns: `Answer`, `Votes`, `Score`. Score is rounded to 3 decimal places.
  * Old output: `'Final Result: {answer} | Votes: {votes} | Calls: {calls}'`.
  * New output: `'Final Answer: {answer}'`.
  * Added fallback: if `scored_answers` is empty, prints `'Final Answer: 0'` and returns 0.
* **Results display:** Added `results_dataframe['Entropy'] = results_dataframe['Entropy'].round(3)` when displaying the results DataFrame.
* **Whitespace normalization:** Extensive whitespace changes throughout `AIMO3Solver` — blank lines after method signatures changed from `\n\n` (empty line) to `\n    \n` (line with trailing spaces). This is purely cosmetic.

---

## Version 7

**Library availability expansion.**

* **`preference_prompt`** changed from ``'Use `math`, `numpy` and `sympy` to solve the problem.'`` to ``'You have access to `math`, `numpy`, `sympy`, `collections` and `itertools` in order to solve the problem.'``
* Everything else is identical to Version 6.

---

## Version 8

**Numerical precision and fallback answer detection.**

* **`preference_prompt`** changed from ``'You have access to `math`, `numpy`, `sympy`, `collections` and `itertools` in order to solve the problem.'`` to ``'You have access to `math`, `numpy` and `sympy` to solve the problem.'`` (removed `collections`/`itertools` mentions).
* **`search_tokens`**: 64 → **32**.
* **Sandbox imports:** Added `import mpmath` and `mpmath.mp.dps = 64` to both `__init__` and `reset()` (arbitrary-precision floating-point).
* **`_scan_for_answer()` enhanced:** Added a second fallback regex pattern `r'final\s+answer\s+is\s*([0-9,]+)'` (case-insensitive) that fires when no `\\boxed{}` match is found.

---

## Version 9

**Comprehensive high-precision library integration.**

* **`preference_prompt`** changed from ``'You have access to `math`, `numpy` and `sympy` to solve the problem.'`` to ``'To solve the problem, you have access to the following Python libraries: `math`, `numpy`, `sympy`, `decimal`, `fractions`, `itertools`, `collections`.'``
* **Sandbox imports expanded** (both `__init__` and `reset()`):
  * Added `import decimal`, `import fractions`, `from fractions import Fraction`, `from decimal import Decimal, getcontext`.
  * Added `getcontext().prec = 128` (Decimal precision).
  * Added `numpy.seterr(all="raise")` (raise exceptions on numpy errors instead of warnings).
  * Moved `import mpmath` earlier in the import order (before `itertools`/`collections`).

---

## Version 10

**Developer role architecture experiment.**

* **Harmony import:** Added `DeveloperContent` to the import list from `openai_harmony`.
* **Config rename:** `preference_prompt` → `developer_prompt` (same text as V9).
* **`AIMO3Template` expanded:**
  * New method `get_developer_content()` returning `DeveloperContent.new().with_instructions(developer_prompt)`.
  * `apply_chat_template()` now takes an additional `developer_prompt` parameter.
  * Message list changed from `[system_message, user_message]` to `[system_message, developer_message, user_message]`.
* **`_process_attempt` signature:** Added `developer_prompt` parameter. Passes it through to `apply_chat_template()`.
* **User input:** Reverted to passing bare `problem` text (removed the `user_input = f'{problem} {self.cfg.preference_prompt}'` line). The preference/developer instructions now go through the Developer role instead.
* **`solve_problem()`:** Task tuples changed from `(system_prompt, attempt_index)` to `(system_prompt, developer_prompt, attempt_index)`. Unpacking and submission updated accordingly.

---

## Version 11

**Reversion to V8 architecture with timeout reduction.**

* **Harmony import:** Removed `DeveloperContent`.
* **Config rename:** `developer_prompt` → back to `preference_prompt`.
* **`preference_prompt`** changed from V9's exhaustive library list to ``'You have access to `math`, `numpy` and `sympy` to solve the problem.'`` (same as V8).
* **`base_problem_timeout`**: 300 → **270**.
* **`AIMO3Template`:** Removed `get_developer_content()` method. Removed `developer_prompt` parameter from `apply_chat_template()`. Message list reverted to `[system_message, user_message]`.
* **`_process_attempt`:** Removed `developer_prompt` parameter.
* **User input:** Re-introduced `user_input = f'{problem} {self.cfg.preference_prompt}'` in `solve_problem()`.
* **`solve_problem()`:** Task tuples reverted to `(system_prompt, attempt_index)`.
* **Sandbox imports trimmed:** Removed `import decimal`, `import fractions`, `from fractions import Fraction`, `from decimal import Decimal, getcontext`, `getcontext().prec = 128`, and `numpy.seterr(all="raise")` from both `__init__` and `reset()`. Kept `mpmath` and `mpmath.mp.dps = 64`.

---

## Version 12

**Timeout reversion to 300s.**

* **`base_problem_timeout`**: 270 → **300** (reverted to V8/V1 default).
* Everything else is identical to Version 11.

---

## Version 13

**Prompt simplification, prefix caching disabled, and timing instrumentation.**

* **`preference_prompt`** changed from ``'You have access to `math`, `numpy` and `sympy` to solve the problem.'`` to ``'Use `math`, `numpy` and `sympy` to solve the problem.'`` (imperative "Use" instead of declarative "You have access to").
* **vLLM Prefix Caching disabled:** Removed `--enable-prefix-caching` from the server command line.
* **Timing instrumentation added to `_process_attempt`:**
  * Added `start_time = time.time()` at the top.
  * Added `elapsed = time.time() - start_time` and `time_duration = f'{int(elapsed // 60):02d}:{int(elapsed % 60):02d}'` after the attempt completes.
  * Added `'Time': '00:00'` to the early-exit dict and `'Time': time_duration` to the normal return dict.

---

## Version 14

**Prompt reversion (prefix caching still disabled).**

* **`preference_prompt`** reverted from ``'Use `math`, `numpy` and `sympy` to solve the problem.'`` to ``'You have access to `math`, `numpy` and `sympy` to solve the problem.'``
* **Trivial whitespace fix:** Removed a trailing comma after `float('inf')` in the early-exit dict (formatting-only).
* Everything else is identical to Version 13 (prefix caching remains disabled).

---

## Version 16

> **Note:** Version 15 does not exist.

**Major refactoring: method extraction, regex compilation, and server re-optimization.**

* **vLLM Configuration:**
  * Re-enabled `--enable-prefix-caching`.
  * `high_problem_timeout`: 900 → **840**.
  * `base_problem_timeout`: 300 → **240**.
  * `session_timeout`: 960 → **900**.

* **Regex compilation:** Moved `\\boxed{}` and `final answer is` regex patterns from inline `re.findall()` calls to pre-compiled `re.compile()` patterns stored as instance attributes (`self.boxed_pattern`, `self.final_answer_pattern`) in `AIMO3Solver.__init__()`. `_scan_for_answer()` now uses `self.boxed_pattern.findall()` and `self.final_answer_pattern.findall()`.

* **New method `_stream_completion()`:** Extracted inline streaming logic from `_process_attempt`'s turn loop into a dedicated method. Signature: `_stream_completion(self, prompt_ids, attempt_seed, logprobs_buffer, stop_event, deadline) -> tuple[list[int], list[str], int | None]`. Returns `(token_buffer, text_chunks, final_answer)`. Handles `max_tokens` check internally (returns `([], [], None)` if too few tokens remain).
  * `total_tokens` is now incremented after the call (`total_tokens += len(token_buffer)`) instead of inside the streaming loop.

* **New method `_handle_tool_call()`:** Extracted tool response processing from `_process_attempt`. Signature: `_handle_tool_call(self, last_message, local_tool) -> tuple[int, list[Message]]`. Returns `(python_errors, tool_responses)`.

* **`_process_attempt` simplified:** Now calls `_stream_completion()` and `_handle_tool_call()` instead of having inline logic.

* **Whitespace normalization:** Extensive changes from `\n    \n` to `\n        \n` (indentation-level trailing whitespace). Purely cosmetic.

---

## Version 17

**Intelligent early stopping and system prompt overhaul.**

* **System prompt changed for the first time since V1:**
  * From: `'You are a world-class International Mathematical Olympiad (IMO) competitor. The final answer must be a non-negative integer between 0 and 99999. You must place the final integer answer inside \\boxed{}.'`
  * To: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is a non-negative integer between 0 and 99999. Place your final answer inside \\boxed{}.'`

* **New config field:** `low_entropy = 0.65`.

* **Low-entropy early stopping logic:** After an answer is collected, if the top answer has `early_stop - 1` votes (i.e., 3 votes), and *all* matching attempts have entropy ≤ `0.65`, the `stop_event` is set immediately and remaining futures are cancelled. This triggers earlier than the standard `early_stop` (4 votes) threshold.

* **Attempt seed formula changed:** From `int(math.pow(self.cfg.seed + attempt_index, 2))` to `hash((self.cfg.seed, attempt_index, 'salt')) % (2**31)`.

---

## Version 18

**Removal of low-entropy early stopping and entropy weight floor change.**

* **System prompt tweak:** Changed `'Place your final answer inside \\boxed{}.'` to `'You must place the final answer inside \\boxed{}.'` (re-introduced "You must").
* **Removed `low_entropy`** config field.
* **Removed low-entropy early stopping logic** entirely (the `elif counts and counts[0][1] >= self.cfg.early_stop - 1:` block).
* **`_select_answer()` entropy floor:** Changed from `1.0 / max(entropy, 1e-9)` to `1.0 / max(entropy, 0.1)`. This caps the maximum weight at 10 instead of 1 billion, reducing the dominance of very-low-entropy answers.
* **`_handle_tool_call()` return type annotation fix:** Changed from `-> tuple[int, int]` to `-> tuple[int, list[Message]]` (the second element was always `list[Message]`, annotation was wrong in V16/V17).

---

## Version 19

**Seed change.**

* **`seed`**: 42 → **33**.
* Everything else is identical to Version 18.

---

## Version 20

**System prompt modification.**

* **System prompt:**
  * From: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is a non-negative integer between 0 and 99999. You must place the final answer inside \\boxed{}.'`
  * To: `'You are an International Mathematical Olympiad (IMO) Competitor. You have practiced extensively with problems from Art of Problem Solving (AoPS). The answer is a non-negative integer between 0 and 99999. You must place the final answer inside \\boxed{}.'`
* **`seed`**: 33 → **42** (reverted).

---

## Version 21

**Complete reversion to V11-era architecture.**

* **System prompt reverted** to the V1/V11 standard:
  * `'You are a world-class International Mathematical Olympiad (IMO) competitor. The final answer must be a non-negative integer between 0 and 99999. You must place the final integer answer inside \\boxed{}.'`

* **Config changes:**
  * `high_problem_timeout`: 840 → **900** (reverted to V1 default).
  * `base_problem_timeout`: 240 → **276**.
  * `session_timeout`: 900 → **960** (reverted to V1 default).
  * `workers`: 16 → **8** (1:1 ratio with `attempts`).

* **De-refactored:** Removed `_stream_completion()` and `_handle_tool_call()` methods (introduced in V16). All logic inlined back into `_process_attempt()` as in V11/V12.

* **Regex de-compiled:** Removed `self.boxed_pattern` and `self.final_answer_pattern` instance attributes. `_scan_for_answer()` reverted to inline `re.findall()` calls.

* **`_process_attempt` changes:**
  * Removed timing instrumentation: no `start_time`, no `'Time': time_duration` in the return dict, no `'Time': '00:00'` in the early-exit dict.
  * Attempt seed reverted from `hash((self.cfg.seed, attempt_index, 'salt')) % (2**31)` back to `int(math.pow(self.cfg.seed + attempt_index, 2))`.
  * `total_tokens` increment moved back inside the streaming loop (`total_tokens += len(new_tokens)`).
  * Tool call error checking is inlined again (instead of via `_handle_tool_call()`).

* **`_select_answer()` entropy floor:** Reverted from `1.0 / max(entropy, 0.1)` to `1.0 / max(entropy, 1e-9)`.

* **Whitespace normalization:** Return to `\n\n`-style blank lines (no trailing spaces).

---

## Version 22

**Backport of Mk II optimizations.**

* **`context_tokens`**: 65536 → **81920** (27% increase — allows longer reasoning chains).
* **`gpu_memory_utilization`**: 0.96 → **0.99**.
* **`base_problem_timeout`**: 276 → **270**.
* **`workers`**: 8 → **16** (reverted to V1 default).
* **`batch_size`**: 256 → **64**.
* **New config fields:**
  * `batched_tokens = 2048` — controls `--max-num-batched-tokens` in vLLM server command.
  * `capture_size = 64` — controls `--max-cudagraph-capture-size` in vLLM server command.
* **vLLM server command additions:**
  * `--max-num-batched-tokens` (set to `batched_tokens`).
  * `--max-cudagraph-capture-size` (set to `capture_size`).
* Everything else is identical to Version 21.

---

## Public Leaderboard Performance

| Version | LB Score |
| :--- | :--- |
| 1 | 40 |
| 2 | 39 |
| 3 | 40 |
| 4 | 37 |
| 5 | 37 |
| 6 | 38 |
| 7 | 39 |
| 8 | 41 |
| 9 | 38 |
| 10 | NA |
| 11 | 42 |
| 12 | 39 |
| 13 | 37 |
| 14 | 37 |
| 16 | 41 |
| 17 | 35 |
| 18 | 39 |
| 19 | 36 |
| 20 | 37 |
| 21 | 39 |
| 22 | 39 |
