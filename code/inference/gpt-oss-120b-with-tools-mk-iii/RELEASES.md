## Version 1

**Mk III Baseline — Prompt consolidation, Harmony metadata removal, context window reduction.**

The initial Mk III release, forked from Mk III Preview-03-03 with prompt architecture simplifications and a reduced context window:

* **Prompt architecture — merged back to single-role:**
  * Preview-03-03 split the prompt into `system_prompt` (persona only) + `developer_prompt` (task constraints) with a three-message template `[system, developer, user]`.
  * Version 1 merges them back into a unified `system_prompt` with a two-message template `[system, user]`. `DeveloperContent` and `developer_prompt` are removed entirely (imitating every version of the Mk III preview series before Preview-03-02).
* **System Prompt:** `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'` — reunifies the persona identity (from Preview-03-03's `system_prompt`) and the task constraints (from Preview-03-03's `developer_prompt`) into a single system-role message.
* **Harmony metadata removed:** Removed `conversation_start_date` and `knowledge_cutoff` config fields (were `'2025-11-20'` and `'2024-06'` in Preview-03-03). `get_system_content()` no longer calls `.with_conversation_start_date()` or `.with_knowledge_cutoff()` on the `SystemContent` builder chain.
* **`get_system_content()` signature simplified:** Removed `conversation_start_date` and `knowledge_cutoff` parameters. Signature: `(system_prompt, tool_config)`.
* **`apply_chat_template()` signature simplified:** Removed `conversation_start_date`, `knowledge_cutoff`, and `developer_prompt` parameters. Signature: `(system_prompt, user_prompt, tool_config)`. Constructs two messages: `[system_message, user_message]`.
* **`get_developer_content()` removed:** The `AIMO3Template.get_developer_content(developer_prompt)` method (returning `DeveloperContent.new().with_instructions(developer_prompt)`) is removed entirely. `DeveloperContent` is no longer imported from `openai_harmony`.
* **`_process_attempt` — template call updated:** The `apply_chat_template()` call inside `_process_attempt` passes only `self.cfg.system_prompt`, `user_prompt`, and `local_tool.tool_config` (removed `self.cfg.conversation_start_date`, `self.cfg.knowledge_cutoff`, `self.cfg.developer_prompt`).
* **`context_tokens`: `98304` → `81920`.** Maximum concurrency for 81,920 tokens per request: **7.77×** (up from 6.51× at 98,304 on vLLM 0.16.0).
* **`solve_problem` signature simplified:** From `solve_problem(self, id_value: int, problem: str)` to `solve_problem(self, problem: str)`. The `id_value` parameter is removed — the method no longer prints the problem ID prefix.
* **`solve_problem` print format changed:** From `f'\n{id_value} | {problem}\n'` to `f'\n{problem}\n'`.
* **`predict()` — call updated:** `solver.solve_problem(id_value, question_text)` → `solver.solve_problem(question_text)`.
* **Tool Prompt:** `'Use this tool to execute Python code. The tool executes code in a stateful Jupyter notebook environment. The tool returns the execution output or times out after 6 seconds.'` (inherited from Preview-03-02).
* **User Prompt:** ``'Use `math` and `sympy` to solve the problem, supported by `collections`, `itertools`, `fractions` and `decimal`.'`` (inherited from Preview-02-23).
* **User input composition:** `user_prompt = f'{problem} {self.cfg.user_prompt}'`.
* **Sandbox architecture:** Identical to Preview-02-28+ (port management, kernel retry, output truncation, `_kill_descendants`, enhanced `reset()`, CPU isolation, kernel priority reduction). `AIMO3Sandbox.__init__` takes 11 parameters.
* **Sandbox imports:** `math`, `numpy`, `scipy`, `sympy`, `mpmath` (`dps=64`), `decimal`, `fractions`, `functools`, `itertools`, `collections`, `numpy as np`, `sympy as sp`, `Fraction`, `Decimal`, `getcontext` (with `getcontext().prec = 64`), `os` (for `os.nice(10)`). Identical to Preview-02-28+.
* **`_format_error()`:** Identical to the preview series — dynamic `self._custom_errors` dictionary with substring and regex-tuple triggers, `ipykernel_` → `Cell` sanitisation, and `<cell line:>` regex cleanup scoped to kernel frames.
* **`custom_errors` dictionary:** Same three entries as the preview series: 4300-digit `ValueError` guidance, `sympy.valuation` → `sympy.multiplicity` correction, and `NameError` regex trap.
* **`AIMO3Tool`:** Takes `tool_prompt` and `sandbox`. No lock (removed in Preview-03-02). Unchanged from Preview-03-02+.
* **Answer detection:** Only `boxed_pattern` regex (`r'\\boxed\s*\{\s*([0-9,_]+)\s*\}'`). No `text_pattern`. `rfind('}')` approach in `final` channel handler. Unchanged from the preview series.
* **Streaming:** `_stream_completion()` returns `tuple[list[int], list[str], float, int]` — inline entropy accumulation (arithmetic mean). No real-time answer detection.
* **Scoring/voting:** IEW weight formula: `weight = 1 / max(entropy, entropy_floor)`, where `entropy` is the arithmetic mean of per-token Shannon entropies across all generation turns:

$$H = \frac{1}{N} \sum_{t=1}^{N} H_t, \qquad w = \frac{1}{\max(H,\, \varepsilon_{\text{floor}})}$$

  No Python-corroborated boosting. `from_python` flag is still computed and logged.
* **Preload weights:** `ThreadPoolExecutor(max_workers=self.cfg.workers)`.
* **Kernel initialization:** `ThreadPoolExecutor(max_workers=self.cfg.workers)`.
* **`_process_attempt`:** Uses `hashlib.sha256`-based deterministic seeding. Signature: `(user_prompt, attempt_index, stop_event, deadline)`.
* **`_handle_tool_call`:** Returns `tuple[int, list[Message], str]`. Error detection: `'Execution timed out' in response_text or 'Error:' in response_text`.
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
| `top_logprobs` | 5 |
| `batch_size` | 64 |
| `max_errors` | 32 |
| `early_stop` | 4 |
| `attempts` | 8 |
| `workers` | 16 |
| `turns` | 192 |
| `seed` | 42 |
| `gpu_memory_utilization` | 0.99 |
| `temperature` | 1.0 |
| `min_p` | 0.02 |

---

## Version 2

**Stale import cleanup and system prompt shortening.**

* **Removed import:** `DeveloperContent` — stale class left in the `openai_harmony` import block after all usage was removed in Version 1.
* **System prompt:** Removed answer-range sentence:
  * From: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'`
  * To: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. Place your final answer inside \\boxed{}.'`
* Everything else is identical to Version 1.

---

## Version 3

**System prompt — answer-range sentence restored.**

* **System prompt:** Re-added answer-range sentence (reverted Version 2 removal):
  * From: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. Place your final answer inside \\boxed{}.'`
  * To: `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'`
* Everything else is identical to Version 2.

---

## Version 4

**Vanguard Heuristic (Pass@16 -> Pass@8) for maximum compute density.**

* **Config changes:**
  * `attempts`: 8 → **16** (Doubled the initial starting population).
  * `workers`: 16 → **24** (Increased to support the higher initial parallel load).
  * `early_stop`: **4** (Unchanged - triggers when 4 sequences agree).
* **Vanguard heuristic implementation:**
  * `AIMO3Solver.__init__()` now calculates a dynamic cutoff threshold: `self.target_attempts = self.cfg.attempts // 2`.
  * `AIMO3Solver.solve_problem()` initializes a thread-safe race state before submitting tasks: `vanguard = set()`, `vanguard_lock = threading.Lock()`, and `pruning_event = threading.Event()`. These are passed to `executor.submit()`.
  * `_process_attempt()` signature updated to accept `vanguard: set`, `vanguard_lock: threading.Lock`, and `pruning_event: threading.Event`.
  * **Turn Loop Check:** At the start of every iteration in the `cfg.turns` loop, it checks `if pruning_event.is_set() and attempt_index not in vanguard: break` to prevent unnecessary prompts from being rendered.
  * **The Race Condition:** A sequence claims a spot in the `vanguard` set the moment it successfully executes its first Python block without errors (`errors == 0`), or if it arrives at a final answer using pure mental math. It claims the spot using `if attempt_index not in vanguard and len(vanguard) < self.target_attempts:`.
  * **The Kill Switch:** The instant the `vanguard` set reaches the dynamic threshold (`if len(vanguard) == self.target_attempts:`), the global `pruning_event` is set.
* **Mid-generation stream termination:**
  * `_stream_completion()` signature updated to accept `pruning_event: threading.Event`, `attempt_index: int`, and `vanguard: set`.
  * Inside the vLLM chunk generator loop, it checks the race state on every token chunk. If the `pruning_event` is set and the current sequence is *not* in the vanguard, it instantly breaks the streaming loop, terminating token generation mid-sentence.
* **Dataframe filtering:**
  * In `solve_problem()`, before generating the final Pandas DataFrame for logging, `detailed_results` is filtered: `if len(vanguard) > 0: detailed_results = [result for result in detailed_results if (result['Attempt'] - 1) in vanguard]`. This ensures only the 8 "survivors" (or fewer, if `early_stop` triggers) are displayed and passed to `_select_answer()` for entropy-weighted voting.
* Everything else is identical to Version 3.

---

## Version 5

**Vanguard revert — back to Pass@8 with tighter timeouts.**

* **Config changes:**
  * `attempts`: 16 → **8** (reverted to pre-Version 4 value).
  * `workers`: 24 → **16** (reverted to pre-Version 4 value).
  * `high_problem_timeout`: 996 → **960** (36-second reduction).
  * `base_problem_timeout`: 276 → **240** (36-second reduction).
* **Vanguard heuristic removed:** All race-state machinery from Version 4 is stripped out:
  * `AIMO3Solver.__init__()`: `self.target_attempts` calculation removed.
  * `_process_attempt()` signature: `vanguard`, `vanguard_lock`, and `pruning_event` parameters removed.
  * Turn loop check (`if pruning_event.is_set() and attempt_index not in vanguard: break`) removed.
  * Vanguard claim logic (the `errors == 0` and `final_answer is not None` branches) removed.
  * `_stream_completion()` signature: `pruning_event`, `attempt_index`, and `vanguard` parameters removed; mid-stream pruning check removed.
  * `solve_problem()`: `vanguard`, `vanguard_lock`, `pruning_event` locals removed; `executor.submit()` call reverts to `(deadline,)` tuple; post-solve `detailed_results` vanguard filter removed.
* Everything else is identical to Version 4.

---

## Version 6

**CUDA graph size scaling — batch and capture size increase.**

* **Config changes:**
  * `compilation_config['cudagraph_capture_sizes']`: `[1, 2, 4, 8, 16, 32, 64]` → **`[1, 2, 4, 8, 16, 32, 64, 128]`** (added batch-size-128 CUDA graph).
  * `compilation_config['max_cudagraph_capture_size']`: `64` → **`128`**.
  * `batch_size` (`--max-num-seqs`): `64` → **`128`**.
  * Despite these changes, maximum concurrency at the `81,920`-token context remained unchanged at **7.77×** (adding graph size 128 applied no additional throughput headroom at this context length). VRAM increased marginally from **78.8 GB → 79.2 GB**.
* **Result dict key rename:**
  * `'Response Length'` → **`'Response Tokens'`** (cosmetic — tracks the same `response_tokens` counter).
* Everything else is identical to Version 5.

---

## Mk III Public Leaderboard Performance

| Version | LB Score |
| :--- | :--- |
| Version 1 | 38 |
| Version 2 | NA |
| Version 3 | 41 |
| Version 4 | 32 |
| Version 5 | 40 |
| Version 6 | NA |
