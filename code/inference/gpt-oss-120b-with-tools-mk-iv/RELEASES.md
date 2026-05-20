## Version 1

**Mk IV Baseline — vLLM 0.17.0 Migration, Marlin MoE Backend, and Pipeline Consolidation.**

The initial Mk IV release, building upon the architectural stability of Mk III Version 5 with critical backend upgrades and inheriting the expanded CUDA graph logic from Mk III Version 6. As the foundation of the Mk IV series, the full pipeline architecture is detailed below:

* **vLLM Upgrade:** Migrated from vLLM 0.16.0 (Mk III) to **vLLM 0.17.0**.
* **MoE Backend Configuration:**
* New config parameter `moe_backend`: `'marlin'` — passed to the `--moe-backend` vLLM server flag.
* This is a strict requirement for the 120B model on vLLM 0.17.0. Using the default `triton` backend on the H100 instance triggers a CUDA OOM error (falling short by ~46 MB), whereas the `marlin` backend successfully executes at `gpu_memory_utilization=0.99` and reduces overall VRAM consumption to **78.6 GB** (down from 78.8 GB on vLLM 0.16.0).
* **Inherited from Mk III Version 6 (CUDA Graph Expansion):**
  * `batch_size`: **128**.
  * `compilation_config['max_cudagraph_capture_size']`: **128**.
  * `compilation_config['cudagraph_capture_sizes']`: **`[1, 2, 4, 8, 16, 32, 64, 128]`**.
* **Prompt architecture — single-role template:** Unified `system_prompt` with a two-message template `[system, user]` (inherited from Mk III V1). No `DeveloperContent` channel is utilised.
* **System Prompt:** `'You are an International Mathematical Olympiad (IMO) Gold Medalist. The answer is an integer between 0 and 99999. Place your final answer inside \\boxed{}.'` (Inherited from Mk III V3).
* **Tool Prompt:** `'Use this tool to execute Python code. The tool executes code in a stateful Jupyter notebook environment. The tool returns the execution output or times out after 6 seconds.'`
* **User Prompt:** ``'Use `math` and `sympy` to solve the problem, supported by `collections`, `itertools`, `fractions` and `decimal`.'``
* **User input composition:** `user_prompt = f'{problem} {self.cfg.user_prompt}'`.
* **Sandbox architecture:** 2× overprovisioned setup (`workers=16` for `attempts=8`). Features robust port management, kernel retry logic, output truncation, `_kill_descendants`, enhanced `reset()`, CPU isolation (`OPENBLAS_NUM_THREADS='1'`, `NUMEXPR_NUM_THREADS='1'`, `MKL_NUM_THREADS='1'`, `OMP_NUM_THREADS='1'`), and kernel process deprioritisation (`os.nice(10)`).
* **Sandbox imports:** `math`, `numpy`, `scipy`, `sympy`, `mpmath` (`dps=64`), `decimal`, `fractions`, `functools`, `itertools`, `collections`, `numpy as np`, `sympy as sp`, `Fraction`, `Decimal`, `getcontext` (with `getcontext().prec = 64`), `os` (for `os.nice(10)`).
* **`_format_error()`:** Dynamic `self._custom_errors` dictionary with substring and regex-tuple triggers, `ipykernel_` → `Cell` sanitisation, and `<cell line:>` regex cleanup scoped to kernel frames.
* **`custom_errors` dictionary:** Three entries: 4300-digit `ValueError` guidance, `sympy.valuation` → `sympy.multiplicity` correction, and `NameError` regex trap.
* **Answer detection:** Only `boxed_pattern` regex (`r'\\boxed\s*\{\s*([0-9,_]+)\s*\}'`). No `text_pattern`. Evaluated via `rfind('}')` substring search in the `final` channel handler.
* **Scoring/voting:** Pass@8 with `early_stop=4`. IEW weight formula: `weight = 1 / max(entropy, entropy_floor)`, where `entropy` is the arithmetic mean of per-token Shannon entropies across all generation turns:

$$H = \frac{1}{N} \sum_{t=1}^{N} H_t, \qquad w = \frac{1}{\max(H,\, \varepsilon_{\text{floor}})}$$

* **Config:**

| Parameter | Value |
| :--- | :--- |
| `kv_cache_dtype` | `fp8_e4m3` |
| `load_format` | `safetensors` |
| `performance` | `throughput` |
| `moe_backend` | `marlin` |
| `debug_mode` | `False` |
| `high_problem_timeout` | 960 |
| `base_problem_timeout` | 240 |
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
| `server_port` | 8000 |
| `batch_size` | 128 |
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

**Native Python Integer String Conversion Limit Override.**

* **Sandbox initialization:** Added `import sys` and `sys.set_int_max_str_digits(0)` to the initial kernel startup execution string (alongside the existing `os.nice(10)` and imports).
* **Sandbox `reset()`:** Added `import os`, `import sys`, and `sys.set_int_max_str_digits(0)` to the kernel reset string. This change globally disables Python's 4300-digit limit for integer-to-string conversions within the stateful Jupyter environment.
* **Custom errors:** Removed the `'ValueError: Exceeds the limit (4300 digits) for integer string conversion'` intercept from `CFG.custom_errors`. By natively overriding the limit at the kernel level, the model is no longer artificially restricted from using code such as `len(str(N))` with N having more than 4300 digits, rendering the prompt-guidance intercept obsolete.
* Everything else is identical to Version 1.

---

## Version 3

**Timeout reduction.**

* **`base_problem_timeout`**: 240 → **204** (the surplus time for the first problem increased from 5400 to 7200 seconds).
* Everything else is identical to Version 2.

---

## Version 4

**Tool-use oriented performance model.**

* **`performance`**: `throughput` → `interactivity`. The sequential structure of frequent tool use produces repeated prefill restarts at small effective batch sizes as sequences stall on tool round-trips. The fine-grained CUDA graphs and latency-oriented kernels of `interactivity` reduce per-restart overhead, which compounds across the call chain.
* Everything else is identical to Version 3.

---

## Version 5

**ModuleNotFoundError rule inclusion and reversion to Version 2 arguments**

* **`performance`**: `interactivity` → `throughput` (the README analysis indicates the `throughput` mode is better for the highest token generation). 
* **`base_problem_timeout`**: 204 → **240** (reverted to Version 2 value). 
* **Custom errors**: Added `ModuleNotFoundError` regex trap (`r'^ModuleNotFoundError: No module named \'(.+?)\''`), dynamically capturing the missing module name and responding with an explicit list of available sandbox imports.

```python
    custom_errors = {
        'AttributeError: module \'sympy\' has no attribute \'valuation\'':
            'AttributeError: module \'sympy\' has no attribute \'valuation\'. '
            'Use `sympy.multiplicity(p, n)` to compute the p-adic valuation of n.', 
            
        ('regex', r'^NameError: name \'(.+?)\' is not defined'):
            'NameError: name \'\\1\' is not defined. '
            'Ensure you define every variable and helper function before use.', 
            
        ('regex', r'^ModuleNotFoundError: No module named \'(.+?)\''):
            'ModuleNotFoundError: No module named \'\\1\'. '
            'You have access only to: `os`, `sys`, `math`, `numpy`, `scipy`, '
            '`sympy`, `mpmath`, `decimal`, `fractions`, `functools`, `itertools`, `collections`. '
            'Do not attempt to import or install other modules.'
    }
```

* Everything else is identical to Version 4.

---

## Version 6

**ModuleNotFoundError rule removal and internal naming fixes.**

* **Custom errors:** Removed the `ModuleNotFoundError` regex trap (`r'^ModuleNotFoundError: No module named \'(.+?)\''`) that was introduced in Version 5. The `custom_errors` dictionary reverts to two entries: the `sympy.valuation` attribute correction and the `NameError` regex trap.
* **`CFG.performance` → `CFG.performance_mode`:** The config attribute was renamed from `performance` to `performance_mode` to match the vLLM `--performance-mode` flag name exactly. The `_start_server()` method updated accordingly (`self.cfg.performance` → `self.cfg.performance_mode`). No numeric value changed; `'throughput'` is retained.
* **Result dict key renames:** `'Entropy'` → `'Mean Entropy'` and `'Time'` → `'Wall Time'` across `_process_attempt()` (both the early-exit fallback dict and the final return dict) and `_select_answer()` / `solve_problem()` (all downstream reads and the DataFrame rounding call). These are purely cosmetic / display-clarity changes with no behavioural effect.
* Everything else is identical to Version 5.

---

## Version 7

**Naming change.**

* **Model path rename:** Changed the path to the model's weight from `model_path` to `model` (to match the vLLM naming for the server arguments). This change was transferred to all uses of the argument in the `AIMO3Solver` class.
* Everything else is identical to Version 6.

---

## Version 8

**XML-Structured Prompt Redesign and Dead Code Removal.**

* **`system_prompt`:** Restructured from a flat concatenated string into three explicit XML-tagged sections. Content is otherwise equivalent to Version 7, with the addition of a concrete formatting example in the `<response>` block.
* **`tool_prompt`:** Substantially expanded from a single flat description into four explicit XML-tagged sections. The timeout and truncation constraints are extracted into a dedicated `<limitations>` block; a new `<modules>` block explicitly enumerates available sandbox imports, prohibits use of any module outside that list, and illustrates the required explicit import pattern; a new `<precision>` block surfaces the kernel's floating-point configuration (`mpmath.mp.dps = 64`, `getcontext().prec = 64`) directly in the prompt.

```python
    system_prompt = (
        '<identity>'
        'You are an International Mathematical Olympiad (IMO) Gold Medalist.'
        '</identity>'
        '<answer>'
        'The answer is an integer between 0 and 99999.'
        '</answer>'
        '<response>'
        'Place your final answer inside \\boxed{}. '
        'Example: \\boxed{12345}.'
        '</response>'
    )

    tool_prompt = (
        '<description>'
        'Use this tool to execute Python code. '
        'The tool executes code in a stateful Jupyter notebook environment. '
        'The tool returns the execution output.'
        '</description>'
        '<limitations>'
        'The tool times out after 6 seconds. '
        'The tool truncates output beyond 2048 characters.'
        '</limitations>'
        '<modules>'
        'The sandbox pre-imports: '
        '`math`, `sympy`, `mpmath`, `collections`, `itertools`, `functools`, `fractions`, `decimal`. '
        'Do not import or use any library outside of the pre-imported modules. '
        'You have to import individual functions explicitly. '
        'Example: `from math import comb`.'
        '</modules>'
        '<precision>'
        'Floating-point operations are reliable for up to 64 significant digits. '
        'The sandbox sets `mpmath.mp.dps = 64` and `getcontext().prec = 64` by default.'
        '</precision>'
    )
```

* **`AIMO3Template.__init__`:** Removed the no-op `def __init__(self) -> None: pass` body. Python's implicit default constructor is functionally identical; the explicit override carried no behaviour.
* **`AIMO3Sandbox.close()`:** Removed. The method was fully implemented (channel teardown, kernel shutdown, resource cleanup) but had no call site anywhere in the pipeline — sandboxes were exclusively `reset()` and returned to the pool. Its presence implied a teardown contract that was never exercised.
* **`AIMO3Tool.instruction` property:** Removed. The property was a single-indirection wrapper over `self._tool_prompt` with no external callers; its sole use was as `description=self.instruction` inside `tool_config`. Inlined to `description=self._tool_prompt` directly.
* Everything else is identical to Version 7.

---

## Mk IV Public Leaderboard Performance

| Version | LB Score |
| :--- | :--- |
| Version 1 | 42 |
| Version 2 | 41 |
| Version 3 | 40 |
| Version 4 | 41 |
| Version 5 | 38 |
| Version 6 | 40 |
| Version 7 | 39 |
| Version 8 | 39 |
