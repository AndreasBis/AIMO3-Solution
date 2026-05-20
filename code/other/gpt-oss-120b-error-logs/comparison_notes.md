# AIMO3 Sandbox Comparison: v12 vs v13p1 vs v13p2 vs v13p3 vs v13p4 vs v13p5 vs v14p1

> The logs of the runs correspond *only* to problem #10 unless otherwise noted.
> `v13p5` was saved as `Version 13` for Mk II (the only change was the `debug_mode`, which was set to False).
> `v14p2` was saved as `Version 14` for Mk II (the only change was the `debug_mode`, which was set to False).
> `v15p2` was saved as `Version 15` for Mk II (the only change was the `debug_mode`, which was set to False).

---

## 1. `_format_error` Method

The most consequential difference across all versions.

### v12 & v13p1 — Identical

```python
def _format_error(self, traceback: list[str]) -> str:
    
    clean_lines = []
    
    for frame in traceback:
        clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
        
        if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
            continue
            
        clean_lines.append(clean_frame)
        
    return ''.join(clean_lines)
```

**What it does:** Strips ANSI colour codes, drops external library frames (anything with `File "` that isn't `ipython-input`). Keeps everything else: dashed dividers, "Traceback (most recent call last)" headers, `/tmp/ipykernel_` paths, `<cell line:>` stubs, and the final exception line.

### v13p2

```python
def _format_error(self, traceback: list[str]) -> str:
    
    clean_lines = []
    
    for frame in traceback:
        clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
        
        if '/tmp/ipykernel_' in clean_frame:
            continue
            
        if '<cell line:' in clean_frame:
            continue
            
        if 'Traceback (most recent call last)' in clean_frame:
            continue
            
        if '-----' in clean_frame:
            continue
            
        if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
            pass  # ← bug: should be continue
            
        clean_lines.append(clean_frame)
        
    return ''.join(clean_lines)
```

**What it does:** Adds four new guards that strip structural noise (kernel paths, cell stubs, the traceback header, dashed dividers). However, the external-library `continue` was silently changed to `pass`, meaning all SymPy/NumPy/stdlib internal frames now survive. Net effect: less formatting noise, but full library call stacks leak through.

### v13p3

```python
def _format_error(self, traceback: list[str]) -> str:
    
    clean_lines = []
    
    for frame in traceback:
        clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
        
        if '/tmp/ipykernel_' in clean_frame:
            continue
            
        if '<cell line:' in clean_frame:
            continue
            
        if 'Traceback (most recent call last)' in clean_frame:
            continue
            
        if '-----' in clean_frame:
            continue
            
        if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
            continue  # FIXED: was 'pass'
            
        if '/dist-packages/' in clean_frame:
            continue  # NEW: catches IPython-style library frames
            
        clean_lines.append(clean_frame)
        
    return ''.join(clean_lines)
```

**What it does:** Fixes the `pass` bug and adds `/dist-packages/` guard to catch IPython-style external library frames. However, `/usr/local/lib` and `/usr/lib` frames still leak through (v13p4 will add this guard).

### v13p4 & v13p5

```python
def _format_error(self, traceback: list[str]) -> str:
    
    clean_lines = []
    
    for frame in traceback:
        clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
        
        if '/tmp/ipykernel_' in clean_frame:
            continue
            
        if '<cell line:' in clean_frame:
            continue
            
        if 'Traceback (most recent call last)' in clean_frame:
            continue
            
        if '-----' in clean_frame:
            continue
            
        if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
            continue
            
        if '/dist-packages/' in clean_frame:
            continue                
            
        if '/usr/local/lib' in clean_frame or '/usr/lib' in clean_frame:
            continue
            
        clean_lines.append(clean_frame)
        
    return ''.join(clean_lines)
```

**What it does:** Adds the final guard to filter out standard library frames (e.g., `fractions.py`, `random.py`) that were leaking through in v13p3. The guard covers both `/usr/local/lib` (e.g., dist-packages paths that slipped past the previous filter) and `/usr/lib` in a single condition. This results in the most concise error messages possible.

### v14p1

```python
def _format_error(self, traceback: list[str]) -> str:
    
    clean_lines = []
    
    for frame in traceback:
        clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            
        if 'Traceback (most recent call last)' in clean_frame:
            continue
            
        if '----------' in clean_frame:
            continue
            
        if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
            continue
            
        if '/dist-packages/' in clean_frame:
            continue                
            
        if '/usr/local/lib' in clean_frame or '/usr/lib' in clean_frame:
            continue
            
        if '/tmp/ipykernel_' in clean_frame:
            clean_frame = re.sub(r'/tmp/ipykernel_\d+/\d+\.py', 'Cell', clean_frame)
            
        clean_lines.append(clean_frame)
        
    return ''.join(clean_lines)
```

**What it does:** Reverses the "minimalist" direction of v13. Instead of stripping `<cell line:>` stubs and kernel paths entirely, it **sanitizes** them (renaming temp kernel files to `Cell`). It retains the structural guards against Standard Library and external package frames (`/usr/lib`, `/dist-packages/`) but explicitly keeps the stack trace information that points to the user's code (or the model's generated code). This restores the "locality" of errors (line numbers, source snippets) at the cost of increased verbosity.

### v14p2

```python
def _format_error(self, traceback: list[str]) -> str:
    
    clean_lines = []
    
    for frame in traceback:
        clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
        
        if 'Traceback (most recent call last)' in clean_frame:
            continue
            
        if '----------' in clean_frame:
            continue
            
        if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
            continue
            
        if '/dist-packages/' in clean_frame:
            continue                
            
        if '/usr/local/lib' in clean_frame or '/usr/lib' in clean_frame:
            continue
            
        if '/tmp/ipykernel_' in clean_frame:
            clean_frame = re.sub(r'/tmp/ipykernel_\d+/\d+\.py', 'Cell', clean_frame)
            
        clean_frame = re.sub(r'^.*<cell line: \d+>\(\)\s*\n?', '', clean_frame, flags=re.MULTILINE)
        
        if 'Exceeds the limit (4300 digits) for integer string conversion' in clean_frame:
            clean_frame = (
                'ValueError: Exceeds the limit (4300 digits) for integer string conversion. '
                'Use `d = int(mpmath.log10(n)) + 1; d += (10**d <= n) - (10**(d-1) > n)` '
                'to calculate the exact number of digits.'
            )
            
        clean_lines.append(clean_frame)
        
    return ''.join(clean_lines)
```

**What it does:** Builds upon v14p1 by adding two specific interventions: strictly removing `<cell line: ...>` stubs using a multi-line regex, and intercepting the `ValueError` triggered when integer-to-string conversion exceeds 4300 digits, replacing the error message with the exact `mpmath.log10` formula to guide the model.


### v15p1

```python
def _format_error(self, traceback: list[str]) -> str:
    
    clean_lines = []
    
    for frame in traceback:
        clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
        
        if '/tmp/ipykernel_' in clean_frame:
            continue
            
        if '<cell line:' in clean_frame:
            continue
            
        if 'Traceback (most recent call last)' in clean_frame:
            continue
            
        if '-----' in clean_frame:
            continue
            
        if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
            continue
            
        if '/dist-packages/' in clean_frame:
            continue                
            
        if '/usr/local/lib' in clean_frame or '/usr/lib' in clean_frame:
            continue
            
        if 'ValueError: Exceeds the limit (4300 digits) for integer string conversion' in clean_frame:
            clean_frame = (
                'ValueError: Exceeds the limit (4300 digits) for integer string conversion. '
                'Use logarithms to count digits or modular arithmetic to check equality.'
            )
            
        clean_lines.append(clean_frame)
        
    return ''.join(clean_lines)
```

**What it does:** Builds upon v13p5 by securely filtering all internal library and `<cell line:>` frames, and intercepts the `ValueError` for the 4300-digits conversion limit by suggesting a conceptual mathematical approach ("Use logarithms...") rather than providing an exact code snippet.

### v15p2

```python
def _format_error(self, traceback: list[str]) -> str:
    
    clean_lines = []
    
    for frame in traceback:
        clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
        
        if 'Traceback (most recent call last)' in clean_frame:
            continue
            
        if '----------' in clean_frame:
            continue
            
        if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
            continue
            
        if '/dist-packages/' in clean_frame:
            continue                
            
        if '/usr/local/lib' in clean_frame or '/usr/lib' in clean_frame:
            continue
            
        if '/tmp/ipykernel_' in clean_frame:
            clean_frame = re.sub(r'/tmp/ipykernel_\d+/\d+\.py', 'Cell', clean_frame)                
            clean_frame = re.sub(r'^.*<cell line: \d+>\(\)\s*\n?', '', clean_frame, flags=re.MULTILINE)
            
        for trigger, replacement in self._custom_errors.items():
            if isinstance(trigger, tuple) and trigger[0] == 'regex':
                pattern = trigger[1]
                
                if re.search(pattern, clean_frame):
                    clean_frame = re.sub(pattern, replacement, clean_frame.rstrip()) + '\n'
                    
            else:    
                if trigger in clean_frame:
                    clean_frame = replacement
                    
        clean_lines.append(clean_frame)
        
    return ''.join(clean_lines)
```

**What it does:** Re-introduces the sanitized `<cell line:>` removal from v14p2 to maintain error locality while keeping frames concise. Most importantly, it completely refactors the specific error replacements (like the 4300-digit integer conversion) into a dynamic `self._custom_errors` dictionary configured in the `CFG` class. This allows passing simple substring matches or regex tuples to cleanly intercept and append actionable guidance to common, repeating errors (e.g. `NameError`).

The custom errors defined are the following:

```python
    custom_errors = {
        'ValueError: Exceeds the limit (4300 digits) for integer string conversion': 
            'ValueError: Exceeds the limit (4300 digits) for integer string conversion. '
            'Use logarithms to count digits or modular arithmetic to check equality.', 
            
        'AttributeError: module \'sympy\' has no attribute \'valuation\'': 
            'AttributeError: module \'sympy\' has no attribute \'valuation\'. '
            'Use `sympy.multiplicity(p, n)` to compute the p-adic valuation of n.', 
            
        ('regex', r'^NameError: name \'(.+?)\' is not defined'): 
            'NameError: name \'\\1\' is not defined. '
            'Ensure you define every variable and helper function before use.'
    }
```

### Filter Comparison Table

| Frame type | v12 | v13p1 | v13p2 | v13p3 | v13p4/p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ANSI colour codes | stripped | stripped | stripped | stripped | stripped | stripped | stripped | stripped | stripped |
| Dashed divider (`-----`) | ❌ kept | ❌ kept | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed |
| `Traceback (most recent call last)` | ❌ kept | ❌ kept | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed |
| `/tmp/ipykernel_` paths | ❌ kept | ❌ kept | ✅ removed | ✅ removed | ✅ removed | ⚠️ sanitized | ⚠️ sanitized | ✅ removed | ⚠️ sanitized |
| `<cell line:>` stubs | ❌ kept | ❌ kept | ✅ removed | ✅ removed | ✅ removed | ❌ kept | ✅ removed | ✅ removed | ✅ removed |
| External library frames (standard format) | ✅ removed | ✅ removed | ❌ **kept** (bug) | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed |
| External `/dist-packages/` (IPython format) | ❌ kept | ❌ kept | ❌ kept | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed | ✅ removed |
| External `/usr/local/lib` or `/usr/lib` frames | ❌ kept | ❌ kept | ❌ kept | ❌ **kept** | ✅ **removed** | ✅ removed | ✅ removed | ✅ removed | ✅ removed |
| User code frames (`ipython-input`) | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept |
| Final exception line | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept | ✅ kept |

---

## 2. Timeout Configuration

Two separate timeout values govern execution: `jupyter_timeout` (the kernel execution wall clock) and `sandbox_timeout` (how long the solver waits to acquire a sandbox from the pool).

| Parameter | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `jupyter_timeout` | 6 s | **12 s** | 6 s | 6 s | 6 s | 6 s | 6 s | 6 s | 6 s | 6 s |
| `sandbox_timeout` | 3 s | **6 s** | 3 s | 3 s | 3 s | 3 s | 3 s | 3 s | 3 s | 3 s |

v13p1 doubled both timeouts as an experiment. The longer timeout in v13p1 did not eliminate timeouts — it only delayed them, consuming more wall-clock budget per failed attempt. v13p2 through v13p5 returned to 6 s / 3 s.

---

## 3. Kernel Preloaded Imports

Imports executed automatically on sandbox initialisation, before the model writes a single line.

| Module | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `math` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `sympy` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mpmath` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `decimal` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `fractions` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `itertools` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `collections` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `from fractions import Fraction` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `from decimal import Decimal, getcontext` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mpmath.mp.dps = 64` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `getcontext().prec = 64` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `sys` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `numpy` / `numpy as np` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `scipy` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `sympy as sp` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `json` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `functools` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

v12 gives the model only the bare mathematical stack. v13p1 added `numpy`, `scipy`, as well the `np` (for `numpy`) and `sp` (for `sympy`) aliases - because the error logs revealed `gpt-oss-120b` prefers to use `sympy` as `sp`. v13p2 further added `json` and `functools`. The fix `import sympy as sp` effectively eliminated `NameError: name 'sp' is not defined` from v13p1 onwards.

---

## 4. Tool Prompt

The instruction string passed to the model describing how to use the Python tool.

**v12, v13p1, v13p5, v14p1, v14p2, v15p1, v15p2 — Minimal:**
```
Use this tool to execute Python code.
The environment is a stateful Jupyter notebook.
Use `print()` to output results.
```

**v13p2, v13p3, v13p4 — Defensive:**
```
Use this tool to execute Python code.
The environment is a stateful Jupyter notebook.
Check if a result is not None before you access it.
Ensure all variables and helper functions are available before use.
Use `print()` to output results.
```

The two added lines in the "Defensive" prompt were intended to address `NoneType` and `NameError` exceptions. However, empirical analysis of v13p4 showed these errors actually increased or remained steady, suggesting the model ignored the "soft" constraints of the prompt. **v13p5 reverted to the Minimal prompt.**

---

## 5. System Messages: Timeout & No-Output

| Message | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Timeout | `[ERROR]...` | `[ERROR]...` | `Execution...` | `Execution...` | `Execution...` | `Execution...` | `Execution...` | `Execution...` | `Execution...` | `Execution...` |
| No output | `[WARNING]...` | `[WARNING]...` | `No output...` | `No output...` | `No output...` | `No output...` | `No output...` | `No output...` | `No output...` | `No output...` |

v13p2 drops the `[ERROR]` and `[WARNING]` severity prefixes. Every preview and version that follows retains this cleaner format.

---

## 6. Error Detection Logic (`_handle_tool_call`)

| Version | Detection Logic | Timeout Detection |
|---|---|:---:|
| v12 | `startswith('[ERROR]') or 'Traceback' in text or 'Error:' in text` | ✅ Works |
| v13p1 | `startswith('[ERROR]') or 'Traceback' in text or 'Error:' in text` | ✅ Works |
| v13p2 | `startswith('[ERROR]') or 'Traceback' in text or 'Error:' in text` | ❌ **BROKEN** |
| v13p3 | `'Execution timed out' in text or 'Error:' in text` | ✅ **FIXED** |
| v13p4 | `'Execution timed out' in text or 'Error:' in text` | ✅ **FIXED** |
| v13p5 | `'Execution timed out' in text or 'Error:' in text` | ✅ **FIXED** |
| v14p1 | `'Execution timed out' in text or 'Error:' in text` | ✅ **FIXED** |
| v14p2 | `'Execution timed out' in text or 'Error:' in text` | ✅ **FIXED** |
| v15p1 | `'Execution timed out' in text or 'Error:' in text` | ✅ **FIXED** |
| v15p2 | `'Execution timed out' in text or 'Error:' in text` | ✅ **FIXED** |

**v13p2 Bug:** When the `[ERROR]` prefix was removed from timeout messages, the detection logic was not updated. Timeouts occurred but were **not counted** as errors.
**v13p3+ Fix:** Explicit check for `'Execution timed out'` string catches timeouts regardless of prefix.

---

## 7. Empirical Error Statistics (Problem #10 only)

| Metric | v12 | v13p1 | v13p2* | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Total errors logged | 36 | 33 | 13* | 29 | **56** | 27 | 32 | 29 | 32 | 32 |
| Total error chars | 13,083 | 12,073 | 3,662 | 1,815 | 3,141 | **1,401** | 3,920 | 7,481 | 2,378 | 5,023 |
| Average chars/error | 363.4 | 365.8 | 281.7 | 62.6 | 56.1 | **51.9** | 122.5 | 258.0 | 74.3 | 157.0 |
| Timeout errors | 16 | 14 | 0* | 9 | 28 | 9 | 19 | 4 | 5 | 17 |
| Reduction vs v12 | - | 0% | - | 83% | 85% | **86%** | 66% | 29% | 82% | 53% |

**\*v13p2 Timeout Bug:** The "0 timeouts" figure is incorrect due to the detection bug.

**Analysis:**
*   **v13p4:** Highest error volume (56). The extremely short error messages likely allowed the model to fail-fast and retry many times within the budget.
*   **v13p5:** Lowest error volume (27) and lowest verbosity (51.9). Reverting to the minimal prompt correlated with a significant drop in errors compared to v13p4, suggesting the defensive prompt may have been counter-productive.
*   **v14p1:** Higher error volume (32) and significantly higher timeout rate (19), with increased verbosity (122.5 chars/error) due to the retention of stack traces.

---

## 8. Error Type Breakdown (All Problems)

#### Problem 1

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Timeout` | 2 | 2 | 0 | 1 | 2 | 3 | 2 | 2 | 1 | 1 |
| `SympifyError` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `TypeError` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `NonlinearError` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Total** | **2** | **3** | **0** | **1** | **3** | **3** | **2** | **3** | **2** | **1** |

#### Problem 2

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Timeout` | 0 | 7 | 0 | 3 | 1 | 1 | 0 | 14 | 2 | 3 |
| `TypeError` | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| `ModuleNotFoundError` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 0 | 3 |
| `NameError` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| `IndexError` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `ValueError` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `AttributeError` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `KeyError` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| **Total** | **0** | **9** | **1** | **6** | **1** | **1** | **0** | **22** | **2** | **9** |

#### Problem 3

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Timeout` | 3 | 3 | 0 | 3 | 6 | 1 | 1 | 5 | 5 | 13 |
| `NameError` | 2 | 1 | 1 | 5 | 0 | 0 | 4 | 0 | 1 | 2 |
| `AttributeError` | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |
| `KeyError` | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `TypeError` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 |
| `ValueError` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| `UnboundLocalError` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Total** | **5** | **6** | **2** | **9** | **8** | **2** | **5** | **7** | **13** | **15** |

#### Problem 4

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Timeout` | 4 | 5 | 0 | 7 | 4 | 7 | 8 | 6 | 6 | 8 |
| `NameError` | 0 | 0 | 2 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| `KeyError` | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ValueError` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `IndexError` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `AssertionError` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `TypeError` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Total** | **6** | **7** | **3** | **8** | **4** | **8** | **10** | **7** | **7** | **8** |

#### Problem 5

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Timeout` | 3 | 6 | 0 | 4 | 6 | 2 | 0 | 4 | 0 | 1 |
| `NameError` | 2 | 2 | 5 | 3 | 0 | 1 | 0 | 1 | 3 | 2 |
| `TypeError` | 0 | 2 | 3 | 1 | 1 | 1 | 1 | 3 | 2 | 1 |
| `AttributeError` | 3 | 1 | 2 | 2 | 0 | 0 | 1 | 0 | 0 | 2 |
| `KeyError` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `RecursionError` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **10** | **11** | **10** | **10** | **7** | **4** | **2** | **8** | **5** | **6** |

#### Problem 6

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `AttributeError` | 1 | 0 | 3 | 2 | 1 | 1 | 2 | 1 | 0 | 1 |
| `Timeout` | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| `ValueError` | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 2 | 1 | 0 |
| `OverflowError` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **2** | **1** | **5** | **3** | **1** | **2** | **5** | **3** | **1** | **1** |

#### Problem 7

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Timeout` | 17 | 16 | 0 | 10 | 6 | 17 | 21 | 18 | 15 | 11 |
| `TypeError` | 5 | 10 | 16 | 15 | 25 | 13 | 5 | 4 | 15 | 5 |
| `NameError` | 9 | 5 | 4 | 5 | 2 | 8 | 6 | 5 | 13 | 7 |
| `AttributeError` | 10 | 3 | 6 | 1 | 2 | 8 | 8 | 4 | 5 | 4 |
| `ValueError` | 5 | 4 | 6 | 2 | 12 | 5 | 3 | 2 | 17 | 3 |
| `IndexError` | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 0 |
| `KeyError` | 0 | 2 | 0 | 0 | 1 | 1 | 1 | 0 | 3 | 0 |
| `ModuleNotFoundError` | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `Unknown` | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ZeroDivisionError` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| `UFuncTypeError` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `RecursionError` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **51** | **45** | **33** | **35** | **48** | **53** | **44** | **35** | **74** | **30** |

#### Problem 8

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `ValueError` | 2 | 1 | 5 | 0 | 1 | 0 | 4 | 3 | 0 | 2 |
| `Timeout` | 3 | 1 | 0 | 1 | 3 | 0 | 6 | 1 | 1 | 1 |
| `KeyError` | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 1 | 0 | 1 |
| `NameError` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `RecursionError` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `AttributeError` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `IndexError` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| **Total** | **5** | **4** | **5** | **2** | **4** | **3** | **10** | **6** | **1** | **6** |

#### Problem 9

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `AttributeError` | 6 | 7 | 1 | 0 | 9 | 6 | 8 | 2 | 6 | 7 |
| `Timeout` | 6 | 7 | 0 | 4 | 2 | 5 | 11 | 3 | 14 | 8 |
| `TypeError` | 1 | 4 | 2 | 0 | 0 | 8 | 6 | 4 | 2 | 2 |
| `IndexError` | 0 | 0 | 1 | 3 | 5 | 0 | 0 | 0 | 0 | 0 |
| `PolynomialError` | 1 | 0 | 0 | 0 | 2 | 1 | 1 | 2 | 3 | 3 |
| `ValueError` | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| `AssertionError` | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `NameError` | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| `KeyError` | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **18** | **20** | **5** | **9** | **19** | **21** | **27** | **11** | **26** | **21** |

#### Problem 10

| Error type | v12 | v13p1 | v13p2 | v13p3 | v13p4 | v13p5 | v14p1 | v14p2 | v15p1 | v15p2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Timeout` | 16 | 14 | 0 | 9 | 28 | 9 | 19 | 4 | 5 | 17 |
| `ValueError` | 4 | 2 | 6 | 4 | 8 | 6 | 4 | 17 | 8 | 7 |
| `KeyError` | 11 | 8 | 5 | 6 | 1 | 8 | 4 | 1 | 6 | 2 |
| `TypeError` | 3 | 6 | 1 | 6 | 8 | 1 | 3 | 4 | 3 | 3 |
| `NameError` | 2 | 2 | 1 | 3 | 6 | 3 | 1 | 1 | 5 | 1 |
| `AttributeError` | 0 | 1 | 0 | 1 | 4 | 0 | 0 | 2 | 4 | 0 |
| `OverflowError` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| `IndexError` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `ZeroDivisionError` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `UnboundLocalError` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **Total** | **36** | **33** | **13** | **29** | **56** | **27** | **32** | **29** | **32** | **32** |

---

## 9. Additional Analysis

### Failure of Defensive Prompts
The comparison between v13p4 (Defensive Prompt) and v13p5 (Minimal Prompt) suggests that adding instructions like *"Check if a result is not None before you access it"* was ineffective. In Problem 7 (Geometry), v13p4 produced **25 TypeErrors** (mostly `NoneType` unpacking), while earlier versions with minimal prompts produced fewer. The model appears to prioritize the mathematical solving logic over prompt-based code safety guidelines.

### Success on Problem #10
Problem #10 was solved successfully in most of he runs runs (v12 through v13p5 - not v14p1). Ground truth is **8687** and it received 1 or 2 votes (2 votes was the most common case).
In v13p5, the voting mechanism produced a tie:
*   Answer 8687: 2 votes
*   Wrong Answer: 2 votes
The **Inverse Entropy Weight (IEW)** score for 8687 was higher (indicating lower model uncertainty), allowing the system to correctly select it. This confirms the robustness of the scoring strategy.
