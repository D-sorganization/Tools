# Assessment: Confirmed Vulnerabilities from Thesis Defense

**Date:** 2026-02-24
**Status:** Open
**Labels:** `jules:assessment`, `needs-attention`, `security`, `thesis-defense`, `needs-work`

## Overview

During the "Thesis Defense" analysis of the Adversarial Project Reviews, several critiques were verified as **Valid** and require immediate engineering attention.

## 1. Zip Bomb Vulnerability (Folder Packer Pro)

**Location:** `src/tools/folder_tools/folder_packer_pro/folder_packer_pro.py`
**Critique ID:** FPP-02

**Description:**
The `_run_unpack` method reads `encoded_content` from the JSON package, decodes it via `base64`, and writes it to disk. There is **no check** for the size of the decoded content relative to the encoded size (compression ratio) or an absolute maximum size limit.

**Risk:**
A malicious actor could craft a small JSON package (few KB) that decodes into gigabytes of data (e.g., a string of zeros), exhausting disk space and crashing the application or system.

**Remediation:**

- Implement a `MAX_FILE_SIZE` constant.
- Check `len(content)` before writing.
- Abort operation if total unpacked size exceeds a threshold.

## 2. Path Traversal Vulnerability (Folder Packer Pro)

**Location:** `src/tools/folder_tools/folder_packer_pro/folder_packer_pro.py`
**Critique ID:** FPP-03

**Description:**
The code constructs file paths using `file_path = dest_path / rel_path` where `rel_path` is taken directly from the JSON package.

```python
file_path = dest_path / rel_path
file_path.parent.mkdir(parents=True, exist_ok=True)
```

If `rel_path` is `../../etc/passwd` (or Windows equivalent), the `file_path` will point outside the intended `dest_path`.

**Risk:**
Arbitrary file overwrite outside the destination directory.

**Remediation:**

- Use `resolve()` and `relative_to()` check.

```python
final_path = (dest_path / rel_path).resolve()
if not final_path.is_relative_to(dest_path.resolve()):
    raise SecurityError("Path traversal detected")
```

## 3. Unbounded Computation (Calculator)

**Location:** `src/web_applications/calculator/calculator.py`
**Critique ID:** CALC-02

**Description:**
The `TI89Calculator._evaluate_cached` method calls `sp.simplify(substituted)`. SymPy's simplification algorithms can have high time complexity for certain expressions. There is no timeout mechanism wrapping this call.

**Risk:**
Denial of Service (DoS) by hanging the worker process with complex expressions.

**Remediation:**

- Wrap the evaluation in a `multiprocessing` task with a strict timeout (e.g., 2 seconds).
- Or use `signal.alarm` (Unix only) or a thread-based timeout (less reliable for CPU-bound tasks).
