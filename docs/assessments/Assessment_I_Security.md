# Assessment I Results: Security & Input Validation

## Executive Summary

- **CRITICAL VULNERABILITY**: `eval()` is used on user-supplied input in `Data_Processor_r0.py`. This allows arbitrary code execution.
- **Dynamic Execution**: `exec()` is used in `verify_installation.py` to check imports, which is unsafe practice.
- **Secrets Management**: Potential exposure of API keys in `API_KEY_QUICK_REFERENCE.txt`.
- **Dependency Risks**: Unpinned dependencies in `requirements.txt` allow supply chain attacks.

## Top 10 Security Risks

1.  **Remote Code Execution (Blocker)**: `eval()` in `Data_Processor_r0.py`.
2.  **RCE via Fitting (Blocker)**: `eval()` in `fitting.py`.
3.  **Secrets in Git (Critical)**: `API_KEY_QUICK_REFERENCE.txt` might be committed.
4.  **Unsafe Deserialization (Major)**: Pickle usage? (Not checked, but common in numpy).
5.  **Path Traversal (Major)**: File tools take paths without validation.
6.  **XSS (Minor)**: Web apps might be vulnerable (Unit Converter scanned safe, but others?).
7.  **Supply Chain (Moderate)**: No lockfiles.
8.  **Least Privilege (Minor)**: Scripts run as user.
9.  **Hardcoded Credentials (Major)**: Check `.env.example` vs `.env`.
10. **Insecure Randomness (Minor)**: Use of `random` instead of `secrets`.

## Scorecard

| Category                 | Score | Evidence & Remediation                                                                 |
| ------------------------ | ----- | -------------------------------------------------------------------------------------- |
| Code Execution           | 0/10  | `eval()` present. **Fix**: REMOVE IMMEDIATELY.                                         |
| Data Protection          | 4/10  | Secrets potentially exposed.                                                           |
| Input Validation         | 3/10  | Minimal.                                                                               |
| Dependency Security      | 3/10  | No audit.                                                                              |

## Findings Table

| ID    | Severity | Category | Location                 | Symptom            | Root Cause | Fix                  | Effort |
| ----- | -------- | -------- | ------------------------ | ------------------ | ---------- | -------------------- | ------ |
| I-001 | Blocker  | RCE      | `Data_Processor_r0.py`   | `eval(formula)`    | User feature | `numexpr`            | M      |
| I-002 | Critical | Secrets  | `src/.../API_KEY...txt`  | API Key file       | User error | `.gitignore`         | S      |

## Refactoring Plan

**IMMEDIATE (Now):**
-   Remove `eval()` usage.
-   Ensure `API_KEY_QUICK_REFERENCE.txt` is gitignored or deleted.

**2 Weeks:**
-   Implement `pip-audit` in CI.
-   Sanitize all file inputs.

## Diff-Style Suggestions

```python
# fitting.py
<<<<<<< SEARCH
    return eval(expression, {"__builtins__": {}}, local_dict)
=======
    # SECURE: Use asteval or similar restricted evaluator
    from asteval import Interpreter
    aeval = Interpreter()
    return aeval(expression, local_dict)
>>>>>>> REPLACE
```
