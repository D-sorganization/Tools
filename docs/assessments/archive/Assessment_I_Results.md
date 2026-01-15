# Assessment I Results: Security & Input Validation

## Executive Summary

-   **Strong Headers**: `webapp.py` implements CSP, HSTS, X-Frame-Options.
-   **Input Validation**: `webapp.py` has a blocklist for dangerous keywords (`eval`, `exec`) and validates length.
-   **Safe Execution**: `UnifiedToolsLauncher` uses `subprocess.Popen` without `shell=True` (mostly) and explicit paths.
-   **Rate Limiting**: Implemented in Calculator app.
-   **Secrets**: No secrets found in code.

## Top 10 Security Risks

1.  **SymPy Eval (Severity: Medium)**: Calculator uses `sympy.parse_expr`. Even with `evaluate=False` and strict locals, symbolic math engines can be tricky. However, "Bolt Optimization" suggests care was taken.
2.  **MATLAB Injection (Severity: Low)**: Launcher constructs MATLAB command string.
3.  **CSV Injection (Severity: Low)**: Data processor handling of CSVs.
4.  **Prototype Pollution (Severity: Low)**: Unit converter prevents `__proto__`.
5.  **Dependencies (Severity: Low)**: Need to audit `package.json` and `requirements.txt`.
6.  **Local Execution (Severity: Low)**: Tools run locally, reducing attack surface.
7.  **File Access (Severity: Medium)**: Folder tools can manipulate any file the user has access to.
8.  **Updates (Severity: Low)**: How are security updates applied?
9.  **Denial of Service (Severity: Low)**: Rate limiting mitigates this for web app.
10. **XSS (Severity: Low)**: CSP prevents inline scripts.

## Scorecard

| Category             | Score | Evidence & Remediation                                    |
| -------------------- | ----- | --------------------------------------------------------- |
| Input Validation     | 9/10  | Strict checks in calculator.                              |
| Injection Prevention | 8/10  | Blocklists and parameterized queries (if SQL used).       |
| Authentication       | N/A   | Local tools.                                              |
| Headers / Config     | 10/10 | Flask security headers are exemplary.                     |
| Dependency Safety    | 8/10  | Standard management.                                      |

## Findings Table

| ID    | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
| ----- | -------- | -------- | -------- | ------- | ---------- | --- | ------ |
| I-001 | Low      | Security | `UnifiedToolsLauncher.py` | MATLAB cmd string construction | String formatting | Use strict args | S |

## Refactoring Plan

**48 Hours**:
-   None.

**2 Weeks**:
-   Run `bandit` security scan on Python code.

**6 Weeks**:
-   Perform a deep dive pentest on the Calculator parser.
