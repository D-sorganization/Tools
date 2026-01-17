# Assessment I: Security & Input Validation Results

**Date:** 2026-01-14
**Assessor:** Jules

## 1. Code Injection Risks
**Score: 3/10**

*   **Incident**: The "Trojan Horse" commit (894f41c) demonstrates a failure in code review and merge controls.
    *   *Remediated*: Shadow workflows deleted. Misplaced tools moved.
*   **Vulnerability**: The repository was vulnerable to massive unchecked code dumps.

## 2. Input Validation
**Score: 6/10**

*   **Calculator**: Uses `FORBIDDEN_KEYWORDS` and regexes (memory). Good.
*   **Data Processor**: Uses `ast.parse` for formula security. Acceptable for internal tools, risky for public facing.
*   **Web Apps**: `unit_converter` (Client-side) seems safe but relies on JS.

## 3. Dependency Security
**Score: 5/10**

*   **Audit**: No automated `pip-audit` or `npm audit` in evidence in the standard workflow.
*   **Lock Files**: Missing for Python, meaning vulnerable versions could be installed silently.

## Remediation Roadmap
*   **Immediate**: Generate `requirements.txt` with hashes or use a lock file.
*   **Short-term**: Add `pip-audit` to the pre-commit checks.
*   **Long-term**: Enforce "Code Owners" and strict PR reviews to prevent another mass injection.
