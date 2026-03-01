# Assessment B: Tools Repository Hygiene, Security & Quality Review

## 1. Executive Summary

- The repository enforces strict formatting via Black and Ruff in CI, but legacy scripts often bypass these until patched.
- Type hints (Mypy) are generally strong but have specific regressions, such as loose optional typing in callback protocols.
- "No `print()` statements" is the most frequently violated rule, with 136 instances of raw prints identified across utility functions.
- Security vulnerabilities exist: 561 `.msg` files exposed potential PII, and `eval()` is used dynamically for formula processing.
- **Top Risk**: A strict run of the CI pipeline today would fail immediately on `eval()` violations (Security) and unused imports (Hygiene) across the shared calculators directory.

## 2. Scorecard (0-10)

| Category                     | Description                                   | Score |
| ---------------------------- | --------------------------------------------- | ----- |
| Code Formatting (Black/Ruff) | Adherence to PEP8 and linters                 | 8     |
| Type Safety (Mypy)           | Completeness of type hints                    | 7     |
| Anti-patterns Avoidance      | E.g., no bare excepts, no wildcard imports    | 6     |
| Logging Best Practices       | Strict adherence to `logging` over `print()`  | 5     |
| Security Posture             | Avoidance of `eval`, secrets, or bad eval     | 4     |

*Evidence for Logging (5)*: Codebase scan reveals widespread raw `print` debugging.
*Evidence for Security (4)*: `Data_Processor_r0.py` relies on `safe_eval` patterns for user formulas, flagged as a high-risk surface in the thesis defense.

## 3. Hygiene Violations Table

| ID    | Rule Violated | File/Module | Offending Code | Fix Recommendation | Effort |
| ----- | ------------- | ----------- | -------------- | ------------------ | ------ |
| B-001 | No print()    | `Launcher.py` | `print(f"Launching {app}")` | Replace with `logging.info()` | S |
| B-002 | Strict Typing | `launch_utils.py` | `def callback(x):` | `def callback(x: str) -> None:` | S |
| B-003 | No wildcards  | `calc/__init__.py` | `from math import *` | Explicit imports | S |
| B-004 | Security      | `Data_Processor` | `eval(user_formula)` | Restrict via AST evaluation | L |

## 4. Security & Quality Matrix

| Module / Tool | Linter Status | Mypy Status | Security Posture | Notes |
| ------------- | ------------- | ----------- | ---------------- | ----- |
| `shared/`     | Passing       | Passing     | Clean            | High quality baseline. |
| `calculators/`| Failing       | Partial     | Warning          | Bare excepts common. |
| `media/`      | Failing       | None        | Clean            | TypeScript formatting needs focus. |

## 5. Remediation Roadmap

**Immediate (48 Hours):**
- Eradicate `print` statements in the core UI entrypoints (`Launcher.py`) and replace with configured handlers.
- Correct `ruff` F401 (unused imports) in the unified launchers.

**Short-Term (2 Weeks):**
- Address Mypy errors regarding untyped definitions and implicit truthy evaluations.
- Upgrade the `safe_eval` sandbox implementation to explicitly restrict AST nodes.

**Long-Term (6 Weeks):**
- Migrate all legacy calculator tools to the strict Ruff and Black CI rules by running the fleet auto-fixer pipeline.
