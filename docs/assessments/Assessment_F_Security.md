# Assessment F: Security

## Executive Summary
This assessment evaluates the security posture of the Tools repository, focusing on data hygiene, code injection vulnerabilities, and dependency safety (2026-03-05).
The repository has critical security failings. Most severely, 561 Outlook `.msg` files containing email correspondence are checked into the repository, posing a massive data leakage risk (IP/PII). Furthermore, instances of `eval()` are used for parsing mathematical formulas, opening the tools up to arbitrary code execution if exposed to untrusted user input. The project also lacks automated Static Application Security Testing (SAST).

## Scorecard
- **Grade: 4.0/10**

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| F-001 | Blocker | Data Leakage | `src/shared/python/upstream_drift_tools/` | 561 `.msg` binary files in repo | Careless `git add .` | Scrub history via `git filter-repo` | H |
| F-002 | Critical | Injection | `src/data_processing/data_processor/python/data_processor/core/formula_parser.py` (and legacy) | Arbitrary code execution | Unsafe `eval()` usage on user input | Migrate to `ast.literal_eval` or `sympy` | M |
| F-003 | Major | Injection | Folder Packer Pro / `pack_engine.py` | Path traversal risk | Trusting zip archive paths | Validate absolute paths before `os.system`/extraction | M |
| F-004 | Major | Web Security | `src/media_processing/video_processor/apps/web/lib/sanitize.ts` | XSS vulnerability (TODO) | Missing `DOMPurify` implementation | Implement the stubbed DOMPurify logic | S |

## Refactoring Plan
- **Short Term (Immediate)**: Remediate F-001 by completely excising the `.msg` files from the Git tree and adding `*.msg` to the root `.gitignore`.
- **Medium Term**: Remediate F-002 by auditing the 2 known instances of `eval()` and replacing them with safe alternatives like `sympy` for mathematical parsing, or `ast.literal_eval`.
- **Long Term**: Enable GitHub CodeQL scanning across the repository to automatically catch Path Traversal (F-003) and XSS (F-004) vulnerabilities in the future.
