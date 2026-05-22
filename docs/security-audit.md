# Security Audit Report

**Issue:** #2418 — [Epic] Adversarial Review Remediation - Complete Assessment
**Scope:** Tools monorepo `src/` (Python)
**Date:** 2026-05-01
**Auditor:** address-issues agent (automated)
**Status:** No critical findings; existing mitigations are sound

---

## Executive Summary

A pass over `src/` was performed covering:

1. Hardcoded secrets / API keys / passwords
2. Unsafe deserialisation (pickle, `yaml.load` without `safe_load`)
3. Path traversal vulnerabilities
4. Unused / dead code increasing attack surface
5. Dependencies with known CVEs

**No critical vulnerabilities were found.** Prior work (issues #2407, #2356, CHANGELOG
entry "Security: Replaced shell=True subprocess calls") has already addressed the
highest-severity items. Findings below are informational or low-risk.

---

## 1. Hardcoded Secrets

**Result: PASS**

Automated scan using `src/python/src/utils/secrets_scanner.py` against the full `src/`
tree returned **0 non-false-positive findings**.

Key patterns checked:

- AWS Access Key (`AKIA…`)
- GitHub Tokens (`ghp_…`)
- OpenAI keys (`sk-…`)
- Generic `password = "…"` / `api_key = "…"` assignments
- Base64-encoded private keys (`-----BEGIN PRIVATE KEY-----`)

API key handling in `src/document_processing/pdf_renamer/` correctly reads from
environment variables and the OS keyring (via `keyring`), with no fallback to
any hardcoded value.

---

## 2. Unsafe Deserialisation

**Result: PASS**

### Pickle

Pickle support was explicitly **disabled** in both data-access modules:

- `src/data_processing/data_processor/python/data_processor/file_utils.py` —
  `DataReader.read_file` raises `ValueError("Pickle format is disabled for security
reasons (CWE-502).")` when `format_type="pickle"` is requested.
- `src/shared/python/upstream_drift_tools/data_processing/io.py` — identical
  guard in `DataReader.read_file` and `DataWriter.write_file`.

NumPy `.npy` files are loaded with `allow_pickle=False` (explicit).

### YAML

No call to `yaml.load` (without `Loader=`) was found in `src/`. All YAML loading
uses `yaml.safe_load` or goes through `defusedxml` (for XML variants).
`PyYAML>=6.0` is pinned in `pyproject.toml`; the `FullLoader` default regression
introduced in earlier versions does not apply.

---

## 3. Path Traversal

**Result: LOW RISK**

File I/O in the data-processing and document-processing modules accepts
user-supplied paths via CLI arguments or GUI file-chooser dialogs. No
`..`-stripping or canonical-path enforcement was observed, but:

- The tools run locally (not exposed as a network service).
- File-chooser dialogs constrain selection to the local filesystem via the OS.
- CLI entry points (`argparse`) do not serve untrusted input.

**Recommendation (non-blocking):** For any future web-facing use of the
file-read helpers, wrap `file_path = Path(file_path).resolve()` against an
allowed base directory before opening.

---

## 4. Unused / Dead Code

**Result: LOW RISK**

The `src/python/src/utils/` tree contains several debug helpers
(`debug_helpers.py`, `debug_memory.py`, `debug_profiling.py`, `debug_tracing.py`)
that are not imported by any production code path in `src/`. These files expand
the attack surface (additional import hooks, potential for unvetted third-party
calls if a developer adds one) but pose no immediate risk because they are never
loaded at runtime.

**Recommendation:** Track removal of unused debug utilities in a separate issue
or gate them behind `if __debug__:` blocks.

---

## 5. Dependencies with Known CVEs

**Result: PASS (no actionable CVEs found)**

All core dependencies specify minimum versions that post-date their known CVE
disclosures:

| Package      | Minimum pinned | Notes                                              |
| ------------ | -------------- | -------------------------------------------------- |
| `numpy`      | `>=2.0.1`      | CVE-2021-41495 fixed in 1.21.0; 2.x not affected   |
| `PyYAML`     | `>=6.0`        | CVE-2017-18342 fixed in 5.1; 6.x not affected      |
| `defusedxml` | `>=0.7.0`      | Designed for safe XML parsing; no open CVEs        |
| `scipy`      | `>=1.13.1`     | No relevant open CVEs in this range                |
| `pandas`     | `>=2.2.2`      | No relevant open CVEs in this range                |
| `flask`      | `>=3.0.0`      | CVE-2023-30861 fixed in 2.3.2; 3.x not affected    |
| `Jinja2`     | `>=3.1.4`      | CVE-2024-22195 fixed in 3.1.3; 3.1.4+ not affected |

The `requirements-lock.txt` file pins exact transitive versions for
reproducible installs. Re-run `pip install -r requirements.txt && pip freeze >
requirements-lock.txt` after any dependency upgrade to refresh the lock.

---

## Existing Security Controls (Previously Implemented)

| Control                                      | File / Location                           | Issue |
| -------------------------------------------- | ----------------------------------------- | ----- |
| `shell=True` removed from `subprocess` calls | `UnifiedToolsLauncher.py`, `Launcher.py`  | #2407 |
| Pickle disabled (CWE-502)                    | `file_utils.py`, `io.py`                  | #2407 |
| `safe_eval` sandboxed expression evaluator   | `src/shared/python/safe_eval.py`          | #2407 |
| Secrets scanner utility                      | `src/python/src/utils/secrets_scanner.py` | #2407 |
| OWASP-safe test credential naming convention | `SECRETS_MANAGEMENT.md`                   | #2356 |
| OS keyring for interactive API-key storage   | `pdf_renamer/config.py`                   | #2356 |
| `detect-secrets` baseline committed          | `.secrets.baseline`                       | #2407 |
| `defusedxml` for XML parsing                 | `pyproject.toml`                          | #2407 |

---

## Recommendations (Non-Blocking)

1. **Path sanitisation for future web endpoints.** If any calculator or
   data-processing function is ever exposed over HTTP, add `Path.resolve()`
   against a strict base directory before any file open.

2. **Dependency audit cadence.** Add `pip-audit` or `safety check` as a
   periodic CI step (e.g., weekly scheduled run) to catch newly disclosed CVEs
   in the dependency tree before they accumulate.

3. **Dead-code reduction.** Remove or gate the `src/python/src/utils/debug_*`
   modules behind an explicit `DEBUG_TOOLS` opt-in to reduce the module
   import surface.

---

_This report was generated by automated analysis. For a full adversarial review
with manual exploitation testing, see `ADVERSARIAL_REVIEW_COMPLETE.md` and the
`docs/assessments/` archive._
