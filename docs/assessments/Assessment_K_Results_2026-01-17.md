# Assessment K Results: Reproducibility & Provenance

## Executive Summary

- **Scientific Code**: Solar system model and MATLAB scripts imply scientific rigor.
- **Reproducibility**: `requirements.txt` does not pin strict versions (has `>='`), which might hurt reproducibility.

## Scorecard

| Category | Score | Evidence |
| --- | --- | --- |
| Determinism | 8/10 | Code seems deterministic. |
| Version Tracking | 6/10 | Loose requirements. |

## Findings
- **K-001**: `requirements.txt` uses `>=`.

## Remediation
- Use `pip-tools` or `poetry.lock`.
