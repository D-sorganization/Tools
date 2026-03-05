# Assessment F: Security

## Executive Summary
This is a detailed analysis based on the latest codebase metrics (2026-03-05).
**CRITICAL FINDINGS**:
1. **Data Leakage**: `.msg` (Outlook email) files found in `src/shared/python/upstream_drift_tools/...`. This is a major PII/IP risk.
2. **Unsafe Functions**: 2 instances of `eval()` usage detected in legacy tools.
3. **Shell Injection**: Extensive use of `shell=True` in launcher scripts. Score: 4.0/10.

## Scorecard
- Grade: 4.0/10

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| F-001 | High | Security | Codebase | IP/PII Exposure | Committed `.msg` binaries | git filter-repo | H |

## Refactoring Plan
- Address F-001 by implementing the recommended fix (git filter-repo).
- Continue monitoring metrics via the `scripts/generate_fresh_assessments.py` CI step.
