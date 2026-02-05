# Assessment: Security (Category F)

## Grade: 4/10

## Analysis
**CRITICAL FINDINGS**:
1.  **Data Leakage**: `.msg` (Outlook email) files found in `src/shared/python/upstream_drift_tools/...`. This is a major PII/IP risk.
2.  **Unsafe Functions**: `eval()` usage detected in `Data_Processor_r0.py`, `signal_processing.py`, and `fitting.py`.
3.  **Shell Injection**: Extensive use of `shell=True` in launcher scripts.
4.  **SAST**: CodeQL workflow is present but disabled (`codeql-analysis.yml.disabled`).
