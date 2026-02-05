---
labels: jules:assessment, needs-attention
---

# CRITICAL: Data Leakage and Unsafe Eval Usage

The security assessment identified critical vulnerabilities:
1.  **Data Leakage**: Binary Outlook `.msg` files containing email correspondence are present in `src/shared/python/upstream_drift_tools/...`.
2.  **Unsafe Code**: `eval()` is used in `Data_Processor_r0.py` and others without sufficient sanitization.
3.  **SAST**: CodeQL is disabled.

**Action Items**:
-   Remove `.msg` files from history (BFG/filter-branch).
-   Refactor `eval()` usage to use `ast.literal_eval` or a math parser library.
-   Enable CodeQL workflow.
