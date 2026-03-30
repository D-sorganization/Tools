# Code Quality Issue: eval() or exec() used
## Location
- **File**: `src/shared/python/upstream_drift_tools/bootstrap.py`
- **Severity**: CRITICAL

## Description
A CRITICAL code quality issue was detected during the automated review.
eval() or exec() used

## Code Snippet
```python
    exec((_root / "src" / "shared" / "python" / "upstream_drift_tools" / "bootstrap.py").read_text())
```
