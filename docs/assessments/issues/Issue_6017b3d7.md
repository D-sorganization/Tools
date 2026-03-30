# Code Quality Issue: eval() or exec() used
## Location
- **File**: `src/shared/python/scripting/scripting_env.py`
- **Severity**: CRITICAL

## Description
A CRITICAL code quality issue was detected during the automated review.
eval() or exec() used

## Code Snippet
```python
            exec(code, self.namespace)  # nosec B102
```
