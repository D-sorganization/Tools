# Code Quality Issue: eval() or exec() used

## Location

- **File**: `src/shared/python/calc_backend/tests/test_calc_backend.py`
- **Severity**: CRITICAL

## Description

A CRITICAL code quality issue was detected during the automated review.
eval() or exec() used

## Code Snippet

```python
                return eval(expression, {}, namespace)  # noqa: S307
```
