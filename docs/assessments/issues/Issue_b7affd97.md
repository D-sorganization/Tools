# Code Quality Issue: eval() or exec() used

## Location

- **File**: `src/shared/python/calc_backend/routers/ode_solver.py`
- **Severity**: CRITICAL

## Description

A CRITICAL code quality issue was detected during the automated review.
eval() or exec() used

## Code Snippet

```python
    Uses AST-validated safe_eval instead of raw eval().  Math function names
```
