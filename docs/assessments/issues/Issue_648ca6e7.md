# Code Quality Issue: eval() or exec() used

## Location

- **File**: `src/pendulum_simulator/tests/test_main_window.py`
- **Severity**: CRITICAL

## Description

A CRITICAL code quality issue was detected during the automated review.
eval() or exec() used

## Code Snippet

```python
            def exec(self) -> Any:
```
