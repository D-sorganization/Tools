# Assessment: Logging

## Grade: 4/10

## Analysis
Logging is inconsistent:
- **Print Usage**: Approx 400 `print()` statements exist in the codebase, violating the `AGENTS.md` directive to use the `logging` module.
- **Logger Config**: A `logging_config.py` exists, but its usage is spotty.
- **Error Tracing**: Without proper logging, debugging failures in CI or production is difficult.

## Recommendations
1. **Ban Print**: Configure a custom lint rule to forbid `print()` in `src/`.
2. **Universal Logger**: Ensure every module instantiates a named logger (`logger = logging.getLogger(__name__)`).
