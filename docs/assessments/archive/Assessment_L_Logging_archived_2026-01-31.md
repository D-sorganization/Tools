# Assessment: Logging (Category L)

## Grade: 4/10

## Analysis

Logging standards defined in `AGENTS.md` are not consistently followed:

1.  **Violation**: Legacy code relies heavily on `print()` statements for debug and status information.
2.  **Standard**: The `AGENTS.md` file explicitly forbids `print()` in favor of the `logging` module.
3.  **Observability**: Lack of structured logging makes debugging in production/CI environments difficult.

## Recommendations

1.  **Migrate to Logger**: Replace all `print()` calls in `src/` with `logger.info()`, `logger.debug()`, etc.
2.  **Configure Handlers**: Ensure a central logging configuration exists (console + file rotation).
