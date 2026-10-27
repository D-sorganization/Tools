---
issue: 5828
summary: "Sort nominal pipe sizes by true physical diameter instead of string replacement (TOOLS-14)"
branch: "fix/5828-pipe-database-fraction-sort"
dl_state: "in_review"
next_step: "Merge the PR."
---

In `sidekick.process_calculators.pressure_drop_calculator.utils.pipe_database`, `list_available_sizes()` previously sorted nominal pipe sizes using `lambda x: float(x.replace("/", ".")) if "/" in x else float(x)`, which mapped `'1/2'` to `1.2` and `'3/4'` to `3.4`, placing `'1'` before `'1/2'` and `'3'` before `'3/4'`. The key is updated to evaluate fractional strings via `fractions.Fraction`, ensuring sizes are sorted in strictly ascending physical diameter (`'1/2'`, `'3/4'`, `'1'`, `'1.5'`, `'2'`, ...). Added comprehensive unit tests in `test_pipe_database.py`.
