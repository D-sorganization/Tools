---
issue: 5727
summary: "Add CLI turn cancellation, incremental idle timeout, parent-death reaper, and typed lifecycle outcomes (Wave 2.2)"
---

Implements Epic #5727 Wave 2.2 (`#6023` item 2.2; Fable slices 2, 4, 5):
- `CliCancelToken` & `cancel()` on CLI adapters + `StreamWorker` (`<2s` process tree termination).
- Incremental `read1()` capture with `idle_timeout` partial output preservation (`AIIdleTimeoutError`).
- Parent-exit orphan prevention (Windows Job Object / Linux `PR_SET_PDEATHSIG`).
- Typed non-zero exit errors (`AIAuthenticationError`, `AIUsageExhaustedError`, `AIRateLimitError`) and canonical `LifecycleSnapshot` states (`SUCCEEDED`, `FAILED`, `CANCELLED`).
