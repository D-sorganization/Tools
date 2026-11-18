---
issue: 6023
summary: "Preserve caller-owned tracemalloc sessions and clean up owned memory trackers on failure"
---

Memory trackers reuse existing tracing without stopping it, allowing nested trackers to collect snapshots safely. Context-owned tracing is stopped even when measurement setup or snapshot collection fails. Add lifecycle, platform fallback, logging, and decorator tests; allocation statistics remain process-wide.
