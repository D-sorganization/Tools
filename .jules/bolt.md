## 2026-07-27 - Optimize parsing by avoiding split/join array allocations
**Learning:** In high-frequency rendering paths or form parsing, chained array methods like `.split(':')` and `.join(':')` create unnecessary intermediate arrays and garbage collection overhead. This is especially true when parsing simple string formats like key-value pairs where values might contain multiple colons.
**Action:** Replaced `.split(':')` and `.join(':')` with a single-pass loop combined with `indexOf(':')` and `substring()` to improve performance and avoid allocations.
