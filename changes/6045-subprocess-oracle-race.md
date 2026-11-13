---
issue: 6045
summary: "Stabilize CLI subprocess timeout tree-kill oracle and Windows test portability"
---

- Wait boundedly for process and grandchild exit (treating Linux `/proc/<pid>/stat` `'Z'` zombie state as terminated) in `tests/unit/ai/adapters/test_cli_subprocess_runner_oracles.py` so `test_oracle_01_timeout_kills_grandchild_tree` does not race container PID-1 reaping under pytest-xdist load (`#6045`).
- Normalize CRLF line endings in `test_oracle_10_encoding_roundtrip` and handle Windows `CreateProcess` `OSError` / regex escaping in `test_oracle_12_missing_and_non_executable_binary`.
- Add unit coverage (`test_is_zombie_proc_parses_proc_stat`) for `/proc/<pid>/stat` parsing including process names containing spaces and parentheses.
