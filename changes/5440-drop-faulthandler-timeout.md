---
issue: 5440
summary: "Remove faulthandler_timeout=50 added by #5442: its stack dump fired inside legitimately long numerical tests and crashed the tests-shared shard on main (exit 249). A contract test keeps any future watchdog at or above the longest timeout marker."
dl_state: "in_review"
next_step: "Confirm tests-shared (3.12) is green on main after merge."
title: "Drop Faulthandler Timeout That Crashed Tests-Shared"
branch: "fix/5440-drop-faulthandler-timeout"
paths: "pyproject.toml,tests/architecture/test_faulthandler_timeout_contract.py"
---
