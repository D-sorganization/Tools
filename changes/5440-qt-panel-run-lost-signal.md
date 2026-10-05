---
issue: 5440
summary: "test_solver_gui waits on panel state instead of a signal that may already have fired, caps Qt waits below the pytest timeout, and dumps stacks on a hang (faulthandler_timeout=50)"
dl_state: "in_review"
next_step: "Apply the same state-based wait to tests/rate_of_closure/test_variation_gui_execution.py:369-371."
title: "Qt Panel-Run Test Hang Under Xdist"
branch: "fix/5440-qt-panel-run-hang"
paths: "tests/rate_of_closure/test_solver_gui.py,pyproject.toml"
---
