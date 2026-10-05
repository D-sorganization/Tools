---
issue: 5440
summary: "Variation GUI execution tests wait on tab state connected before start instead of a post-start signal wait; shared Qt wait helpers"
dl_state: "in_review"
next_step: "Merge the PR, then root-cause any future hang from its faulthandler stack dump."
title: "Variation GUI Lost Signal"
branch: "fix/5440-variation-gui-lost-signal"
paths: "tests/rate_of_closure/test_variation_gui_execution.py,tests/rate_of_closure/test_solver_gui.py,tests/rate_of_closure/_qt_waits.py"
---
