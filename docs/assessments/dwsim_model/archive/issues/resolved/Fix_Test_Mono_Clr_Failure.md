---
title: "Resolved: Tests fail due to missing clr module"
labels: ["jules:resolved"]
---

Fixes #0

The tests fail because they run without mono/clr on non-Windows environments. This is expected based on the codebase memory, which specifies that tests without mono require mocking the clr module and `get_automation` function in `tests/conftest.py`. The user has asked for a pure Assessment feature, so no further changes are needed for this specific test failure, but it is documented here.
