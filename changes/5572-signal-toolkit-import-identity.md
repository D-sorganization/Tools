---
issue: 5572
summary: "Bind Signal toolkit import aliases before package children capture provider classes."
dl_state: "in_review"
next_step: "Complete governed log migration, full pre-PR validation, and protected provider publication."
title: "Preserve Signal Toolkit Import Identity"
owner: "codex"
branch: "codex/signal-optimized-structural-guards"
paths: "src/shared/python/signal_toolkit/__init__.py, tests/shared/python/signal_toolkit/test_import_identity.py, manuals/tools/manifests/module-inventory.json"
---

Fresh-process bare-first, canonical-first, and legacy-src-first imports now share the same package, core, fitting, I/O, and Signal objects. A separate downstream src namespace remains intact.
