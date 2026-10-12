---
issue: 5569
summary: "Keep Signal structural shape validation active under optimized Python"
dl_state: "in_review"
next_step: "Review and merge Tools fix, then repin and mirror the provider in UpstreamDrift #7704"
branch: "codex/signal-optimized-structural-guards"
paths: "src/shared/python/signal_toolkit/core.py; tests/shared/python/signal_toolkit/test_core_structural_guards.py; manuals/tools/manifests/module-inventory.json; manuals/tools/manifests/module-inventory/entries-src-shared-python-signal-toolkit.json"
---

Three initial `python -O` regression cases were red; all four structural cases now pass. The constructor retains `PreconditionError` for normal DbC consumers. The Tools design-manual source inventory was refreshed; no numerical algorithm or output changed.
