---
issue: 5472
summary: "Repair canonical manual source-count regression while preserving release blockers"
dl_state: "in_review"
next_step: "Complete PR5473 CI; retain artifact integrity and native provider qualification gates"
branch: "fix/canonical-manual-count-5472"
---

PDF line wrapping differs between supported extractors; preserve recorded line
counts as diagnostics while verifying byte identity and semantic page inventory.
