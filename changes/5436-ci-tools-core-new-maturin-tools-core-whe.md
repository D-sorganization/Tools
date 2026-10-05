---
issue: 5436
summary: "ci(tools-core): new maturin-tools-core wheel-gate proves the crate's >=3.10 floor on Python 3.10-3.12; floor policy test now has no exemptions"
dl_state: "in_review"
next_step: "Merge PR #5438 once the wheel-gate is green on the fleet."
title: "Prove Tools-Core Python Floor in CI"
branch: "fix/5436-tools-core-wheel-gate"
paths: ".github/workflows/maturin-tools-core.yml,tests/ops/test_workflow_python_floor.py"
---
