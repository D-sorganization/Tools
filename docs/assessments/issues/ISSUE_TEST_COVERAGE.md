---
title: "Testing: Increase Unit Test Coverage for Shared Libraries"
labels: ["jules:assessment,needs-attention", "testing", "technical-debt"]
assignees: []
---

## Description
**Assessment Found**: Category C (Test Coverage) - Grade 5/10

The repository has low test coverage relative to its size. Specifically, the `src/shared` libraries, which are critical dependencies for multiple tools, lack comprehensive unit tests.

## Goals
- Increase test file count from ~31 to >50.
- Achieve 60% code coverage on `src/shared`.

## Action Items
1. [ ] Install `pytest-cov`.
2. [ ] Identify top 5 critical modules in `src/shared` without tests.
3. [ ] Write unit tests for these modules.
4. [ ] Add a coverage check to the CI pipeline (start with a low threshold and ratchet up).

## Reference
- See `docs/assessments/Assessment_C_Test_Coverage.md` for details.
