---
name: Code Review Finding
about: Document a finding from code review or assessment
title: 'fix: '
labels: code-quality
assignees: ''

---

**Severity**: <!-- LOW / MEDIUM / HIGH / CRITICAL -->
**Category**: <!-- Error Handling / Security / DRY / DbC / Testing / Documentation / Performance -->
**Component**: <!-- e.g. Pendulum Simulator, C3D Reader, URDF Generator -->
## File Paths
<!-- MUST BE CANONICAL PATHS. Use exact src/... or tests/... paths. Do not use aspirational or descriptive names (e.g., 'notes_tab.py'). Provide a bulleted list of files the implementation must touch. -->
- 

## Description
<!-- What was found during review -->

## Current Behavior
<!-- What the code does now -->

## Expected Behavior
<!-- What the code should do -->

## Proposed Fix
<!-- How to fix it -->

## Acceptance Criteria
- [ ] Failing test added first (TDD)
- [ ] Fix implemented following DbC principles
- [ ] No DRY violations introduced
- [ ] CI passes

## Estimated Effort
<!-- S (< 2 hours) / M (2-8 hours) / L (1-3 days) / XL (3+ days) -->
