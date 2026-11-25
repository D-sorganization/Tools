---
issue: 6019
summary: "Tests-only stdio MCP acceptance slice for the flare calculator"
---

Acceptance coverage exercises real stdio MCP `list_calculators`,
`describe_calculator`, and `run_calculation` for calculator id `flare`, plus a
fail-closed invalid-input path that returns a structured `FAILED` envelope (not a
transport or serialize error).