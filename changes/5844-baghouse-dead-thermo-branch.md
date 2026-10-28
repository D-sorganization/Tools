---
issue: 5844
summary: "Delete dead unreachable HAS_THERMO branch and update docstrings in baghouse_calculator (TOOLS-25)"
branch: "fix/5844-baghouse-dead-thermo-branch"
dl_state: "in_review"
next_step: "Merge PR"
---

Delete dead module-level HAS_THERMO, FlowUnit, GasStream, and ThermodynamicCalculator placeholders and unreachable guarded thermo block in `BaghouseCalculator._calculate_outlet_thermal`. Update class docstring to accurately describe the standalone ideal-gas calculation contract (TOOLS-25, #5844).
