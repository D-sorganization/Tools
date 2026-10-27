---
issue: 5830
summary: "Include elevation pressure drop in compressible flow calculations (TOOLS-12)"
branch: "fix/5830-pressure-drop-compressible-elevation"
dl_state: "in_review"
next_step: "Merge the PR."
---

In `sidekick.process_calculators.pressure_drop_calculator.engine.pressure_drop_calculation_engine`, `_apply_compressibility()` previously passed total incompressible dP (including static elevation head) to `calculate_compressible_flow_correction()` and assigned the horizontal isothermal output directly to `total_dp`, completely dropping static elevation head ($\rho g \Delta z$) and corrupting `dp_acceleration`. Decoupled horizontal pipe friction loss from elevation head, computed compressible flow on the horizontal component, added back elevation head to total dP, and computed acceleration loss as the difference between compressible and incompressible horizontal losses. Added failing test reproduction in `test_pressure_drop_engine.py`.
