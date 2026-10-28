---
issue: 5852
summary: "refine gas viscosity calculation to avoid non-physical (MW/MW_air)^0.25 scaling (TOOLS-21)"
---

Refine `calculate_gas_viscosity` in `src/shared/python/sidekick/process_calculators/scrubber_calculator.py`
to evaluate Sutherland's law using physical pure-component parameters for syngas constituents
and physical molecular-weight interpolation, ensuring polyatomic CO2 dynamic viscosity is lower
than diatomic N2/CO at 300 K.
