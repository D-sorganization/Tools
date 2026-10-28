---
issue: 5837
summary: "Evaluate glass conductivity strictly at glass temperature without uncited power density bias in sidekick glass_interface (GLASS-07)"
branch: "fix/5837-glass-conductivity-uncited-bias"
dl_state: "in_review"
next_step: "Merge PR"
---

Remove the uncited heating bias `_POWER_DENSITY_HEATING_COEFFICIENT` (0.0001 K per unit power density) and the term `temp_kelvin += power_density * _POWER_DENSITY_HEATING_COEFFICIENT` from `GlassPropertiesInterface._default_conductivity_model`. Evaluates glass conductivity strictly at the specified glass temperature, ensuring default Arrhenius conductivity is independent of power density (#5837).
