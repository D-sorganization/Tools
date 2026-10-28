---
issue: 5840
summary: "Correct Colebrook friction factor solver method from Newton-Raphson to fixed-point Picard iteration and document transitional regime (TOOLS-13)"
branch: "fix/5840-colebrook-fixed-point-docstring"
dl_state: "in_review"
next_step: "Merge the PR."
---

In `sidekick.process_calculators.pressure_drop_calculator.engine._friction_factors`, `friction_factor_colebrook` docstring and internal comments previously claimed the implicit Colebrook-White equation is solved using the "Newton-Raphson method". Corrected the documentation and comment to accurately state fixed-point (Picard) iteration ($f_{k+1} = \frac{0.25}{[\log_{10}(\frac{\varepsilon}{3.7 D} + \frac{2.51}{Re \sqrt{f_k}})]^2}$) seeded by the Swamee-Jain explicit approximation. Added flow regime guidance noting transitional regime uncertainty ($2300 \le Re < 4000$) and recommending `friction_factor_churchill` for transitional modeling. Added verification-first unit test in `test_pressure_drop_engine.py`.
