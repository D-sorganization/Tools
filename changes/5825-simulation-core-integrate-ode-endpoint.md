---
issue: 5825
summary: "Ensure integrate_ode includes t_end in uniform output grid (TOOLS-27)"
branch: "fix/5825-simulation-core-integrate-ode-endpoint"
dl_state: "in_review"
next_step: "Merge the PR."
---

In `double_pendulum_golf.simulation_core.integrate_ode`, the output evaluation grid `t_eval` previously constructed via `np.arange(0.0, t_end, dt)` omitted the endpoint `t_end`. The grid construction is updated to guarantee `t_end` is included, satisfying the Design by Contract postcondition `np.isclose(t[-1], t_end)` and ensuring callers receive states evaluated through the requested end time. Added unit tests in `src/pendulum_simulator/tests/` and top-level bridge test in `tests/pendulum_simulator/`.
