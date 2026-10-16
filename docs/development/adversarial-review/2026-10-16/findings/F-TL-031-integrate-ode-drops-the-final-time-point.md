# F-TL-031

**integrate_ode drops the final time point**

| Severity | Category    | Confidence | Effort | Status   |
| -------- | ----------- | ---------- | ------ | -------- |
| ⬜ Low   | correctness | high       | S      | proposed |

**Locations:** `src/pendulum_simulator/src/double_pendulum_golf/simulation_core.py:80`

## Finding

- Severity: Low - Category: correctness
- Confidence: high
- Location: src/pendulum_simulator/src/double_pendulum_golf/simulation_core.py:80
- Evidence: `t_eval = np.arange(0.0, t_end, dt)`.
- Failure scenario: `t_end=2.0, dt=0.005` → last sample 1.995 s; anything reading "state at t_end" gets the state one step early. (RK45 at rtol 1e-8 is fine for the chaotic horizon; not symplectic but adequate for sub-10 s runs.)
- Proposed change: `t_eval = np.linspace(0.0, t_end, int(round(t_end / dt)) + 1)`.
- Verification: `assert t[-1] == pytest.approx(t_end)`.
- Effort: S

## Draft Pull Request

**Title:** `fix(pendulum): include t_end in integrate_ode output`

### Why This Matters (The Pitch)

integrate_ode drops its final time point because arange excludes the endpoint, so a request for t_end of 2.0 with dt 0.005 ends at 1.995 s and anything reading the state at t_end gets it one step early. One line with linspace fixes it, and the test is a single approx assertion on t[-1].

### Proposed PR Body

## Problem

`src/pendulum_simulator/src/double_pendulum_golf/simulation_core.py:80` uses `t_eval = np.arange(0.0, t_end, dt)`, so `t_end=2.0, dt=0.005` ends at 1.995 s.

## Change

- [ ] Use `t_eval = np.linspace(0.0, t_end, int(round(t_end / dt)) + 1)`

## Verification

`assert t[-1] == pytest.approx(t_end)`.

## Risk and Rollback

Output arrays gain one sample; callers indexing by length should be checked. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-031

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
