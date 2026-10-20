# Swing Objective Comparison — design contract

**Epic:** [#4766](https://github.com/D-sorganization/Tools/issues/4766)
**Package:** `src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/`
**Surface:** `gui/swing_objective_lab.py`, tile `swing_objective_lab`

## 1. Question

The simulator can already optimize a downswing for clubhead speed. Golf coaching
instead talks about _mechanisms_: hold the lag and let centrifugal force release
the club; sequence proximal-to-distal so the arms decelerate into the club; route
the body's power through the grip; pull hard on the club for as long as possible.

Each of those is a well-defined functional of the trajectory. This feature
optimizes the same golfer against each in turn, under one shared torque budget,
and cross-scores every resulting swing against every objective. The question it
answers is whether optimizing for a mechanism produces the same swing as
optimizing for the outcome.

## 2. What this feature does _not_ do

It does not re-derive any physics. The relative-coordinate equations of motion,
the Rust kernel, the `transfer_strategy` grip-force contract, and the existing
optimizer all pre-date this work and are consumed unchanged.

A research prototype (`Double-Pendulum-Optimization`) derived the same
relative-coordinate model independently. Its equations are identical to
`physics.py` — `h = -me*L1*L2*sin(phi)`, `c1 = h*(2*dtheta1*dphi + dphi**2)`,
`c2 = -h*dtheta1**2`. That repository remains the research and notebook home and
an independent cross-check; it is **not** vendored, and nothing here is a port of
it.

## 3. Coordinates, frames, units

SI throughout. World frame, hub at the origin, `theta1` measured from the
downward vertical, `phi` the wrist cock angle of the club relative to the arms.

| Symbol   | Meaning                                 | At the top | At impact |
| -------- | --------------------------------------- | ---------- | --------- |
| `theta1` | arm angle from downward vertical        | ~2.6 rad   | 0         |
| `phi`    | wrist cock angle, club relative to arms | ~1.75 rad  | 0         |
| `mu`     | coupling `(m2/2 + mClub) * L1 * L2`     | constant   | constant  |

The club is the canonical distributed club of `physics.mass_matrix` (Tools #5431): a
uniform shaft `m2` (COM at `L2/2`) plus a head point mass `mClub` at the tip of link 2.
Before #5431 the club was a single point mass at the tip, for which `mu = (m2 + mClub) * L1 * L2`;
that is the `m2 -> 0` limit and survives only as a regression baseline.

## 4. Velocity-term partition (#4767)

`physics.coriolis_vector` returns `C(q, q̇) q̇` combined. With `h = -mu*sin(phi)`:

```
centrifugal = [ h*dphi**2 , -h*dtheta1**2 ]
coriolis    = [ 2*h*dtheta1*dphi , 0 ]
```

Two properties make the mechanisms separable:

- The Coriolis term exists **only** in the hub row. There is no `dtheta1*dphi`
  term acting on the wrist.
- The wrist centrifugal drive depends on `dtheta1**2` and is **independent of**
  `dphi` — it does not care how fast the wrists are already uncocking.

**Binding contract:** the partition must reproduce `physics.coriolis_vector`
exactly. This is enforced as a runtime postcondition and as a randomized test, so
the decomposition cannot drift away from the kernel it describes.

## 5. The −2 identity, and why CENTRIFUGAL is an impulse

Centrifugal and Coriolis **power** are not independent:

```
P_coriolis_hub = -2 * P_centrifugal_wrist
```

identically, for every trajectory, because both reduce to
`mu*sin(phi)*dtheta1**2*dphi`. They are one energy flow read at its two ends.

Defining both objectives as work would therefore make them the same optimization
problem with a rescaled cost, returning identical swings. `CENTRIFUGAL` is
instead the **angular impulse** `∫ mu*sin(phi)*dtheta1**2 dt`. Dropping the
`dphi` factor changes what it rewards — sustaining lag at high arm speed rather
than uncocking quickly — and makes it a genuinely different problem. Both the
identity and the resulting independence are pinned by tests.

## 6. Objectives

All are **maximized**, over the interval from the top of the backswing to impact.

| Key                 | Quantity                                              | Units |
| ------------------- | ----------------------------------------------------- | ----- |
| `clubhead_speed`    | tip speed at impact                                   | m/s   |
| `centrifugal`       | `∫ mu*sin(phi)*dtheta1**2 dt` release angular impulse | N·m·s |
| `coriolis`          | energy Coriolis coupling drains from the arms         | J     |
| `energy_transfer`   | work delivered to the club by grip force              | J     |
| `impulse_transfer`  | `∫ ‖F_grip‖ dt`                                       | N·s   |
| `hand_path_impulse` | `∫ F_grip · v_hand / ‖v_hand‖ dt`                     | N·s   |

`energy_transfer` and `impulse_transfer` are computed from the grip force, which
is pinned against `physics.net_joint_forces`. `hand_path_impulse` keeps only the
signed force component along hand travel. It is distinct from hand-path work and
from MacKenzie-style average force over path length.

## 7. Transcription and conditioning

Direct collocation (Hermite-Simpson by default), states and torques as decision
variables at every node, dynamics as equality constraints between nodes.

Held identical across every objective: golfer, top-of-backswing posture, torque
limits, torque slew-rate limits, duration, wrist range, impact condition, and the
effort regularizer weight.

Three settings are **load-bearing**, each with a regression test:

1. **Non-dimensional decision vector.** Radians (order 1), rates (order 30) and
   torques (order 200) in one SLSQP trust region leaves defects near `1e-1`.
   Scaled, they reach `1e-13`.
2. **Tight `ftol`.** At the SciPy default the solver reports success as soon as
   it finds a feasible trajectory and returns the initial guess unchanged.
3. **Torque slew-rate limits.** Without them the optimum reverses full hub torque
   between adjacent nodes to stop the arms dead at impact — optimal on paper,
   impossible for a golfer.

Feasibility is always reported from the measured dynamics defect, never from the
solver's own success flag.

## 8. Two failure modes the feature must report, not hide

### 8.1 An unreachable downswing

`DownswingConfig` rejects a duration below

```
t_min = sqrt(2 * arm_sweep / (tau_hub_max / M11_at_top))
```

with the computed bound in the message. This turns an opaque "positive
directional derivative for linesearch" into a statement about the golfer.

**The bound is necessary, not sufficient.** It ignores the slew ramp, gravity,
and the work of releasing the wrists, all of which push the true minimum higher.
For the lumped 0.5 kg club of the original preset (250 N·m, 180 N·m in the tests) the
analytic bound was 0.286 s while the problem did not become solvable until roughly
0.34 s. Re-run under #5523 for the shipped distributed preset the analytic bound is
0.227 s and the problem solves at 0.28 s; the practical minimum for the distributed
club was not re-measured.

### 8.2 A degenerate comparison

Close to that practical minimum the constraints pin the trajectory. The feasible
set collapses, **every objective returns the identical swing**, and the
cross-evaluation matrix fills with 100% entries.

That table reads as unanimous agreement between the mechanisms. It is an artifact
of the configuration. `SwingComparison.is_degenerate` detects it from the
pairwise RMS torque distance between swings, the flag travels on the wire, and
the GUI states it in plain language. **Callers must check it before reporting
agreement as a finding.**

The shipped preset therefore carries deliberate slack: originally 0.36 s at 250 N·m
rather than 0.34 s at 180 N·m for the lumped club, and 0.28 s at 250 N·m (analytic
minimum 0.227 s) for the distributed club. Re-run under #5523, the shipped preset is not
degenerate (maximum RMS swing distance 0.229 of the torque budget against the 1e-3
threshold), and the lumped 0.5 kg club at 0.36 s is not either (0.200).

## 9. Report wire

`swing-objective-comparison`, schema `1.0.0`, canonical JSON, fail-closed on a
missing or foreign `schema_version` and on a matrix that is not square against
the key list. It carries the tables, diagnostics, saturation, swing distance and
the degeneracy flag — but not the trajectories, so it stays embeddable.

## 10. Scientific boundary

This is a planar two-link model with a point-mass arm and a distributed club (uniform
shaft plus head mass, #5431).
It has no separate torso segment, no shaft flex, no ground reaction, no
three-dimensional plane change, and constant rather than posture- and
velocity-dependent torque limits. Results are statements about the mechanics of a
two-link kinetic chain under a torque budget and must not be presented as
anatomical attribution or coaching authority. The proximal link's angular rate is
not an anatomical shoulder or thorax velocity.

Whether the mechanism objectives agree with clubhead speed is
**configuration-dependent**, not a settled result. The surface exposes duration
and torque precisely so that boundary can be explored.

## 11. Results on the distributed club (re-run under #5523)

Reproduce with `python scripts/rerun_distributed_club.py comparison` from
`src/pendulum_simulator` (`PYTHONPATH=src`, headless variables as in
[`SWING_ACTUATION_AND_REALISM.md`](SWING_ACTUATION_AND_REALISM.md) section 7), on Tools
`main` at `4f941378c`. Inputs: the shipped preset (shaft 0.1078 kg, head 0.2022 kg,
0.28 s, 21 nodes, 250 / 20 N·m) and all six objectives. Every swing converged with
defects below 1e-11; the comparison is not degenerate (maximum swing distance 0.229).

Cross-evaluation matrix (row = swing optimized for that objective, column = value of
that objective on the swing, percent of the column best):

| swing optimized for (row) vs scored on (col) | clubhead | centrifugal | coriolis | energy | impulse | hand path |
| -------------------------------------------- | -------- | ----------- | -------- | ------ | ------- | --------- |
| clubhead_speed                               | 100.00   | 98.53       | 99.13    | 99.97  | 99.93   | 99.75     |
| centrifugal                                  | 97.25    | 100.00      | 91.88    | 92.66  | 95.78   | 97.80     |
| coriolis                                     | 99.73    | 99.38       | 100.00   | 98.94  | 99.28   | 99.99     |
| energy_transfer                              | 99.99    | 98.26       | 98.71    | 100.00 | 100.00  | 99.62     |
| impulse_transfer                             | 99.99    | 98.26       | 98.71    | 100.00 | 100.00  | 99.62     |
| hand_path_impulse                            | 99.79    | 99.31       | 99.99    | 99.15  | 99.39   | 100.00    |

Old versus new, same six objectives on the lumped 0.5 kg club at 0.36 s (same script,
`comparison_legacy_lumped_0.36s`): clubhead speed, centrifugal, energy, impulse and hand
path return the same swing (those five rows are identical, all 100.00 against each
other) and **Coriolis is the lone dissenter** (its row scores 97.43 / 93.51 / 100 / 92.74 /
96.73 / 96.07). On the distributed club the structure changes:

- **The dissenter moves from Coriolis to centrifugal release impulse.** Centrifugal's swing
  scores 97.25% on clubhead speed (51.4 to 50.0 m/s), 91.9% on Coriolis transfer and
  92.7% on energy transfer; every other swing reaches 98.3 to 99.4% of the centrifugal
  value. Coriolis is within 0.3% of the clubhead-speed optimum.
- **Energy transfer and impulse transfer produce the same swing** (rows identical to
  1e-12), so they are one mechanism on this model, not two.
- Separation between mechanisms is small (worst off-diagonal 91.9%) and sits on a
  configuration with 8 m/s hands and a 90% release point, so this stays an
  **exploratory**, configuration-dependent observation; it is not a statement about
  what golfers optimize.

Raw values on the swing optimized for clubhead speed: 51.40 m/s clubhead, centrifugal
impulse 5.214 N·m·s, Coriolis 86.5 J, energy transfer 226.8 J, impulse transfer
36.58 N·s, hand-path impulse 25.46 N·s. Force-based objectives (energy, impulse, hand
path) use the grip force from `net_joint_forces`, so these values also pick up the
shoulder-reaction fix of #5522.
