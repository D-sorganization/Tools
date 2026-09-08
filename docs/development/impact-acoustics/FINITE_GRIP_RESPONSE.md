# Finite-Pose Grip Constitutive Response

## Scope and Integration

The private `_grip_finite_response.py` evaluates an explicitly supplied ideal
coordinate impedance on the finite kinematics derived in
`MOVING_GRIP_KINEMATICS.md`. The supplied anchor pose defines zero displacement
and the coefficient axes. A hand-marker-to-anchor calibration, if used, must
be supplied independently; the source identifier is not proof of calibration.
No hand mass, grip-pressure law or acoustic response is inferred.

The existing `PassiveGripImpedance` and `GripPortState` retain their fixed,
small-rotation meaning. Both constitutive paths share `_grip_energy.py` for
Gram-factor effort, storage and loss algebra. The new finite path does not
pass finite rotation vectors to the old physical small-rotation API. All
existing public API entries are unchanged; only two empty private entries
are added. The full inventory is regenerated from tracked source.

## Declared Law, Units and Work

Let q=[d;phi], where d is actual Cartesian root-to-anchor separation in anchor
axes and phi is the principal relative rotation vector. All states remain
inside the existing principal-log chart margin; continuation across its branch
boundary is refused. Let M=Fm^T Fm, C=Fc^T Fc and K=Fk^T Fk be constant
coefficients in these coordinates. The implemented effort is

```math
g=M\ddot q+C\dot q+Kq,\qquad
E=\tfrac12\|F_m\dot q\|^2+\tfrac12\|F_kq\|^2,\qquad
D=\|F_c\dot q\|^2.
```

Coordinates use m and rad; their conjugate effort uses N and N m. Gram-factor
columns must have the corresponding mixed units: Fm qdot and Fk q have units
sqrt(J), and Fc qdot has units sqrt(W). Numerical factor validation does not
establish these physical units experimentally. Finite rank-deficient factors
are permitted; an active or nonfinite coefficient is never silently repaired.

This is a holonomic coordinate law: the Lagrangian uses the relative storage
above, and the Rayleigh loss is qdot^T C qdot/2. The chain rule gives physical
material reactions through the existing motion maps:

```math
w_r=-A_r^Tg,\qquad w_a=-A_a^Tg,\qquad
w_r^TV_r+w_a^TV_a=-\dot E-D.
```

The returned effort is coordinate-conjugate, not an unchanged physical wrench.
All physical reaction vectors are owned immutable tuples. Energy/loss uses
squared factor norms, preserving nonnegativity for singular coefficients.
Arithmetic overflow and nonfinite power closure are refused; numerical
closure residual is returned separately and is never labelled damping.

Individual port powers depend on the observer. In the synthetic translational
control, anchor speed is -3 m/s, root speed -2 m/s and damping is 4 N s/m.
Root power is +8 W, anchor power -12 W and physical loss 4 W. The sum is -4 W;
root power alone would incorrectly suggest an active damper. For an inertial
energy ledger, anchor input into the attached system is -wa^T Va using inertial
motion states. A moving observer requires its separate frame-work terms.

The ideal coordinate inertance is not a rigid hand's absolute kinetic energy.
It does not include a measured body mass, its transport inertia or neuromuscular
feedback. Constant local coefficients also do not establish a valid frequency
band or a time-varying grip-pressure constitutive law.

## TDD and Independent Controls

The initial test collection is RED because the finite-response module is absent
(7.99 s). The first four new controls plus all existing moving/local grip tests
pass (35 tests, 9.00 s). Six final new controls and the 31 prior grip tests pass
(37 tests, 10.11 s). No physical or perceptual measurements are involved.

A known finite-angle coordinate trajectory supplies independent q, qdot and
qddot for coupled-factor energy and power checks. A second oracle perturbs
physical poses with SciPy matrix exponentials, computes rotation vectors and
potential energy independently, and checks all six reaction components against
the negative centered energy gradient (step 2e-6, rtol/atol 2e-8).

The accelerating-observer control differentiates L(t)H(t) through direct matrix
products. It preserves relative storage, loss, material wrench and summed port
power while changing individual port power. Other controls check zero/free
coefficients, copied inputs, invalid coefficients/source identity and overflow.
All nine API tests pass (7.82 s); repository Ruff 0.14.10 passes (3,745 files),
actual three-module pre-push mypy passes, and all production functions remain
within 50 lines. Complete Linux golf/API regression passes 634 tests (232.32 s),
with two optional CAD skips and three unavailable-plugin configuration warnings.
The same one-thread BLAS settings and 60-second per-test limit are retained.
All nine final manual gates pass. Implementation is published at
`28c45eb15f73ab7cfc551efbe97996b21ed22ccb` through every normal commit/push
hook, including repository unit tests, type checks, Bandit, dependency audit
and fleet guardrails; the remote SHA is verified.

## Remaining Loaded-Shaft Coupling

This checkpoint evaluates a law; it does not solve a shaft trajectory or a
loaded equilibrium. At a relatively stationary operating point, the left-side
root contribution is Ar^T g. Its directional derivative must retain
(dAr)^T g + Ar^T K dq, including preload geometry. Adding K alone is not a
consistent finite-pose boundary tangent. The full chain and grip residual
must use the same material/fixed-chart convention and retain frame terms.

Loaded root balance/tangent, time evolution with anchor work, mass/stability
qualification and mesh/time/modal/FRF bandwidth checks remain open. Impact
contact, head/shaft ringdown, identified acoustic transfer/radiation and
physical/blinded sweetness validation remain separate requirements. A passive
relative coordinate law does not prove stability of a driven rotating swing.

Inventory classifier correction #5103 is published separately at `81b28da05`.
Its formerly failing Python 3.12 unit shard now passes; final public CI is still
running and private consumer repository lookup still fails before tests. T3
must integrate that classifier and regenerate the full inventory before final
combined delivery. Current conservative labels are not scientific approval.
