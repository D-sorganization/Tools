# Impact-Interval Club Dynamics: Formulation and Validation Basis

Status: Reference implementation for Tools issue #4130. This document defines
the model boundary and the tests that must remain true as the solver expands.

## Purpose and Publication Boundary

The model resolves club and ball state throughout a short contact interval. It
is intended for comparative engineering questions about face motion, off-center
recoil, grip-boundary assumptions, contact force, and launch sensitivity. It is
not a validated equipment-certification model. Shaft bending modes, evolving
contact patches, nonlinear/viscoelastic ball constitutive laws, and explicit
stick-slip transitions remain tracked extensions; their absence must not be
hidden when interpreting results.

All vectors use one right-handed Cartesian frame and SI units. The Rate of
Closure adapter uses the application frame: x toward the target, y up, and z to
the player's right. The club body frame is user-defined through
`ClubRigidBody`; its face normal, contact point, attachment point, mass, and full
3x3 inertia tensor are explicit inputs.

## State and Newton-Euler Equations

The free-club state is

\[
\mathcal{x}=\{\mathbf{x}\_c,\mathbf{R},\mathbf{v}\_c,\boldsymbol{\omega}\_c,
\mathbf{x}\_b,\mathbf{v}\_b,\boldsymbol{\omega}\_b\},
\]

where c denotes the club CG, b the ball center, and R maps body-frame vectors to
the world frame. With contact force F acting on the ball,

\[
m_c\dot{\mathbf v}\_c=-\mathbf F,\qquad
\mathbf I_w\dot{\boldsymbol\omega}\_c+
\boldsymbol\omega_c\times(\mathbf I_w\boldsymbol\omega_c)
=\mathbf r_c\times(-\mathbf F)+\boldsymbol\tau_g,
\]

\[
m*b\dot{\mathbf v}\_b=\mathbf F,\qquad
I_b\dot{\boldsymbol\omega}\_b=\mathbf r_b\times\mathbf F,
\qquad \dot{\mathbf R}=[\boldsymbol\omega_c]*\times\mathbf R.
\]

The implementation advances R with the SO(3) exponential map, avoiding Euler
angle singularities and retaining orthonormality. The ball uses the existing
uniform-solid-sphere inertia approximation from `impact/constants.py`.

## Moving Contact Point and Force Law

For body-frame CG-to-contact vector r0 and face normal n0,

\[
\mathbf r_c=\mathbf R\mathbf r_0,\quad
\mathbf n=\mathbf R\mathbf n_0,\quad
\delta=R_b-(\mathbf x_b-\mathbf x_c-\mathbf r_c)\cdot\mathbf n.
\]

The contact-point approach rate includes club and ball surface velocities:

\[
\dot\delta=(\mathbf v_c+\boldsymbol\omega_c\times\mathbf r_c
-\mathbf v_b-\boldsymbol\omega_b\times(-R_b\mathbf n))\cdot\mathbf n.
\]

The canonical unilateral Kelvin-Voigt law is

\[
F*n=\max(0,\min(F*{max},k\delta+c\dot\delta))\quad\text{for }\delta>0,
\]

and zero otherwise. Both the legacy translating spring-damper model and the
interval solver call the same `impact.contact.KelvinVoigtContactLaw`.
Regularized Coulomb friction is bounded by `mu F_n`; this is a stable reference
approximation, not the advanced evolving stick-slip/contact-patch model.

For a requested restitution e and effective mass m\*, the linear-oscillator
calibration uses

\[
\zeta={-\ln e\over\sqrt{\pi^2+(\ln e)^2}},\qquad
c=2\zeta\sqrt{k m^\*}.
\]

This analytic relationship supplies a reproducible instantaneous-limit target.

## Attachment Boundary Idealizations

- **Free:** six unconstrained club degrees of freedom; total linear momentum is
  an audit invariant.
- **Pinned:** the attachment position is fixed and its reaction does no work.
  Rotation is integrated about the attachment using the parallel-axis inertia.
- **Torsional Grip:** the pinned constraint plus
  `tau = -k_theta theta - c_theta theta_dot` about the shaft axis. Stored
  torsional energy is included in the energy audit.

These are selectable hypotheses, not claims about a real golfer's hands. A
future flexible-shaft model must implement a new boundary/provider behind the
same facade rather than adding conditionals to callers.

## Timescale and Non-Dimensionalization

Let T be contact duration and `omega_n = sqrt(k/m*)` the normal contact
frequency. The contact compliance number is `Pi_c = omega_n T`. Let
`Omega_c` be a characteristic club angular rate; `Pi_r = Omega_c T` measures
free face rotation during contact. The loaded interval matters when either
`Pi_r` is not negligible for the requested angular accuracy or off-center
contact torque changes angular rate appreciably during T. These dimensionless
groups are reported conceptually rather than used as tuning constants.

## Trace and Audit Contract

Every solve returns aligned histories for time, club/ball position and velocity,
club orientation, angular velocities, attachment and contact positions, moving
normal, compression, normal/friction force, face angle, dynamic loft, and
shaft-axis twist. `channel(name)` and `at_time(t)` provide stable query seams.

The audit records initial/final kinetic energy, dashpot/friction loss,
unilateral-release loss, torsional stored energy, residual energy, integrated
normal/friction impulses, and total linear-momentum residual. A small residual
is a numerical quality indicator; it is not permission to infer experimental
validity.

## Binding Validation Program

1. A centered symmetric strike produces zero club twist and angular recoil.
2. A stiff, short central contact approaches the established COR impulse result.
3. A toe/high offset produces signed angular recoil from `r x F`.
4. The pinned attachment remains fixed to numerical tolerance.
5. A torsional grip reduces shaft-axis twist relative to the pinned case.
6. Free-body total linear momentum closes, and the energy ledger reconciles.
7. Every public input rejects non-finite, nonphysical, or frame-invalid values.
8. Python reference and any future Rust kernel must pass identical parity cases.

The current automated cases live under
`src/shared/python/swing_sim/impact_interval/tests/`. Comparison with measured
contact histories and the UpstreamDrift putting bands remains required before
claiming external predictive accuracy.

## Source Map

- Golf-ball mass/diameter and driver constants: in-repository provenance in
  `src/shared/python/swing_sim/impact/constants.py`.
- Friction rolling-cap background: Cross, “Grip-slip behavior of a bouncing
  ball,” _American Journal of Physics_ 70 (2002), as already documented in
  `impact/models.py`.
- Existing impact-model and literature-file inventory:
  `docs/physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md` in UpstreamDrift.
- 3-D impact papers retained by the project:
  `docs/references/papers/Development and Comparison of 3D Dynamics Models of
Golf Clubhead Ball Impacts.pdf` and `Three Dimensional Golf Clubhead Ball
Impact Models for Drivers and Irons.pdf` in UpstreamDrift.

No numerical parameter in the reference implementation is presented as fitted
to those papers unless the corresponding code/docstring explicitly says so.
