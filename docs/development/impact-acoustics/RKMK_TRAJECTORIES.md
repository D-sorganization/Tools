# Rotating disturbances, local Lie RK4 and convergence

Tools #5072, 2026-09-09. This development note extends
[NONLINEAR_TRAJECTORIES.md](NONLINEAR_TRAJECTORIES.md) and
[JOINT_TRAJECTORY_CONVERGENCE.md](JOINT_TRAJECTORY_CONVERGENCE.md).
It is not an approved engineering-manual pathway, physical parameter fit,
impact-band qualification or acoustic prediction. Existing material, grip,
frame, strain and work-port definitions remain authoritative.

## Why another explicit method is useful

The existing Lie midpoint method has a documented 2N+1 acceleration-call budget.
New two-plane bending disturbances of a centrifugally extended, finite-grip
shaft show its expected second-order time convergence. However, at 256 steps
over 0.004 s, scaled errors of 0.00687005 and 0.02375532 on two/four elements
miss the declared 0.001 target. The first tests fail; their four-element runtime
is 55.98 s, close to the unchanged 60-second per-test deadline. Neither the
target, physical interval nor deadline is relaxed. The failures are retained
as midpoint counterexamples alongside the new method's acceptance tests.

## Local exponential coordinates with the correct trivialization

During one step write each nodal pose as H(t)=H0 Exp(q(t)), q(0)=0, using the
existing linear-first SE(3) coordinate q=(translation generator, rotation).
The physical equations are Hdot=H hat(V), Vdot=a(t,H,V). The existing right
Jacobian gives the exact local change of variables in exact arithmetic:

    V = Jr(q) qdot,       Jr(q) = phi1(-ad(q)),
    qdot = Jr(q)^-1 V,    Vdot = a(t,H0 Exp(q),V).

Apply classical RK4 to these local coordinates, the material twist and each
of the three work integrals. Its abscissae are 0,1/2,1/2,1 and its weights
1/6,1/3,1/3,1/6. Reconstruct H1=H0 Exp(q1), retain V1 and reset q=0 for the
next step. Matrix-exponential Jacobians and existing proper-pose validators are
reused. No quaternion normalization or energy correction is inserted.

The general use of an inverse exponential differential to apply conventional
Runge–Kutta methods in local Lie coordinates is described by
[Celledoni, Marthinsen and Owren, arXiv:1207.0069, Section 2.1](https://arxiv.org/pdf/1207.0069).
Our body-twist/right-update convention has the sign in Jr above; one must not
copy another group action's differential signs without translating conventions.
The implementation uses the existing full Jacobian, not the paper's optimized
commutator polynomial. An independent quaternion integration control verifies
fourth-order convergence, and intentionally omitting or reversing the Jacobian
correction increases its error. This qualification does not make explicit RK4
symplectic, exactly energy-conserving, adaptive or unconditionally stable.

## Shared contracts and accounting

The two entrypoints share one integration loop, pose reconstruction, owned
endpoint records, numerical refusal handling and work ledger. The original
midpoint step and its work weight are retained. A separate concrete control
type selects the fourth-order entrypoint. A second RED test catches accidental
acceptance of fourth-order controls by the midpoint entrypoint; this mismatch
now refuses instead of silently changing the method and call-count contract.

For RK4, the initial response is cached; each step evaluates three internal
stages plus its final accepted endpoint. Thus exactly 4N+1 acceleration/history
evaluations are needed. Repeated midpoint/end times refer to different states
and count separately. Full-grid and budget checks precede all history access.
Stage or endpoint domain failures return no partial trajectory. History must
remain deterministic and kinematically consistent; evaluating it cannot prove
those assumptions. The inherited numerical grid guards remain unchanged.

At every stage, the same nonlinear force law supplies applied power, positive
grip-on-anchor power and passive dissipation. RK weights integrate these three
quantities separately. The reported defect remains

    E(t)-E(0)-W_applied+W_anchor+D.

There is no velocity rescaling, load change or corrective work. Constitutive
domains are checked only at evaluated stages/endpoints, not continuously
enclosed between them. Smoothness and resolved step/mesh scales remain needed.

## Disturbed rotating time qualification

The synthetic shaft/grip/tip prescription is the existing uniform reference
family, with angular speed 3 rad/s about the observer x axis. Start
from its finite-root radial equilibrium. At material coordinate u=s/L, apply
the local generator

    translation = epsilon (u^2, -u^2/2, 0),
    rotation = epsilon/L (u, 2u, 0),

with epsilon=0.001 m, L=1 m. Initial material twists follow exact corotation of
the perturbed poses; initial relative observer velocity is zero. The prescribed
anchor retains its exact rotating history. Both bending planes are excited.
Position/rotation errors are scaled by epsilon and epsilon/L; velocity errors
use those scales times the declared 100 rad/s reference rate.

The quaternion/DOP853 reference shares the nonlinear mechanical force law. It
is independent integration, not an independent constitutive model or physical
measurement. Two tolerances (rtol 1e-10/1e-12, atol rtol/10) are compared under
the existing 2,000-call reference budget. Their discrepancy must be below 1%
of the accepted final method error. All tests retain their 60-second deadline.

On two/four elements, RK4 at 16/32/64 steps shows fourth-order state convergence.
At 64 steps the scaled errors are approximately 0.000119933 and 0.000978578,
both below 0.001; absolute energy defects are below 1e-6 J. Nonzero anchor work
is retained (about 5.98e-5/5.73e-5 J), rather than treating the driver as a fixed
zero-work support. The 0.004 s observation interval does not establish an
impact or acoustic frequency validity range. These are prescribed-model tests.

## Nonlinear-to-linear amplitude consistency

Transform the inertial result back to the rotating observer, then compute
q=Log(H_equilibrium^-1 H_observer). Subtract the instantaneous frame-induced
material twist and apply Jr(q)^-1 to obtain qdot. Use the existing work-conjugate
length/time scales and compare against the constant gripped model's augmented
matrix exponential. Its operating residual is retained, not forced to zero.

On two elements, amplitudes 0.001, 0.0005 and 0.00025 m give scaled discrepancies
approximately 3.61009e-6, 9.02523e-7 and 2.25631e-7: a quadratic absolute
remainder. Reference tightening stays below 1% of each discrepancy. This tests
agreement between the existing inertial nonlinear and rotating tangent
formulations within a resolved small-disturbance regime. It does not establish
arbitrary-amplitude validity or an omitted-nonlinearity error certificate.

## Nested-mesh release qualification

A distinct linear rotating study holds a synthetic tip force
(1e-5,-0.5e-5,0) N, then removes it at t=0. Initial displacement solves the
full rotating stiffness with the operating residual retained. The same physical
loading protocol applies to 4/8/16/32 elements; finite-system matrix exponentials
avoid fixed-step temporal truncation in this mesh comparison. Their ordinary
floating evaluation remains part of the numerical limitations.

All six position and velocity coordinates at the same five material locations
are compared. Initial-state, final-state and motion-increment differences are
recorded separately. The 32-element result is a **fine-mesh reference, not an
independent continuum oracle**. Relative final errors on 4/8/16 elements are
approximately 0.00834143, 0.00212460 and 0.000526066. Initial/final errors decrease
and meet the declared fine-mesh limit; no universal rate is claimed for the
motion-increment difference. Previous independent stationary Timoshenko and
radial-equilibrium continuum checks remain separate evidence.

## Evidence and remaining scope

Missing-module RED precedes the new method. Independent augmented-exponential
state/work checks, geometric wrong-correction controls, inherited strict types,
budget/history/strain/chart failures and original midpoint regressions accompany
the studies above. See RKMK_TRAJECTORY_RESULTS.json for exact source and JUnit
provenance after final validation. Ordinary floating calculations, shared-force
references and finite numerical tolerances retain explicit limitations.

Physically identified versioned coefficient/FRF data, flexible head/face contact,
calibrated radiation, exact-pin consumers and physical/blinded perceptual evidence
remain required. No result here identifies a player's grip impedance, explains
the sweetness of a real strike, or closes the full impact/acoustics epics.
