# Nonlinear Shaft Acceleration with Moving Grip Ports

## Scope and existing capabilities

Tools #5072 now composes the existing finite-pose section, inertia and grip
kernels into an instantaneous nonlinear acceleration solve. Every shaft node
is retained. The model includes distributed section quadratures, existing full
head/COM inertia, prescribed point forces/couples and zero or more moving grip
ports. It returns body-twist derivatives and a work/energy ledger. This is a
development derivation, not a registered or approved textbook pathway.

The existing stationary `GripAttachment`, equilibrium and
`ConstantGrippedModel` contracts remain stationary/constant. A separate
`MovingGripAttachment` accepts a prescribed anchor pose, body twist and body
twist derivative. `InertialMovingChain` explicitly selects an inertial observer;
nonzero observer angular velocity, angular acceleration or origin acceleration
is refused, including the smallest nonzero representable value. A zero frame
snapshot alone does not prove a history; construction selects the prescription.

This formulation evaluates the current state, not a trajectory. It does not
integrate anchor histories, qualify time/mesh convergence, infer a player's
impedance or declare a physical/acoustic bandwidth. Those remain required.

## Coordinate convention and literature context

Each material pose H maps into the common inertial observer. Linear-first
body twist V=(v,omega) satisfies Hdot=H hat(V). The solved a=Vdot contains
body-twist derivatives. Physical pose-origin acceleration expressed in material
axes is a_linear + omega cross v; it is not just a_linear. Physical angular
acceleration in those axes is a_angular. Existing section-inertia transport
already accounts for this distinction.

Simo and Vu-Quoc's [1986 dynamic finite-strain rod report, UCB/ERL M86/11](https://digicoll.lib.berkeley.edu/record/139314/files/ERL-86-11.pdf)
describes an inertial-frame formulation for large overall rod motion and treats
rotations as group-valued configurations. Its introduction and Section 2 inform
the frame/geometry choices here; its covariant Newmark algorithm and reported
spin-up benchmarks are not implemented or reproduced by this snapshot solver.

[Featherstone's spatial-vector seminar](https://royfeatherstone.org/teaching/seminar.pdf),
especially slides 4, 10-11 and 14, distinguishes spatial acceleration from point
acceleration and pairs motion with force through work. His ordering is angular
first; this code uses linear first. Existing tested kernels perform the
corresponding convention consistently; formulas are not copied between orderings.

## Moving-port mass and forcing

The existing grip coordinates q are actual Cartesian separation in anchor axes
followed by the principal relative rotation vector. They are not the SE(3)
translation generator. The existing kinematics supplies maps Ar and Aa such that

    qdot = Ar Vr + Aa Va
    qddot = Ar ar + b

Here b is evaluated with the actual root/anchor twists, prescribed anchor twist
derivative and zero root twist derivative. It includes the moving-map terms and
the prescribed-anchor acceleration; none is discarded or called damping.

For constant coordinate Gram-factor coefficients B, C and K, the ideal grip law
is g=B qddot+C qdot+K q. Its physical root reaction is -Ar.T g. Therefore the
unknown root acceleration contributes an additional mass Ar.T B Ar, while the
known left-side wrench is Ar.T (B b+C qdot+K q). The implementation builds the
mass using (F_B Ar).T(F_B Ar) and obtains the known physical reaction from the
existing `finite_grip_response`, preserving its finite-pose law and both ports.
Repeated ports add independently; this does not identify physiological coupling
between a player's hands.

Section quadratures supply full assembled M_s, material inertia bias b_s and
mass rate Mdot_s. The existing elastic chain supplies r_e=internal-external
material work, retaining force offsets and free couples. With nodal scatter
understood, the complete solve is

    (M_s + sum Ar.T B Ar) a
        = -(r_e + b_s + sum Ar.T (B b+C qdot+K q)).

There is no root clamp or implicit mass lump. A singular or numerically
unresolved positive-mass condition is refused without diagonal regularization.
The residual retains actual current loading; it is not replaced by a balanced
or unloaded state. Centrifugal/Coriolis effects of actual body motion belong to
the inertial dynamics already present. Extra rotating-observer inertial loads
must not be added to this inertial model.

## Energy and driver work

Total stored energy is shaft kinetic energy plus section elastic energy plus
the relative-coordinate inertance and elastic energy of every grip. It excludes
the existing force-only potential, because a general prescribed couple does
not share that scalar potential. Applied power uses `SpatialPointLoad.power`.

For each grip the reused work ports satisfy

    P_root + P_anchor + E_grip_dot + P_dissipation = 0.

Both reactions act from the grip onto its attached bodies. Thus positive
P_anchor is energy delivered out to the prescribed driver. The combined ledger
is

    E_total_dot = P_applied - sum P_anchor - sum P_dissipation.

Shaft kinetic-energy rate is evaluated from V.T M_s a + V.T Mdot_s V/2, separately
from the bias term used in the acceleration equation. Elastic-energy rate is
V.T r_e + P_applied. The resulting numerical power residual is diagnostic; it
is not a fitted or physical loss channel. An inertial observer is essential to
this ledger. A moving observer requires additional transport/work accounting.

## Numerical contracts and TDD evidence

The solve uses work-conjugate translation scaling S and solves
(S.T M S) a_scaled = -S.T r, then returns a=S a_scaled. Positive mass, symmetry
and reciprocal-condition checks reuse the existing spectrum mass validator.
The time member of `SpectrumScales` is not used for instantaneous acceleration.
Material strain limits, proper poses, observer IDs, topology and finite real
inputs are checked. No physical history or stability is inferred from success.

The initial missing-module test failed before production implementation. The
first 15 controls passed, followed by independent observer/scaling and
multisection linear/angular-momentum controls. A deliberately 10% inaccurate
solve under a 1e-180 N synthetic load exposed squared-norm underflow: the old
new-solver residual mistakenly appeared zero. That test failed before the fix.
The solver now uses a shared hypot-based finite norm; the frequency-interval
norm delegates to the same helper without changing its numerical definition.
This extreme value tests the arithmetic contract, not measured force resolution.

All 91 combined Windows moving-chain and frequency controls pass in 7.49 s.
They include independent two-mass moving-anchor equations, finite-difference
total-energy rates with nonplanar moving ports, repeated grips, full COM/spin
momentum rates across two sections, observer invariance, scaling independence,
strict domains, singular mass and inaccurate-solve refusal. Actual mypy passes
all nine changed Python files. Only two private empty-export module entries are
added to the API baseline; all existing parsed public API data is unchanged.
Zero-spin and loaded 3 rad/s corotation also agree between rotating and inertial
assemblies. The exact-source Linux run passes 988 golf/signal/API tests in
87.28 s, with two optional CAD skips and three absent-plugin warnings. Repository
Ruff passes (3,819 formatted files), as do all nine final governance gates after
canonical inventory regeneration. Normal publication remains; trajectory
integration remains unqualified.

## Remaining T3 and downstream work

Implement and qualify actual moving-anchor trajectories with proper-pose
updates, bounded work, original material domains and joint time/mesh convergence.
Complete rotating/prestressed controls and full/reduced continuous-band port
error, including absolute error near antiresonances. Identify/version physical
parameters before equipment claims. Flexible impact, contact/event/energy
convergence, calibrated acoustic radiation, exact-pin consumer studies,
physical/blinded experiments and final AffineDrift synthesis remain open.
