# Spatial Contact Kinematics

Candidate IA-T4 #5073 now integrates numerical provider review #5133 at
0cd6dce22f5762ee0c6342000fb035eaa76579d5. Pre-integration artifacts below
retain their original 608e85b24 base identities. This private increment supplies geometric and
normal constitutive work ports, not a resolved impact trajectory or qualified
physical/acoustic model.

## Inventory and Integration

The canonical `swing_sim/impact/contact.py` owns unilateral Kelvin–Voigt
forces. Existing T2 `golf_club/_coupled_impact_solver.py` resolves lumped
force release, geometric clearance and separate viscous/cutoff losses.
`SpatialPointLoad` already supplies material force/moment and power maps;
the new port reuses those maps. It does not reuse their fixed-point tangent
as a derivative of a migrating contact point.

The existing `SpringDamperImpactModel` has a fixed normal, translational
integration and unchanged angular velocities. The gear-effect helper uses
an approximate recoil ramp and rolling cap. Neither is a resolved spatial
friction trajectory, and that helper must not be applied on top of one.
Ground bounce geometry uses a fixed plane. Educational frame conversion
uses angular-first twists, whereas the shaft uses linear-first body twists.

`ContactBodyState` retains proper pose, body twist and observer identity.
It uses the shaft's strict validators without inventing an acceleration
to satisfy the separate moving-grip acceleration contract. The new module
is private with no public facade exports; the eventual provider integration
must preserve import-order and existing API contracts.

## Geometry and Work

Let the sphere center be b, undeformed radius R, face reference point a and
face-attached unit normal n. The signed gap is

    g = n · (b − a) − R.

Positive g denotes clearance. Face rotation contributes n_dot · (b − a) to
its derivative. Project b onto the plane to obtain the common point

    c = b − [n · (b − a)] n.

The face's material velocity at c is v_h + omega_h × (c − x_h), not the
derivative of the migrating projection c. Evaluate the ball velocity at
the same point. Their difference u satisfies g_dot = n · u. A force f on
the ball and −f on the face, both applied at c, have zero total spatial
force/moment and combined power f · u. Normal-only power is F_n g_dot.

Using the projected common point makes the ball's tangential moment arm
compression-dependent. Applying forces at two distinct undeformed surface
points instead leaves a net tangential couple. The common-point convention
preserves angular momentum and power, but is still an approximation to a
deformed contact patch. Finite face boundaries, curvature, indentation
limits and local compliance require their own model and sensitivity checks.

## TDD and Current Evidence

The missing-module test failed at collection in 5.07 s. The initial
implementation then passed 18 Windows cases in 8.65 s: independent pose
matrix-exponential differences, force/moment/work closure, normal torque,
compression-dependent tangential lever arm, common rigid motion, inertial
frame/boost invariance, owned state, observer compatibility and domains.
The actual two-file mypy hook and NumPy-aware production check pass. Ruff found
a constant setattr in the immutability test; direct guarded assignment corrects it.
The full impact-directory regression passes 102 tests on Windows (8.02 s) and
102 under Linux coverage (3.21 s), without warnings or skips. Existing 228
swing_sim API module records are identical; the sole new module declares an
empty public export list. The selected API test passes. Exact staged-tree/JUnit
digests are in SPATIAL_CONTACT_RESULTS.json. Provider repair integration,
canonical inventory/handoff recording and publication remain.

## Analytical Contact Reference

Existing T2 already has an underdamped atan2 force-release oracle.
[Schwager and Pöschel](https://arxiv.org/pdf/cond-mat/0701278) analyze how
the repulsive force-cutoff criterion differs from a full damped half-cycle.
Their four-page manuscript is a mathematical granular-contact reference,
not golf-ball calibration. Ambiguous extracted signs/functions must not
be transcribed without checking the governing equation.

An independent dimensionless derivation for x'' + 2 zeta x' + x = 0,
x(0)=0, x'(0)=1 gives first decreasing force-zero time

    tau = 2 acos(zeta) / sqrt(1 − zeta²), zeta < 1;
    tau = 2,                              zeta = 1;
    tau = 2 acosh(zeta) / sqrt(zeta² − 1), zeta > 1.
    restitution = exp(−zeta tau).

A separate exploratory DOP853 root calculation agrees for nine zeta values
from 0.01 to 20 at predeclared 1e-9 relative/1e-11 absolute checks. This is
not production API or physical evidence; it can extend later contact tests.

## Normal and Tangential Work

The private normal-work increment reuses the canonical normal force and the
existing T2 loss decomposition. With compression d and compression rate v,
active Kelvin–Voigt contact has force k*d+c*v, stored energy k*d²/2, storage
rate k*d*v and viscous loss c*v². After tensile-force cutoff while d remains
positive, force and viscous loss are zero; the existing model records
cutoff loss -k*d*v. That channel represents the unilateral constitutive
cutoff, not extra measured material damping. A resolved spatial model must
report any force ceiling as an explicit domain/model decision. Silently
clipping a force while retaining the unmodified spring/dashpot energy ledger
would violate the power identity. Supplied damping and its provenance must
remain explicit; the legacy half-cycle restitution factory is not a force-cutoff
calibration oracle.

The private tangential port implements a candidate isotropic elastic/Coulomb
return map, retaining an elastic tangent vector z with stiffness k_t. Transport its old
value to the current contact frame by an orthogonal map Q, then form

    z_trial = Q*z_old + ds;
    z_new = projection of z_trial onto the disk |z| <= mu*F_n/k_t;
    dp = z_trial - z_new; tau = k_t*z_new.

Here ds is the material relative-slip increment in that same frame, tau is
the resisting effort and the force on the ball is -tau. This is an implemented private
discrete constitutive update, not a resolved spatial friction trajectory. For fixed
isotropic stiffness, its end-step work identity is

    tau · ds = Delta(k_t*|z|²/2) + tau · dp
               + k_t*|z_new-Q*z_old|²/2.

The radial projection makes the plastic term nonnegative. The final term is
algorithmic energy loss from the end-step update; it must be recorded separately
and converge away rather than be called physical damping or acoustic energy.
Separation/reset also requires explicit handling of any remaining stored energy.
Time refinement must resolve the changing normal-force bound, especially near
release. A vanishing cap alone does not qualify a coarse friction history.

Transport must preserve tangency, norm and observer objectivity, including
contact-normal rotation and spin about the normal. Face-attached transport
and mean-contact-spin transport are distinct constitutive choices under
relative ball/face spin, so their effect needs a declared convention and a
sensitivity control. The [objective contact-update literature](https://arxiv.org/abs/2002.10231)
is a method reference for this review, not golf-ball calibration. Independent
normal rotation, twirl, observer rotation, shrinking/zero bounds, sticking,
sliding, reversal, unloading and discrete work controls now pass. In monotone
slip, 100/200/400 steps recover the independent continuous elastic/plastic work
and halve the separately measured algorithmic loss on each refinement. A
sudden zero-force reset removes retained storage into that algorithmic channel;
it must not be relabeled as measured material or acoustic energy.

The normal port has 20 additional missing-module RED-to-GREEN controls.
Combined geometry, legacy impact and T2 audits pass all 148 tests on Windows
(40.57 s) and Linux coverage (57.80 s). The first Linux attempt timed out in
the existing deterministic T2 report. Converting six ODE float64 coordinates
once to Python floats removed NumPy scalar dispatch from scalar force/work
algebra; all equations, events, scientific tolerances and deadlines stay.
Both attempts and immutable source/JUnit hashes are recorded in
SPATIAL_CONTACT_RESULTS.json. The new tangential port starts with missing-module RED, then 18 cases pass;
10 additional adverse/reversal/changing-bound controls also pass. The full
impact package now passes 150 Windows tests in 6.86 s. NumPy-aware typing
passes both the new constitutive module and shared geometric normal validation.
Linux full-impact coverage also passes 150 tests in 4.68 s. All 228 existing
API records remain unchanged, with three private empty-export modules added.
Renewed inventory and provider integration remain.

## Remaining IA-T4 Scope

Couple ball and shaft states with contact forces recomputed at every stage;
retain full head inertia and face/hosel modes. Add objective tangential
history, stick/slip/separation, energy accounting and no double gear effect.
Verify centered/eccentric impacts, independent time/mesh/mode/event
convergence, matched-state unloaded/preloaded interventions and their energy,
force/torque histories and ringdown. Held-out force, spin and face-map data
remain necessary. This geometric step closes none of those requirements.

## Integrated Port Qualification

Provider 0cd6dce22 integration passes all 176 impact and legacy coupling
controls on Windows (51.01 s) and Linux coverage (62.71 s), without warnings
or skips. Exact source tree 218006534 and JUnit digests are retained in
SPATIAL_CONTACT_RESULTS.json. No coupled spatial trajectory or physical/
acoustic qualification is inferred. Further consumer CI repair is in progress
in the separate shaft/provider worktrees; this contact source is preserved.

## Coupled Response Design: Next TDD Boundary

The next implementation must recompute the common-point contact snapshot from
the current ball and shaft-node poses at every force evaluation. The existing
`MovingTrajectoryProblem.chain_at(time)` changes prescribed grip anchors only;
its fixed applied-load contract must not silently become a contact callback.
The head's full COM offset and tensor already enter `attach_nodal_body` through
one integrated endpoint inertia sample. They must not be counted again in a
separate head mass after attachment.

For a free rigid body with a fixed material origin, use linear-first material
twist `v`, spatial inertia `M`, and origin-resolved wrench `w`. The canonical
shaft convention gives `M vdot = w + ad(v).T M v`. Since `ad(v) v = 0`, its
kinetic-energy derivative is `v.T w`; transport is not a loss channel. Reuse
the existing spatial-inertia and Lie-bracket kernels. A ball record must name
its material convention and COM datum explicitly; a club-component role must
not be invented just to reuse a mass-property container.

Independent RED controls should include full-tensor Euler acceleration,
translation/rotation coupling from an eccentric COM, observer linear and
angular momentum derivatives, zero-wrench energy conservation, force-offset
torque, and malformed/singular/unresolved mass refusal. The pair's observer
forces and moments must cancel at the declared common contact point. Its
power remains force dotted with relative **material** velocity, including
normal gap rate and tangential slip; contact-point migration is excluded.

For the normal-only coupled response, ball kinetic plus shaft/grip energy
plus normal storage must balance external load work, anchor work and the
separate viscous/cutoff channels. Prove the axial synthetic three-body limit
independently before a finite-rotation trajectory. The tangential end-step
return map cannot be inserted as an ordinary RK derivative: its history,
frame transport, normal-cap changes and algorithmic work need a consistent
discrete coupling and refinement study. The free-body response boundary is now implemented and separately tested
as recorded below; the coupled normal response and discrete trajectory
remain open.

## Rigid-Body Response Qualification

The private full-tensor free-body response starts with missing-module RED.
Its 18 initial controls and 57 existing moving-shaft controls pass together
(75 tests). Six additional relative-inertia domain cases expose two failures
for tiny nonphysical tensors; scaling the existing realizability check by
the tensor magnitude makes all 24 body controls pass. No tensor is projected
or regularized. NumPy-aware mypy passes all four changed production files.
The canonical shaft mass solve is extracted unchanged and shared, and its
linear-first spatial inertia accepts read-only physical fields without
inventing a club-component role for the ball. Observer momentum derivatives,
full-tensor Euler acceleration, eccentric COM, mechanical power, material-axis
relabeling and singular/unresolved mass refusal have independent controls.
Exact staged tree 45ccf7726 is archived for expanded Linux coverage. This is
an instantaneous response; spatial coupling and trajectories remain open.

## Coupled Normal-Response Qualification

The normal-only shaft/ball composition now recomputes the contact point,
normal force and common-point reaction for every supplied stage state. It
retains the shaft's existing head inertia, loads and moving grips. It adds no
second head mass and no approximate gear-effect correction. Independent
three-mass controls cover compression, clearance and unilateral cutoff;
nonplanar offset contact matches the total-energy derivative with moving
grips and remains invariant under observer rotation/translation. Central
normal contact creates no spurious spin of a centered isotropic ball.
The incomplete-state RED is corrected by validating the node count before
indexing. All 12 composition and 24 body tests pass together; five production
files pass NumPy-aware mypy. API RED exposed a changed annotation, so the old
club entry-point signature is preserved over a private shared inertia kernel.
Every pre-existing golf_club and swing_sim API record is unchanged; new
modules have no public exports. The earlier body/shaft archive passes all
311 Linux coverage tests. Expanded composition qualification passes all 323 Windows tests (43.21 s) and 323 Linux coverage tests (60.28 s), with no failures or skips, at archived source tree ec863402f. Source/JUnit digests are in SPATIAL_CONTACT_RESULTS.json.

Full spatial trajectories, discrete tangential-history coupling, face/hosel
modes, independent time/mesh/mode/event convergence and measured impact and
acoustic qualification remain required. The coefficient values are synthetic.
