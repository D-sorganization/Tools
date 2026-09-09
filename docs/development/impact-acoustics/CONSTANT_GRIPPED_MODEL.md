# Explicit Constant Gripped Operating Models: Tools #5072

`constant_gripped_model` selects a local affine model under explicit operating
prescriptions. It reuses the existing full-node gripped balance and dynamics,
preserves the residual, and connects their coefficients to the conditional
response assessment. Calling it is a modeling choice, not evidence that a
measured swing has constant inputs. A frozen snapshot alone cannot supply that
history. The model continues to report physical/nonlinear stability as
`unqualified`.

## What is prescribed

For an observer-to-inertial rotation R(t), prescribe a constant observer-resolved
angular velocity omega0 and origin acceleration a0:

```text
Rdot(t) = R(t) [omega0]x
p_origin_ddot(t) = R(t) a0.
```

Here a0 is the physical inertial acceleration expressed in the observer, as in
the existing `RotatingFrameState` contract. It is not a coordinate derivative
of the observer's own origin. The constant omega0 prescription implies zero
angular acceleration, so any nonzero supplied angular acceleration is refused
exactly. Conversely, zero angular acceleration at one instant does not prove
constant angular velocity over an interval.

Constant observer acceleration can rotate in inertial coordinates when omega0
is nonzero. Reconstructing an inertial trajectory additionally requires initial
position, velocity and orientation; this adapter does not invent them. Initial
constant translational velocity does not enter these local force operators.

The adapter holds the existing grip anchor poses fixed in this observer and
holds the existing observer-resolved force/couple laws constant. Material point
offsets continue to follow their nodes, through the already derived load-work
kernel. Observer-constant force/couple components are not generally constant
inertial components or body-following components. Finite-pose grip coefficients
retain their existing coordinate-law meaning and source IDs.

An observer-fixed anchor may move in inertial space. Neither its inertial
velocity nor its power is generally zero. World-frame energy accounting still
requires the previously derived moving-boundary work terms. Constant input
prescriptions do not convert the scaled response norm into mechanical energy.

## Assembly, coordinates and preserved contracts

The factory first requires a gripped rotating chain with actual shaft inertia.
It rejects incompatible angular acceleration, then calls the existing
`balanced_gripped_dynamics` with the supplied strain and force/moment
tolerances. It checks every node, including supported nodes. It does not solve
for another configuration, clamp the root, relax balance, or zero a residual
that merely lies below tolerance.

The new shared `_shaft_gripped_coordinates` helper applies

```text
q = S y
Ms = S.T M S, Gs = S.T G S, Cs = S.T C S, Ks = S.T K S
rs = S.T r.
```

Translations use the declared length scale; rotations retain unit scale.
The helper preserves every row, nonsymmetric stiffness, and separate G and C.
It validates complete six-axis nodes and strictly real finite coefficients
before arithmetic, and refuses unrepresentable scaling including underflow.
The existing gripped spectrum now uses this same helper, so its matrix scaling
cannot drift from the affine adapter. Its frozen-spectrum status remains
`unqualified`; no new damping, regularization or symmetrization is introduced.

The returned record owns the scaled pencil/residual, reference poses, frame,
scales and ordered grip source IDs. Strings and mappings are not accepted as
sequences of source IDs. Its `assess_response` delegates to the existing affine
kernel with exactly these coordinates. Direct record construction validates
types/shapes, not assembly provenance; versioned provider/calibration/report
qualification remains a separate task.

See [AFFINE_RESPONSE.md](AFFINE_RESPONSE.md) for residual sign, physical-time
conversion, assumed operator/input errors and the conditional response bound;
[LOADED_DYNAMIC_OPERATORS.md](LOADED_DYNAMIC_OPERATORS.md) and
[ROTATING_TRANSPORT.md](ROTATING_TRANSPORT.md) for the reused mechanics; and
[GRIPPED_SPECTRUM.md](GRIPPED_SPECTRUM.md) for the existing spectral controls.

## TDD and independent evidence

The initial absent-module test failed collection. Minimal implementation passed
11 controls. Additional tests reproduced acceptance of a string and a mapping
as source-ID sequences (two RED failures); explicit sequence validation repairs
both. Sixteen operating-model cases now cover constant spin, supported loaded
configuration, nonzero observer acceleration, strict angular-acceleration
refusal, copied poses/frame/source IDs, residual preservation and scaling
underflow. A factory pose-argument annotation exposed by mypy was repaired with the
already validated owned-pose conversion.

The independent two-node axial rod oracle uses EA=1000, line mass 0.2, unit
length, tip mass 0.1 and a root law with stiffness 400, damping 2 and relative
inertance 0.03 in SI units:

```text
M_axial = (0.2/6) [[2,1],[1,2]] + diag(0.03,0.1)
K_axial = 1000 [[1,-1],[-1,1]] + diag(400,0)
C_axial = diag(2,0).
```

Tests check these coefficients after length scaling and check the independently
known residual of a small nonzero tip load. The input survives the nominal
balance check and reaches the affine response assessment with its sign intact.
These parameters and trajectories are synthetic numerical fixtures, not
identified club or hand properties.

The 16 new cases, existing gripped-spectrum tests and affine tests pass together:
57 Windows tests in 17.95 s. Scoped Ruff 0.14.10 and the actual mypy 1.13 push
configuration pass. All functions remain within 50 lines. The API snapshot
adds only two empty-export private modules, with prior public entries unchanged.
The combined Linux golf/signal/API regression passes 824 tests in 273.50 s,
with two optional CAD skips and three unavailable-plugin configuration warnings.
It uses the preserved Python 3.11.15, NumPy 2.3.5 and SciPy 1.15.3 runtime
with one BLAS/OMP/MKL thread. All nine final manual/inventory/handoff gates and repository Ruff 0.14.10
pass (3,792 formatted files). Normal commit/push publication completed at
36aae1d57816799e4fd4023b8d591bf908e0885b, with the remote SHA verified.

## Remaining scope

The constant local model does not bound departure from its reference strain
domain, identify a time-varying swing, or predict large-motion contact. Continue
with validated transient/modal/mesh/FRF bandwidth, moving/nonlinear dynamics,
flexible contact and measured acoustic/radiation/blinded studies. Comparing
different prescribed spin/load states must retain the changed equilibrium and
consistent inertia/transport/stiffness; it must not assume universal dynamic
stiffening or a scalar effective mass. The #5103 calculation-classifier repair
and manual scientific/publication gates remain prerequisites for combined
delivery. No approved manual chapter or physical sweetness claim is created.
