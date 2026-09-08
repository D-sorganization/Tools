# Deformed-Section Frame Inertia: Private Verification

Tools #5072 remains partial. Published section kinetic quadrature
`22cfc8df957bfea02faa1e31e5003ad2c239104c` now has a companion instantaneous
rotating-frame residual, moving-material derivative and gyroscopic operator.
The companion uses the actual supplied section shape; no straight-state
coefficient or empirical scalar stiffening factor is substituted.

## Source, Reuse and Coordinates

The derivation starts from the material inertial balance and interpolation in
[Sonneville, Cardona and Brüls (2014), author manuscript](https://orbi.uliege.be/bitstream/2268/159471/1/pa_SonnevilleCardonaBruls2014_GeometricallyExactBeamFEOnSE3.pdf),
equations 69–71 and 78–82, reviewed in `LOADED_INERTIA.md`. The rotating-frame
specialization below is derived here and independently tested; it is not a
claim that the source qualifies a golf shaft or measured frequency bandwidth.

Reuse: `RotatingFrameState` supplies physical origin acceleration a0 [m/s²],
angular velocity omega [rad/s] and physical angular acceleration alpha [rad/s²].
`SectionInertia` supplies reference-arclength-weighted `InertiaSample` records.
`tip_spatial_inertia` supplies the full COM-offset mass tensor D. Existing SE(3)
pose, logarithm, velocity-map and Frechet-derivative kernels remain canonical.
No new frame-motion convention, mass-property model or interpolation is added.

H maps section material axes into the declared observer frame. All six-vectors
are linear first; nodal perturbations are right-trivialized material twists.
At the evaluation instant choose a Galilean frame in which observer-origin
velocity is zero. With spatial generators W=[0;omega], A=[a0;alpha], define

```text
w = Ad(H^-1) W = [R^T(omega x p); R^T omega]
a = Ad(H^-1) A = [R^T(a0 + alpha x p); R^T alpha]
f = D a - ad(w)^T D w
g_nodes = sum Q^T f
```

This f is required physical inertia on the **left** of the balance, not its
negative centrifugal pseudo-force. Even at zero relative velocity, frame
motion generates nonzero g. Angular momentum, COM offset and section rotary
inertia all contribute. `frame.frame_id` identifies the observer axes of every
input pose; the sample body frame identifies its material-axis convention.

## Complete Material Derivative

For material nodal direction z, relative-log direction dd=Pz and point virtual
motion eta=Qz, the advected quantities satisfy dw=ad(w)eta and da=ad(a)eta.
Therefore

```text
df[z] = D da - ad(dw)^T D w - ad(w)^T D dw
Dg_body[z] = sum (DQ[dd]^T f + Q^T df[z])
```

Both the interpolation-map and physical inertial-wrench derivatives are kept.
`moving_jacobian` uses material wrench components at each perturbed pose. To
obtain a fixed exponential-chart derivative, subtract one half of the blockwise
ad(z_i)^T g_i correction. This is the same moving/fixed distinction used by the
existing clamped solver; adding tangents with different conventions is invalid.

At zero relative motion, the point coefficient of relative velocity u is

```text
G_point u = D ad(w)u - ad(u)^T D w - ad(w)^T D u
G_nodes = sum Q^T G_point Q
```

G is skew and has zero quadratic power by the derived physical balance. The
implementation does not impose skew symmetry or classify it as damping. This
zero-relative-velocity coefficient differs from the general velocity-dependent
bias in `SectionInertia`, whose power includes one half v^T Mdot v.

## Energy, Frame Changes and Limits

For alpha=0, the nominal frame residual is the gradient of
V=sum(m a0 dot COM - w^T D w/2). Its fixed-chart derivative is symmetric.
An Euler acceleration generally has nonzero curl; that nonsymmetry must be
retained. Frame rotation and translation require transforming the physical
origin acceleration too: shifting the origin by s adds alpha x s and
omega x (omega x s) before rotating its components.

These terms alone do not establish shaft stiffening. The complete balance must
combine them with internal elastic residual, its stress-dependent geometric
tangent, applied loads and boundary conditions, and solve for the loaded shape.
That root must satisfy explicit material domains and stability/refusal policies.
A frame with time-varying motion is not certified stable by a frozen snapshot.
Moving-boundary work, shear/rotary bandwidth, mesh/time/modal/FRF convergence,
contact, acoustic radiation and empirical validation remain open.

## Verification

TDD first failed on the missing module. Eleven frame tests now pass. Independent
Newton/Euler calculations form COM force and angular-momentum torque, projected
using general 4x4 matrix-log/exponential point-pose differences. They separately
check omega, alpha and origin acceleration, all 144 material Jacobian entries,
Coriolis force, zero gyroscopic power and agreement with the absolute section
mass/bias balance. Common observer rotation/translation, conservative potential
curvature, Euler curl, invalid types/poses and fresh output arrays also pass.
The initial boolean-input expectation was corrected to the existing validator's
TypeError; no production contract was weakened.

All 147 focused frame/inertia/equilibrium/chain/load/section/SE(3) tests pass
(39.86 s). All 554 golf/API tests pass with two optional build123d CAD skips
(199.58 s). Ruff 0.14.10 passes after its loop-variable diagnostic was corrected.
The API addition is one private module with empty exports. The actual pre-push mypy hook passes for the new module. All nine manual checks pass after inventory and handoff refresh. The golf
inventory now has 61 modules; the provisional name-based index adds one Rust
math-primitives test association, without a Rust implementation change. Normal
publication hooks remain required.
