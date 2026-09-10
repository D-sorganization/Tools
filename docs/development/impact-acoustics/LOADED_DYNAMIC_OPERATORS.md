# Loaded Dynamic Operators: Private Verification

Tools #5072 remains partial. The loaded-root checkpoint is published at
`f47f64acf`. `linearized_chain_dynamics` now composes its full material residual
and tangent with the existing consistent section mass and deformed-frame
gyroscopic operators. It does not substitute stationary straight-shaft
coefficients on a deformed state. The new module has no public exports.

## Equation and Coordinate Contract

At the supplied relatively resting configuration, use material increments
H_i(delta)=H_i Exp(delta_i), with linear-first six-axis nodal ordering. The
instantaneous affine balance is

```text
r_0 + M_0 delta_acceleration + G_0 delta_velocity + K_0 delta_pose = 0.
M_0 = assembled sum of Q.T D Q at the current section geometry.
G_0 = assembled deformed-frame relative-velocity derivative.
K_0 = derivative of the complete moving material residual.
```

The kernels and primary-source basis are in `LOADED_INERTIA.md`,
`DEFORMED_FRAME_INERTIA.md` and `ROTATING_LOADED_ROOTS.md`. The assembly reuses
those kernels and the shared material/chart connection. In particular, K is
not silently replaced by a symmetric fixed-chart Hessian. At an exact clamped
root, the connection vanishes on free rows/columns because their residuals
vanish. The full root reaction and off-root residual remain in the record.

Relative velocity is zero at the expansion point, so the quadratic relative
inertia bias has no first-order contribution. Frame transport remains through
G, which is skew and does no quadratic work. No damping matrix is inferred
from it. M, G, K and r are independently allocated finite arrays. A singular
mass quadrature remains singular; no rank repair, eigenvalue clipping, boundary
elimination or stability label is supplied.

Frame acceleration may vary in an actual swing. This frozen snapshot alone
does not show that a trajectory follows a sequence of equilibria, nor certify
time-varying or nonlinear stability. Applied-load conventions, material-domain
qualification and boundary conditions remain explicit obligations.

## Independent Physical and Continuum Controls

For a curved, three-node fixture, general matrix logarithms/exponentials and
centered point-motion derivatives give COM translation and material spin.
Their physical kinetic energy agrees with v.T M v/2 to relative 2e-8. An
independent Newton/Euler oracle gives the assembled gyroscopic wrench to
absolute 5e-8; skew symmetry and zero power hold without projection. A
directional finite difference of material balance checks K away from a root.
Common observer rotation preserves every material operator. Zero angular
velocity removes G while preserving M. Copy isolation, invalid inputs and an
intentionally singular endpoint quadrature are checked.

The radial rod control retains L=1 m, EA=1000 N, reference density mu=0.2 kg/m
and Omega=20 rad/s from the loaded-root control. Its exact continuum extension
is r(X)=sin(kX)/(k cos(kL)), k=Omega sqrt(mu/EA). For a two-element mesh h=L/2,
the axial submatrices are

```text
M_axial = mu*h/6 * [[2,1,0], [1,4,1], [0,1,2]]
K_elastic = EA/h * [[1,-1,0], [-1,2,-1], [0,-1,1]]
K_axial = K_elastic - Omega² M_axial; G_axial = 0.
```

The assembled matrices at the extended root match these independent controls.
For an axially guided rod (all nonaxial perturbations constrained),
mu*u_tt=EA*u_XX+mu*Omega²*u, with u(0)=0 and u_X(L)=0. Therefore

```text
frequency_n_squared = (EA/mu)*((n+1/2)*pi/L)² - Omega², n=0,1,...
```

The fundamental squared frequency converges at second order on 2, 4 and 8
elements: adjacent error ratios lie between 3.8 and 4.2, and the final relative
error is below 0.004. This is a constrained axial frequency, not a free-shaft
mode: transverse Coriolis coupling has been constrained in this control.
The rod stretches under spin while this axial frequency decreases. Neither
centrifugal extension nor tension alone establishes the sign of every mode's
frequency shift. These synthetic controls are not golf measurements.

## Verification and Delivery Boundary

The seven new tests first fail because the module is absent, then all pass on
Windows (28.52 s). The nine API tests pass and the only baseline change is a
five-line private empty-export entry. Existing signatures are unchanged.
Ruff 0.14.10 passes over all 3,733 Python files; changed-module mypy passes.

The broader Windows run and two isolated retries hit the existing 60-second
limit in the unchanged repeated impact-report sweep, including with BLAS thread
counts limited to one. The earlier checkpoint completed that test in 19.27 s;
the present slowdown's cause is unresolved. No test limit, physical parameter,
solver tolerance or numerical assertion was relaxed to obtain a pass.

A separate Linux CPython 3.11.15 environment with NumPy 2.3.5, SciPy 1.15.3,
pytest 8.4.2 and pytest-timeout 2.4.0 passes all 567 golf/API tests with two
optional CAD skips (296.00 s). The impact sweep takes 25.00 s with the same
60-second limit. Three missing optional asyncio/Qt plugin configuration
warnings are recorded. The authority runtime and shared runners are unchanged.

Complete stability/refusal, moving boundary work, loaded passive grip/head
integration and full modal/mesh/time/FRF qualification remain open. T4 contact,
qualified acoustic radiation and physical/blinded validation are separate
remaining program gates. Normal inventory, manual and publication checks apply.

All nine manual governance checks pass before and after this change. Inventory
updates are limited to the new module and its derived test associations; only
the two task-owned handoff hashes change (97 and 150 lines). Existing manual
approval fields are unchanged. Normal commit and push gates remain required.
