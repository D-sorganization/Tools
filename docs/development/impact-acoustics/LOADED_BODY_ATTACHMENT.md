# Rigid Body Attachment to a Loaded Shaft

Tools #5072 remains partial. The private `attach_nodal_body` composes the
existing `RotatingSectionChain`, `SectionInertia` and `InertiaSample` contracts.
It adds one integrated physical body at a node, without new inertia equations,
an extra quadrature-length factor, a topology change or inferred damping.
The input chain remains unchanged. Existing public interfaces are unchanged.

## Coordinates and Body Mechanics

The body's COM offset c and full COM inertia Ic must already be expressed in
the selected section's material frame, relative to the selected nodal origin.
A mismatched frame identifier is refused. A shared interior node uses its
preceding section's right endpoint exactly once; the root uses the first
section's left endpoint. At either endpoint the existing interpolation maps
the sample's twist to that single nodal twist. The caller must exclude this
same body from the distributed mass to avoid physical double counting.

For nodal linear/angular material velocity (v,w), the added kinetic energy is
`T = [m |v + w cross c|^2 + w.T Ic w]/2`. Thus neither an offset COM nor
products of inertia may be replaced by a scalar tip mass in the general model.
At a frozen observer-frame pose (R,p), let d=R c, r=p+d and I=R Ic R.T.
The inertial force and moment about the nodal origin are

```text
F = m [a0 + alpha cross r + Omega cross (Omega cross r)]
N = d cross F + I alpha + Omega cross (I Omega).
```

The existing kernel transfers `(R.T F, R.T N)` into nodal material coordinates.
For a material perturbation velocity mapped to observer angular velocity eta
and COM velocity u, its Coriolis force is `2 m Omega cross u`. Its moment is
`d cross F + I(Omega cross eta) + eta cross (I Omega) + Omega cross (I eta)`.
These independent Newton/Euler expressions verify the existing operators,
including non-diagonal inertia, offset geometry, moving stiffness derivatives
and zero quadratic gyroscopic work. They are derived controls, not measured
golf parameters or evidence of acoustic radiation.

## Independent Continuum Control

A synthetic uniform radial rod has L=1 m, EA=1000 N, reference mass per length
mu=0.2 kg/m, tip mass mt=0.1 kg and constant spin Omega=10 rad/s about a
principal head axis. Its straight equilibrium obeys

```text
EA r'' + mu Omega^2 r = 0, r(0)=0,
EA [r'(L)-1] = mt Omega^2 r(L).
k = Omega sqrt(mu/EA)
r(X) = sin(kX)/[k cos(kL) - mt Omega^2 sin(kL)/EA].
support_on_rod = -EA [r'(0)-1].
```

For guided axial perturbations, the tip condition is
`EA phi'(L) = mt (frequency^2 + Omega^2) phi(L)`.
The first positive root satisfies `lambda tan(lambda)=mu L/mt`, giving
`frequency^2=EA lambda^2/(mu L^2)-Omega^2`. All non-axial degrees of freedom
are constrained for this control; it does not represent a freely bending
shaft's mode. Meshes of 2, 4 and 8 elements must exhibit second-order tip and
frequency convergence, retain the support reaction and meet fixed absolute
and relative error thresholds.

The first test mistakenly reused an off-diagonal inertia fixture: its
`Omega cross (I Omega)` was nonzero, so the full solver correctly departed
from the one-dimensional reference. The control now asserts a principal spin
axis explicitly. General rigid-body tests retain the off-diagonal tensor.
No production physics or solver/test tolerances were relaxed to resolve this.

## Validation and Completion Boundary

The initial tests fail on the absent attachment module. After implementation
and correction of the reference assumption, all ten focused Windows tests
pass (6.23 s). They cover root/interior/tip locality, immutable composition,
kinetic energy, force/moment, stiffness, gyroscopic power, continuum convergence
and malformed node/body/frame refusal. The API baseline adds only this private
module's empty exported surface. The first broader Linux run passes 599 tests
with two optional CAD skips and three optional-plugin warnings (242.33 s).
Mypy then identifies the incompatibility of an `int` annotation with the
`numbers.Integral` runtime test. The node guard now follows the existing
`IndexedPointLoad` convention using `int`/`np.integer`, refusing Python and
NumPy booleans. Twelve final Windows tests pass (10.86 s), including NumPy
integer acceptance and boolean refusal; scoped mypy passes. The final Linux
golf/API run passes 601 tests with the same two skips and three warnings
(292.46 s). The automatic inventory also links this new test to its existing
Rust math-primitives entry; this is generated discovery, not a Rust parity test.

This completes rigid nodal attachment mechanics, not loaded grip impedance,
flexible head modes, contact response, stability qualification, moving-boundary
power, full FRF/bandwidth convergence, calibration or acoustic prediction.
The full program and physical/blinded validation gates remain open.
