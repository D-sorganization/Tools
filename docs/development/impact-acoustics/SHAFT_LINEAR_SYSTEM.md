# Stationary Spatial Shaft Reference and Turnover

Tools #5072 remains open. `golf_club.shaft_linear_system` now assembles a
stationary distributed shaft with full tip-body inertia and the passive grip.
This is the zero-rotation reference for the remaining rotating model. It does
not compute contact, radiation, human calibration or a player's sound quality.
The engineering publication authority remains `manuals/tools` and its gates.

## Inputs, Coordinates and Sources

`ShaftRodProperties` reuses `ShaftProfile` and its measured/raw station grid,
trim, frame and provenance. It additionally requires positive axial EA [N]
and polar mass moment per unit length [kg m] at every station, plus separate
provenance for those arrays. No isotropic material relation or uniform-density
annulus is inferred from EI, GJ or diameters. Profile damping ratios remain
unapplied; converting local ratios to a global damping operator requires a
separately qualified constitutive or modal definition.

The straight shaft axis is +z, with nodes running from exposed butt to exposed
tip. The six nodal coordinates are [ux, uy, uz, theta_x, theta_y, theta_z], with
translations in metres and infinitesimal rotations in radians. Nodal conjugate
loads are [Fx, Fy, Fz, Tx, Ty, Tz]. The existing midpoint bending stiffness and
consistent mass kernels are reused. Bending uses cubic Hermite displacement;
axial extension and torsion use linear interpolation. In this convention,
ux' = theta_y and uy' = -theta_x. Principal EI axes are rotated using the
interpolated spine angle, retaining cross-axis bending terms.

`ShaftAttachments` supplies an optional `ComponentMassProperties` at the exposed
tip and `PassiveGripImpedance` at the butt. Both must use the shaft frame. The
head COM vector is relative to the tip; it is not a position from the butt.
Inserted shaft mass is excluded from the exposed beam, so it must be explicitly
included in attached mass properties when modeling a real assembly.

## Energy and Assembly

For local principal bending axes, the stationary reference energies are

```math
U=\tfrac12\int_0^L
 [EI_y(u_x'')^2+EI_x(u_y'')^2+EA(u_z')^2+GJ(\theta_z')^2],ds,
```

```math
T=\tfrac12\int_0^L
 [\mu(\dot u_x^2+\dot u_y^2+\dot u_z^2)+j_p\dot\theta_z^2],ds.
```

Transverse cross-section rotary inertia, shear deformation, warping and
constitutive bend-twist coupling are omitted. These omissions require a
bandwidth study against an extended beam model or measured FRFs before using
the model for short-contact or acoustic-frequency predictions. The inclusion
of bending, axial and torsional coordinates has precedent in the
[Rodrigues et al. rotating-beam formulation](https://arxiv.org/html/2401.17519v1);
its small-motion assumptions do not validate a golf shaft or this implementation.

The two-node axial and torsional stiffnesses are rigidity/L times
[[1,-1],[-1,1]]. Consistent masses are density\*L/6 times [[2,1],[1,2]], using
linear mass density for extension and polar mass density for torsion.
Midpoint properties converge with mesh refinement but do not exactly integrate
every variable-property or station-knot case. The old unloaded modal path is
unchanged; the zero-spine bending blocks recover its numerical frequencies.

For a tip reference velocity v and angular velocity omega, the attached body
contributes the full independent kinetic energy

```math
T_h=\tfrac12 m\|v+\omega\times r\|^2+
\tfrac12\omega^T I_{COM}\omega.
```

The resulting six-by-six spatial inertia retains off-diagonal COM inertia and
translation/rotation coupling from the COM offset. It is added at the tip only.
The grip's M, K, C Gram products are added at the butt only, preserving its
passive coupled-port law. This represents a stationary reference anchor; it
does not silently substitute grip inertance for absolute human-body mass.

Thus M qdd + C qd + K q = f has the constant-coefficient identity

```math
\frac{d}{dt}(\tfrac12\dot q^T M\dot q+\tfrac12 q^T Kq)
=\dot q^T f-\dot q^T C\dot q.
```

No clamp is imposed or node eliminated during assembly. A free bare shaft
therefore retains six rigid modes. Numerical checks refuse nonfinite matrices,
an unrepresentable element-length cube, or a mass matrix that fails Cholesky.
Outputs contain profile/rod provenance, attachments, exposed node positions,
frame and profile identifiers, model version and assumptions. Arrays are
independent writable snapshots; mutating one does not alter a later assembly.
They are not an immutable interchange or calibrated FRF report (T6 remains).

## Verification

- Initial RED: test collection fails because the public module is absent.
- Initial GREEN: 17 tests pass in 7.16 s, including six rigid null vectors and
  independently integrated rigid kinetic energies; static cantilever bending,
  axial and torsional compliance; old bending modes; axial/torsional mesh
  convergence; full offset-body kinetic energy; butt-only grip assembly;
  rotated anisotropic bending; and input/frame refusals.
- The 4/8/16-element rod refinements reduce first-frequency errors by more than
  a factor of three per refinement, with final relative error below 0.0005.
- Broader reference checks pass 21 tests in 31.67 s, adding raw/exposed trim
  integration, writable-storage independence, overflow refusal and externally
  driven stationary work closure. The latter independently integrates the ODE,
  dissipation and applied work with SciPy DOP853; it is not a new production
  integrator or a claim about rotating/impact energy closure.
- Additional RED: two failures expose torsional mass underflow and an element
  length whose cube underflows before stiffness assembly. Explicit numerical
  domain checks correct both without altering or repairing the inputs.
- Final combined stationary shaft, tensile shaft, grip, legacy modes and API
  run passes 84 tests in 35.63 s. Changed three-module mypy passes with
  `--follow-imports=silent`; pinned Ruff checks pass.
- Broader golf-club suite: 372 pass and two skip in 192.03 s. Repository-wide
  Ruff passes and all 3,707 Python files meet the pinned formatter.

## Remaining T3 Work

Derive rotating-base inertia, gyroscopic/Coriolis, Euler and centrifugal
softening terms from consistent kinematics, including loaded initial state
and full head inertia. Combine geometric stiffness without double-counting
centrifugal work. Qualify inertial/rotating frame agreement, tensile/unstable
domain checks, time/mesh/modal/FRF magnitude-and-phase convergence and moving
boundary work. Compare shear/rotary-inertia extensions or measured FRFs to
establish an actual valid frequency band. Identify passive grip coefficients
with phase and uncertainty from measurements. No empirical data are supplied.
T4-T6, exact-pin consumers/studies, measured/blinded validation and final
AffineDrift synthesis remain in PROGRESS.md; this reference closes none of them.
