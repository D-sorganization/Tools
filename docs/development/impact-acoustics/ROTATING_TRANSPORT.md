# Rotating Head and Distributed Shaft Transport

Tools #5072 remains open. The new `rotating_body` and
`shaft_rotating_transport` modules assemble inertia in an accelerating,
rotating observer frame. They retain full head COM offset/inertia and explicit
section rotary inertia. They are coefficient builders, not a complete loaded
swing, contact or acoustic solver. Engineering publication remains governed by
`manuals/tools` and its qualification gates.

## Frame and Coordinates

`RotatingFrameState` records angular velocity Omega [rad/s], angular acceleration
alpha [rad/s²], and the frame origin's physical inertial acceleration a0 [m/s²],
all expressed in the observer frame. The user must provide orientation separately
when mapping a solution to inertial coordinates. The constructor rejects malformed,
nonfinite, boolean and string vector entries. Body and observer frame IDs must match.

For a body reference point p and COM offset rho, define the canonical small
perturbation q = [u; theta]. The reference body axes initially coincide with
the observer axes. Finite orientation is R = Exp(theta); its left Jacobian maps
theta's derivative to relative angular velocity. The exact kinematics used to
derive and independently check the tangent are

```math
r_c=p+u+R\rho,\qquad \omega_r=J_l(\theta)\dot\theta,
```

```math
v_c=\Omega\times r_c+\dot u+\omega_r\times R\rho,
\qquad \omega=\Omega+\omega_r.
```

An instantaneous inertial velocity boost can set the observer origin velocity
to zero. Its acceleration is retained through the inertial potential below;
constant translational velocity cannot change the local inertial coefficients.

```math
T=\tfrac12m v_c^T v_c+\tfrac12\omega^T R I_{COM}R^T\omega,
\qquad V_a=m a_0^T r_c.
```

The need to carry moving-frame velocities and full spatial inertia is consistent
with [Rodrigues et al., sections 2–4](https://arxiv.org/html/2401.17519v1).
Their quasi-velocity matrices cannot be copied unchanged into canonical
rotation-vector coordinates. The implementation here derives its own kinetic
derivatives and checks them against finite Exp(theta) rotations.

## Tangent Derivation

At q = qdot = 0, define M = T_vv, C = T_vq and H = T_qq. A subscript denotes a
derivative, and v means the canonical coordinate rate. C is linear in Omega;
H is quadratic in Omega. The linearized equation is

```math
M\ddot q+(C-C^T)\dot q+
[-H+C(\alpha)+(V_a)_{qq}]q=f_0+Q_{applied}.
```

Thus the separately returned terms are mass M, gyroscopic G = C-C^T,
centrifugal stiffness Kc = -H, Euler stiffness Ke = C(alpha), and origin-
acceleration stiffness Ka = (Va)\_qq. C is also returned for independent
derivative checks. Ke generally is not symmetric. It must not be relabeled
as damping or discarded just because G does zero quadratic work.

Let c = p + rho. The nominal forcing is the opposite of the inertial wrench
needed to keep the reference body at zero relative state:

```math
a_c=a_0+\alpha\times c+\Omega\times(\Omega\times c),\quad F_0=-m a_c,
```

```math
\tau_0=\rho\times F_0-I_{COM}\alpha-\Omega\times(I_{COM}\Omega).
```

The implementation returns f0 = [F0; tau0] explicitly. An external constraint,
grip load or elastic equilibrium must balance this wrench if zero relative
state is supposed to be the nominal configuration. Dropping it would invent
an equilibrium and fails the free-inertial-motion regression.

The rotational Hessian helper uses the identity

```math
\left.\nabla_\theta^2(a^T\exp([\theta]_\times)b)\right|_0
=\tfrac12(ab^T+ba^T)-(a^Tb)I.
```

The full COM kinetic energy produces translation/rotation coupling in M, C and
H. A point mass cannot reproduce these blocks. As a check on rotation-coordinate
conventions, a spherical inertia at its COM has zero centrifugal orientation
stiffness and rotational G = J[Omega]x, rather than the translational 2m[Omega]x.

## Distributed Shaft Integration

`ShaftRotaryInertia` requires (Jxx, Jyy, Jxy) [kg m] at every raw-profile station.
These are mass tensor entries in shaft axes. Their transverse tensor must be
positive definite. For the centered planar section used here, Jzz = Jxx + Jyy
must agree with `ShaftRodProperties`' polar mass density. Sources and frame/profile
identifiers are retained. No transverse moments are inferred from EI or diameter.
The profile and supplementary property data still require physical qualification.

`ShaftRotatingModel` combines that inertia, the existing rod and optional head
and grip. Section translations use cubic Hermite bending and linear axial
interpolation. Section rotations satisfy theta_y = ux', theta_x = -uy', with
linear torsion. A section motion map S therefore gives local q = S q_element.
For each positive quadrature length, density times length produces a physical
body mass and inertia. The body tangent is projected as S^T A S and S^T f0.

Four-point Gauss quadrature is split at raw-profile knots, using exposed-span
coordinates and the declared butt trim. This integrates the polynomial
transport terms without averaging across a density/inertia knot. The resulting
mass includes transverse section rotary inertia: it is a Rayleigh-beam mass,
not the previous Euler–Bernoulli mass. Existing elastic stiffness remains the
midpoint approximation; no stored damping ratio is silently converted to C.

The full head tangent is added at the exposed tip. The grip contributes its
ideal relative-coordinate M/C/K at the butt. Its inertance is not transported
as an absolute human-body mass. A real moving hand model, preloaded grip and
physical boundary-work audit remain separate requirements.

Global matrices must be finite and mass must pass Cholesky. Spin stiffness is
not projected onto a positive matrix: its negative terms are physical. A
subsequent equilibrium/stability gate must decide whether a proposed case is
within the supported stable domain. Shear deformation, warping and general
loaded-shape geometric stiffness are not present in this transport builder.

## Verification and Evidence Boundaries

- Rotating-head RED: collection fails because the public module is absent.
- First head GREEN: 12 tests pass in 9.66 s. Independent SciPy finite rotations
  and numerical differentiation recover M, C, H and the acceleration-potential
  Hessian; the Euler coefficient is checked at alpha. Other checks cover full
  nominal wrench, stationary spatial mass, spherical-inertia conventions,
  gyroscopic zero work, frame/input refusals and overflow.
- A free body with a centered COM is integrated in a frame with changing
  angular speed and translating-origin acceleration. Mapping its trajectory
  back to inertial coordinates recovers the independently prescribed straight
  path within 1e-10. This is a translational frame-agreement test, not a claim
  of finite-rotation accuracy for the linearized orientation variables.
- Distributed RED: collection fails because the shaft transport module is
  absent. The first combined head/shaft run passes 23 tests in 10.23 s.
  It covers Rayleigh rigid-motion energies, stationary stiffness, integrated
  Coriolis/spin blocks, radial resultant, full head/relative grip assembly,
  density-knot integration and section/profile/frame refusals.
- The expanded shaft suite passes 14 tests in 28.56 s. Its additional benchmark
  explicitly combines the existing radial tensile operator with this transport
  and solves the clamped quadratic gyroscopic eigenproblem. At dimensionless
  spin rates 3, 6 and 12, first out-of-plane frequencies match Table 5's exact
  references within relative 2e-4; in-plane values match its five-TITOP estimates
  within 2e-3. These in-plane references are approximate, not exact solutions.
  The reference uses very small section rotary inertia and a conservative
  nominal axial-strain bound below 1e-4. It does not establish a general loaded
  equilibrium or an empirical golf-shaft bandwidth.

- Combined shaft/grip/head/API verification: 110 passed in 31.76 s. The six
  added production modules pass scoped mypy with imports followed silently;
  existing recursive package diagnostics are not counted as a full mypy pass.

- Broader golf-club regression: 398 passed, two skipped in 109.87 s.
  Repository-wide pinned Ruff passes lint and formatting for 3,715 files.

The implementation is pushed at `59ca36c60c03a67ce8b9cfdfaac0d195f7863ad5`.
All nine manual/handoff gates and normal commit/push hooks pass, including unit
tests, type checks and dependency audit. No publication approval is implied.

## Next Required Work

Construct and qualify the loaded initial state, including axial/bending/torsional
coupling and moving boundary work; combine its geometric stiffness with these
transport terms without double-counting centrifugal effects. Add mesh/modal/time
and full/reduced FRF magnitude-and-phase comparisons, stable/compressive-domain
refusals and a shear/rotary-inertia bandwidth study. Retain arbitrary head offset,
full inertia and passive grip in the general case. T4–T6, exact-pin UpstreamDrift
adapters/studies, physical/blinded validation and final AffineDrift synthesis
remain open in PROGRESS.md. No numerical fixture is experimental evidence.
