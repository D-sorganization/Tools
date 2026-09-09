# Six-Axis Grip Impedance Implementation and Turnover

T3 #5072 remains incomplete. This checkpoint adds a passive local constitutive
port; it has not yet been assembled with the distributed rotating shaft and
full head inertia. It supplies no human calibration or acoustic prediction.
Canonical engineering publication remains governed by `manuals/tools`.

## Coordinates and Constitutive Law

The public module is `golf_club.grip_impedance`. It exports immutable
`PassiveGripImpedance`, `GripPortState`, `GripPortResponse`, an evaluator, a
frequency-domain impedance, and fixed-frame transformation functions. It
reuses the existing `RigidTransform` point convention and identifier checks.

Let q = [u; theta] contain relative translations in metres and infinitesimal
rotations in radians at the same local reference point. Its velocity and
acceleration are time derivatives in that fixed linearization. Let the
conjugate input wrench be [force; torque] in N and N m. Positive power enters
the impedance. Its reaction on the attached shaft is the negative wrench.

\[
w=M\ddot q+C\dot q+Kq,\qquad
M=F_M^T F_M,\ C=F_C^T F_C,\ K=F_K^T F_K.
\]

Users supply the three real six-by-six factors, not arbitrary coefficient
matrices. A factor may have zero rows, allowing a free or partially constrained
port, and off-diagonal entries, allowing translation/rotation coupling. Factor
entries have the dimensions needed to map velocity to square-root energy
(inertance), velocity to square-root power (damping), or displacement to
square-root energy (stiffness). Coefficient blocks consequently have different
SI dimensions. Do not interpret all 36 coefficients as N/m or kg.

This parametrization covers symmetric positive-semidefinite local M/C/K
coefficients by construction. It never projects an active supplied matrix onto
the passive set. In particular, giving a negative damping coefficient as a
factor is not how one supplies negative damping: a factor's sign does not
change its squared single-axis coefficient. This API cannot model active
control, reflex delay, nonsymmetric coupling or arbitrary frequency-dependent
human impedance. Fitting a measured FRF requires a separately qualified model
and validity band, potentially with additional passive internal states.

The inertance is an ideal relative-port constitutive term, not a direct estimate
of hand mass. A moving absolute hand/body with gravity and rotational transport
requires a body model and external-work accounting. Source identification is
mandatory, but a source string does not establish calibration.

## Power and Energy

For constant coefficients in the declared fixed local frame,

\[
E_M=\tfrac12\|F_M\dot q\|^2,\quad
E_K=\tfrac12\|F_K q\|^2,\quad
D=\|F_C\dot q\|^2\geq0,
\]

\[
w^T\dot q=\frac{d}{dt}(E_M+E_K)+D.
\]

Energy and dissipation are evaluated as squared factor norms, retaining
nonnegativity for rank-deficient coefficients. The computed power residual is
input minus storage rate minus dissipation; it is not relabeled as damping.
The response constructor refuses negative stored energies/loss, nonfinite
power, malformed reaction vectors and overflow. Responses retain the declared
frame and source identifiers. Factors and states are copied
into immutable tuples. Booleans, including mixed boolean/numeric lists, strings,
complex coefficients and nonfinite inputs are refused.

With the exp(i omega t) convention, the analytic wrench/velocity impedance is

\[
Z(\omega)=C+i(\omega M-K/\omega),\qquad\omega>0.
\]

Its Hermitian part is C. This is a mathematical transfer function of the local
model; it is not a measured FRF. Zero frequency is refused because the generic
stiffness contribution divided by velocity is singular.

## Frame Invariance

For the existing point transform x_to = R x_from + p, the linear/angular
motion map used here is

```math
A=\begin{bmatrix}R&[p]_\times R\\0&R\end{bmatrix},\quad
q_{to}=Aq_{from},\quad w_{to}=A^{-T}w_{from}.
```

Thus H_to = A^-T H_from A^-1 for each coefficient and the corresponding factor
is F_to = F_from A^-1. This preserves wrench power and quadratic energies.
The independent test checks rotated force and torque shifted by p cross force,
as well as power, storage rate and both energies. It does not merely compare
two uses of the same matrix helper.

The dual motion/force transformation and congruence rule are consistent with
[Featherstone's spatial transformation reference](https://royfeatherstone.org/spatial/v2/xforms.html).
Our ordering is explicitly linear then angular. A coordinate change is distinct
from physically moving the grip. Time-varying transformations require derivative
transport terms; these functions accept a fixed transform only and do not
replace the rotating-base equations still required by T3.

## Verification and Remaining Work

- Initial RED: collection fails because the public module does not exist.
- First GREEN: 18 tests pass in 5.35 s, covering diagonal mechanics, coupled
  damping, free/rank-deficient boundaries, frequency impedance, independent
  wrench/frame checks, periodic work closure, immutable storage and refusals.
- Additional RED: five failures expose mixed-boolean coercion and invalid
  externally constructed energy/power records.
- Corrected GREEN: all 23 grip tests pass in 4.23 s.
- Combined shaft/grip/API run including response provenance: 61 tests pass
  in 22.04 s. Two-module mypy passes with `--follow-imports=silent`.
  API additions retain the existing provider surface.
- Unrestricted import-following mypy reports 12 existing argument-type errors
  in untouched `fitting_document.py`; this is not a whole-package type pass.
- A controlled wrong-source mutation fails the provenance assertion and is
  restored. The attempted pre-implementation provenance run collected after
  the edit and passed; it is not recorded as RED evidence.
- The periodic work test integrates a known sinusoid over a closed cycle and
  checks both the independently known 25 pi J loss and storage cancellation.

Next assemble this port with full distributed axial, bending and torsional
shaft dynamics and head spatial inertia. Rotating transport, spin softening,
prestress consistency, time/mesh/modal/FRF convergence and full work closure
remain required. T4 contact, T5 acoustics, T6 surfaces, exact-pin UpstreamDrift
adapters/studies, measured/blinded validation and final AffineDrift synthesis
remain open. The guitar-string analogy motivates a boundary-damping experiment;
it does not identify the six-axis coefficients or prove that a player's grip
causes a preferred impact sound.
