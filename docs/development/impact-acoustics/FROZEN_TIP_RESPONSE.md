# Frozen Tip Force and Torque Response

Tools #5072 remains partial. Rigid nodal attachment is published at `42f229951`.
The new private response solver reuses current-geometry mass M, gyroscopic G
and full material stiffness K. Spectrum and response calculations now share
one clamped balance/strain check; it recomputes operators rather than trusting
a stored candidate. The root alone is constrained, and both equilibrium and
harmonic root reactions remain visible. Existing public APIs are unchanged.

## Point Port and Harmonic Equations

With the explicit exp(+i omega t) convention, angular frequency omega is in
rad/s and the frozen equations give D q = f, where

```text
D = K - omega^2 M + i omega G.
v_point = B [v_tip; angular_velocity_tip]
B = [[I, -skew(c)], [0, I]].
f_tip = B.T [force_at_point; torque_at_point].
```

The supplied offset c is measured from the tip node in its material axes.
Thus point force adds c cross force to nodal torque, and angular motion adds
angular_velocity cross c to point translation rate. This is a power-conjugate
port map, retaining all six force/torque inputs. The added harmonic load is an
infinitesimal perturbation; products of its amplitude with perturbation-induced
port rotation are second order. Existing baseline-load derivatives stay in K.

Let S repeat (L,L,L,1,1,1) over free nodes, with an explicit positive length
scale L. Solve Dbar x = S.T f, where Dbar=S.T D_ff S, then q=S x. The six
unit point-wrench columns produce a full six-by-six displacement compliance
H and velocity mobility i omega H. Root transfer is D_root,free q, distinct
from the equilibrium support wrench. Row/column units follow material
translations/rotations and forces/torques; the matrix does not have one scalar
unit such as m/N. Zero angular frequency is the static perturbation limit.

## Numerical Refusal

The solver retains nonsymmetric K and skew transport; it adds no damping,
uses no pseudoinverse and performs no mass repair or stiffness projection.
The reciprocal condition of Dbar must exceed the declared numerical floor.
That check alone misses nearly cancelling coefficients: D=epsilon I has
condition number one even when K=I and omega^2 M=(1-epsilon) I.

A second required resolution measure is

```text
eta = sigma_min(Dbar) /
      (norm(Kbar,F) + omega*norm(Gbar,F) + omega^2*norm(Mbar,F)).
```

Eta must exceed the same explicit floor. This bounds coefficient-relative
numerical sensitivity in the chosen scaled coordinates; it is not a physical
parameter-uncertainty certificate. The direct linear-solve residual for each
column is norm(Dbar x-fbar)/[norm(Dbar,F)*norm(x)+norm(fbar)]. It must satisfy
the declared tolerance. Overflow, nonfinite output and unresolved solves fail
closed. Scales affect these numerical measures; resolved physical responses
must agree when coordinate scales change.

The importance of distinguishing small residuals from forward accuracy is
consistent with Tisseur and Meerbergen (2001), sections 4–5,
[The Quadratic Eigenvalue Problem](https://eprints.maths.manchester.ac.uk/466/1/38198.pdf).
The specific forced-response measure and port formulas above are derived here;
the citation does not qualify golf parameters, stability or an acoustic model.

## Independent Controls and Validation

For a nonrotating uniform rod with L=1 m, EA=1000 N, mu=0.2 kg/m and centered
tip mass mt=0.1 kg, the axial response satisfies
EA phi''+mu omega^2 phi=0, phi(0)=0, and
EA phi'(L)-mt omega^2 phi(L)=P. With k=omega sqrt(mu/EA), its tip compliance is
sin(kL)/[EA k cos(kL)-mt omega^2 sin(kL)]. The static limit is L/EA.
The one-element discrete reference independently gives
Hzz=1/[EA/L-omega^2(mu L/3+mt)] and root transfer
-(EA/L+omega^2 mu L/6) Hzz.

Tests cover these static/dynamic controls, second-order 2/4/8-element
compliance convergence, reciprocity in the nonrotating conservative case,
independent offset force/moment and motion transfer on a loaded rotating shaft,
zero mean reactive power, length-scale invariance, resonance refusal, domain
and balance refusal, malformed controls and frequency-square overflow.
The cancellation counterexample first fails despite a tiny direct solve
residual; the coefficient-resolution check fixes it without added damping or
relaxed tolerances. All 41 harmonic/spectral Windows tests pass (30.95 s),
and nine API tests pass with only two empty private-module entries added.
Scoped mypy passes. The final Linux golf/API regression passes all 620 tests
(216.99 s), with two optional CAD skips and three optional-plugin configuration
warnings. All nine documentation and inventory gates pass.

## Physical Completion Boundary

Every response remains stability-unqualified. These are algebraic particular
solutions of frozen linear equations, not validated steady responses of an
actual swing. A nonsingular D can exist for an unstable or nonautonomous model;
a singular mass can admit a frequency solution without qualifying an ODE or
descriptor system. No physical damping, valid frequency band, nonlinear impact,
grip impedance, head-face flexibility, radiation, microphone transfer or human
calibration is inferred. The rod controls establish only their stated axial
limit. Full modal/bandwidth and time-response convergence, moving-boundary power,
stability/refusal policy and physical/blinded acoustic validation remain open.
The scientific-import inventory correction is under separate PR #5103; this
branch must regenerate after that merges rather than edit classifier labels.
