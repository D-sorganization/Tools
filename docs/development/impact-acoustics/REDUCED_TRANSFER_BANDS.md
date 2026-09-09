# Full/reduced shaft transfer error across frequency bands

Tools #5072; synthetic finite-model qualification, 2026-09-09.

This development derivation supports the private shaft kernels. It is not an
approved engineering-manual pathway, physical club calibration, acoustic
prediction, or stability certificate. The four new modules export no public
symbols. Existing Galerkin, full-pencil and frequency-interval contracts apply.

## Physical coordinates and explicit channel normalization

Use the existing work-conjugate coordinates q=S y, where translation entries
of S are the declared length scale and rotation entries are one. The pencil
has coefficients transformed by S.T A S; its conjugate loads transform by S.T.
For physical input channels u and displacement observations v, prescribe
f=F u and v=O y. Thus F already includes S.T and O already includes S. For
example, an axial SI tip force uses a column with the length scale in its axial
entry, and its SI tip displacement uses the corresponding row.

Each channel has an explicit positive reference magnitude su or sv in its own
physical unit. Define B=F diag(su), C=diag(sv)^-1 O. Then the normalized transfer
H=C D^-1 B is dimensionless. This allows different force/torque and displacement/
rotation channels without taking a norm of incompatible raw SI entries. Numeric
arrays cannot prove the declared units or identify a real grip: that remains
the caller's responsibility. Maps are constant, real and owned. This pathway
does not silently convert displacement to velocity or acoustic pressure.

## Constant pencils, uncertainty and a first absolute error bound

With exp(+i omega t), D(omega)=K-omega^2 M+i omega G+i omega C_d. Gyroscopic
G and passive damping C_d remain separate; nonsymmetric K remains intact.
The real basis y=V z produces Dr=V.T D V and projected input V.T B. No
orthonormality, omitted stability or uncoupled modal damping is assumed.

On a symmetric closed interval centered at c with half-width h, the existing
residual-aware Neumann assessment supplies a computed center inverse X,
an inverse bound R and a difference bound d such that, in exact arithmetic,
norm(D^-1)<=R and norm(D^-1-X)<=d. It retains norm(I-X D(c)); X is not assumed
exact. See FREQUENCY_INTERVALS.md. Frobenius norms majorize the operator norms.

A declared uniform full-pencil perturbation epsilon projects to at most
norm(V)\_F^2 epsilon. An additional reduced-pencil perturbation is explicitly
added in reduced coordinates. Neither value is inferred from a solve residual.
The bounds cover these dynamic-stiffness perturbations, not uncertainty in
ports, basis selection, material laws, frequency/time variation or radiation.

Let E0=C(Xf-V Xr V.T)B. Triangle and submultiplicative inequalities give

    norm(Hf-Hr) <= norm(E0)_F
        + norm(C)_F norm(B)_F (df + norm(V)_F^2 dr).

This is an absolute normalized error. It remains meaningful at an antiresonance.
Relative and phase claims require an independently established nonzero response
floor; this API returns neither. Norms use scaled hypot rather than squaring
small values. Unrepresentable positive products and normalization are refused.

## Residual-corrected response polynomial

The first bound can lose cancellation between common full/reduced responses.
For delta=omega-c, write D=D0+delta D1+delta^2 D2, with
D1=-2c M+i G+i C_d and D2=-M. Construct P0=X B, P1=-X D1 P0 and
P(delta)=P0+delta P1. This is an approximation, not an exact Taylor series
whose remaining terms may be ignored. The exact nominal input residual is

    B-D P = R0 + delta R1 + delta^2 R2 + delta^3 R3
    R0 = B-D0 P0
    R1 = -D0 P1-D1 P0
    R2 = -D1 P1-D2 P0
    R3 = -D2 P1.

For the actual pencil D+Delta, add the residual -Delta P. Consequently

    rho = sum(j=0..3, h^j norm(Rj)_F)
        + epsilon (norm(P0)_F+h norm(P1)_F)
    norm((D+Delta)^-1 B-P) <= R rho.

Both R0 and R1 are retained even at zero width; finite solve errors do not
disappear because the formula resembles a Taylor expansion. Apply the same
construction to the reduced pencil with input V.T B and its declared error.
Writing the two remainder bounds as rf, rr gives

    norm(Hf-Hr) <= norm(C P0f-C V P0r)_F
        + h norm(C P1f-C V P1r)_F
        + norm(C)_F rf + norm(C V)_F rr.

The implementation reports both this bound and the inverse-variation bound,
and uses their minimum. Both follow from explicit residual corrections in exact
arithmetic. Floating matrix assembly, norms and final arithmetic are not
outward-rounded, so the status is **conditional-numerical**, not certified.

This local derivation is ours. The distinction between an output-error bound
and a practical estimator is consistent with Feng and Benner's discussion of
omitted dual-residual corrections; their approximation-based estimators can
underestimate error without further control. We do not claim to implement their
algorithm. See [Feng and Benner, arXiv:2003.14319v2, Theorems 3.1–3.3](https://arxiv.org/pdf/2003.14319).
Output-focused reduction bounds for general systems are also motivated in
[Feng, Antoulas and Benner, MPIMD/15-17](https://csc.mpi-magdeburg.mpg.de/preprints/2015/MPIMD15-17.pdf).

## Complete coverage and numerical qualification

The existing binary-endpoint subdivision engine is shared with the inverse-band
API. Every attempted paired cell counts against the budget, including rejected
parents; a failed full assessment can stop a pair before the reduced solve.
Cells are accepted only when both inverses resolve and their absolute transfer
error bound meets the target. Failure never returns a partial cover. Exhausting
the budget does not prove the target impossible: the bound may be conservative.

Independent controls include a two-mode oscillator's analytic omitted resonance
peak, multiple normalized channels with a rank-one omitted transfer, a coupled
two-by-two gyroscopic/circulatory adjugate, nonorthogonal complete bases,
explicit coefficient perturbations, antiresonance, corrupted center solves,
strict types/ownership/scaling, hidden poles and failure after partial progress.
New-module RED tests preceded implementation. The initial assembled 0–20 rad/s
request exhausted its budget with the first bound and later reached the
representable subdivision limit with the residual bound. Investigation found
undamped transverse modes at 10.19019669 and 10.23284437 rad/s in this fixture.
Its axial damping cannot remove them. A separate regression retains this refusal,
even for a complete basis: an unobservable singular full state need not imply
a divergent axial port, but this full-inverse framework cannot qualify it.

The supported assembled example therefore checks the entire **0–8 rad/s** band
(about 0–1.27 Hz), with all full nodal coordinates retained, four shaft elements,
finite root spring/inertance/damping and a tip mass. With a 1 N input reference
and 1 mm displacement reference, target 0.05 means an absolute 50 micrometre/N
transfer-error limit. Windows results are:

| Retained axial modes | Attempted paired cells | Maximum absolute normalized bound |
| -------------------- | ---------------------: | --------------------------------: |
| 2                    |                     21 |                      0.0466470656 |
| 4                    |                     19 |                      0.0404015955 |
| 5 (all axial modes)  |                     19 |                      0.0401672450 |

These are upper bounds, not actual errors or proof of monotone modal convergence.
In particular, the nonzero full-axial bound reflects conservatism and finite cell
width. Endpoint comparisons are supplementary diagnostics; the residual bounds
provide between-sample coverage. This example does not qualify an impact or
audible band. Physically identified grip/head/shaft parameters, broader disturbed
rotating/bending convergence, flexible contact, acoustic radiation, calibrated
measurements and blinded sweetness evidence remain outstanding in the parent epics.
