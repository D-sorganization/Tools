# Rotating Loaded-Chain Roots: Private Verification

Tools #5072 remains partial. `RotatingSectionChain` composes the existing
`SectionChain` with one explicit weighted `SectionInertia` per section and a
prescribed `RotatingFrameState`. It reuses the strain-bounded clamped Newton
solver. No second elastic, mass, frame, load or root-finding implementation is
introduced. The deformed-frame precursor is published at `0f3e591db`.

## Balance and Consistent Derivatives

The instantaneous balance is internal minus applied plus required physical
inertia. Relative velocity and acceleration vanish at the candidate instant.
This does not mean a time-varying swing maintains that shape or follows a
sequence of quasi-static roots. The derivation and source basis for the inertial
terms are in `DEFORMED_FRAME_INERTIA.md`; the elastic interpolation and full
geometric tangent are in `LOADED_STATE_REVIEW.md`.

All nodes use the same right-increment material chart H(q)=H Exp(q).
The frame kernel returns a moving-material Jacobian; the elastic and applied
load kernels return a fixed-chart derivative. Before assembly, subtract the
block-diagonal connection whose action is ad(delta)^T residual/2. The existing
solver adds the connection of the complete residual to obtain its Newton
operator. Both directions reuse one helper, including away from equilibrium.
No symmetric projection removes Euler or applied-couple terms.

Node zero remains clamped at its exact supplied pose. Its retained residual is
the support-on-shaft material wrench. The returned energy is elastic energy
only; the existing force potential continues to exclude couples and frame
work. Neither field is a general total potential or a stability test.

## Independent Continuum Oracles

The synthetic uniform rod uses reference length L=1 m, axial stiffness
EA=1000 N and reference mass density mu=0.2 kg/m. Two-point Gauss samples carry
their physical mass and rotary-inertia weights. Density is not recomputed from
the extended length. These values are numerical controls, not a fitted shaft.

For prescribed axial origin acceleration a, axial position r(X) satisfies
EA r''=mu a, r(0)=0, r'(L)=1. Thus

```text
r(X) = X - mu*a/EA * (L*X - X²/2)
root support = +mu*a*L
```

The three-element model at a=5 m/s² recovers every nodal position to absolute
2e-10 m and the support to 2e-8 N (with zero remaining wrench components).

For rotation about an axis normal to the radial rod, omega=20 rad/s gives
k=omega sqrt(mu/EA), and balance becomes EA r''+mu omega² r=0. Hence

```text
r(X) = sin(k*X) / (k*cos(k*L))
r(L) = tan(k*L)/k
root support = -EA*(sec(k*L)-1)
```

The fixture has kL=sqrt(0.08), safely below the first singular denominator.
Meshes of 2, 4 and 8 elements reduce tip-position error at approximately second
order: each adjacent error ratio lies between 3.8 and 4.2. The final tip error
is below 1e-5 m and support error below 0.02 N. This verifies centrifugal
extension. It does not establish the sign of every transverse or torsional
dynamic frequency shift; those require the complete loaded operators.

## TDD, Contracts and Remaining Work

The new tests first failed because the rotating-chain module did not exist.
The independent matrix-exponential test helper was generalized from two nodes
to arbitrary node count before evaluating the assembled three-node derivative.
All 324 moving-Jacobian entries agree with independent centered differences
(absolute/relative 2e-7); a nonconservative case retains nonzero antisymmetry.
The zero-frame case preserves static residuals, tangents and solver roots.
Wrong records or quadrature counts are refused. A root beyond explicit strain
limits raises nonconvergence instead of returning a false candidate.

All six new tests pass. The first broader run exposed a changed TypeError
message; the implementation now retains the existing `SectionChain` identifier.
The complete golf/API run passes 560 tests, with two optional build123d CAD
skips (80.82 s). Ruff 0.14.10 passes. The API addition is private with no exports.
Tracked inventory and handoff digests are refreshed; all nine manual checks pass.
Normal commit and push hooks remain required.

Every candidate still reports stability as unqualified. Next work must assemble
the complete loaded mass and gyroscopic operators, qualify stability/refusal
and boundary work, add moving passive grip conditions, and establish modal,
mesh, time and FRF convergence. Contact, radiation, acoustics and physical or
blinded perceptual validation remain outside this root-finding checkpoint.
