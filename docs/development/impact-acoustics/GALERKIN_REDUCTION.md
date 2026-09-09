# Work-Conjugate Shaft Reduction: Tools #5072

The private `GalerkinReduction` composes the existing `DampedPencil`, strict
array contracts and plant validation. It supplies a constant real subspace for
modal/FRF studies and later contact reduction. It does not select modes or
declare a frequency band qualified. Original public API entries are preserved.

## Equations and domain

The existing already length-scaled equation is

```text
M ydd + (G+C) yd + K y + r = f.
y approximately V z,  V in R^(N by n),  1 <= n <= N.
Mr = V.T M V, Gr = V.T G V, Cr = V.T C V, Kr = V.T K V.
rr = V.T r, fr = V.T f.
Mr zdd + (Gr+Cr) zd + Kr z + rr = fr.
```

The same projection acts on all four coefficients; G and C stay separate and
K need not be symmetric. V is an explicitly supplied real constant basis in
the existing y coordinates. A physical SI wrench must first use the existing
work-conjugate length transformation. No second length scaling is applied here.
The time scale T remains the same for full and reduced states `(y,T*yd)` and
`(z,T*zd)`. Projecting a force is not an initial-displacement projection.

For any reduced virtual velocity w, `(V w).T f = w.T (V.T f)`.
Likewise, the full and reduced kinetic quadratic forms agree on the subspace,
as do damping dissipation and elastic quadratic forms when K is conservative.
Skew G gives zero gyroscopic power in exact arithmetic. Nonsymmetric K retains
its circulatory work; no general elastic-potential or stability claim follows.
The reduced residual equals V.T times the full residual evaluated on V z.
It need not make the full residual or omitted motion small.

Both original and projected pencils pass the existing positive-resolved-mass,
skew-G and passive-C numerical checks. Invalid discarded original directions
are therefore not hidden by projection. Reduced mass conditioning checks
weighted basis independence. No basis normalization, pseudoinverse, damping
addition, symmetrization or eigenvalue clipping repairs a failed input.
Floating-point loss or conditioning can cause refusal even when an exact
algebraic basis is admissible. No certified forward-error bound is supplied.

The immutable record owns V and both pencils own their coefficients. Force
projection and motion lifting accept strictly real finite vectors and return
owned tuples. `basis_array()` returns a fresh array for complex FRF mapping:
input B becomes V.T B; observation O becomes O V. All node/port identities,
equilibrium and coefficient provenance remain those of the original model.
This private record is not a versioned interchange or provenance certificate.

## Primary literature and choice

Frie and Eberhard, _On shift selection for Krylov subspace based model order
reduction_ (2023), section 2.1, equations 3–7, gives the second-order Galerkin
input/output construction. Their paper also shows why selecting a subspace
requires transfer-error assessment. This implementation uses the basic
congruence; it does not implement or claim their greedy Krylov/SVD algorithm.
[Primary open paper](https://doi.org/10.1007/s11044-022-09872-7).

NASA NESC's _A Unified Approach to Modal Reduction Methods_ distinguishes
reduction bases from response recovery, including residual flexibility and
residual vectors. Those corrections are possible subsequent work, with their
own qualification; none is silently added here.
[Primary course description](https://nescacademy.nasa.gov/video/88799871c9244938b4a3d2eb73a18f8b1d).

## Independent checks and numerical example

TDD began with a missing-module failure before production code. The first 20
cases passed after implementation. Further cases check that omitting an
unstable direction cannot certify stability and that a complete nonorthogonal
basis preserves complex transfer and forced affine transients. Work and power
oracles compare physical quadratic forms and full residuals independently of
the projection implementation. Invalid mass/damping/gyro in omitted directions,
rank deficiency, unresolved mass, ownership and strict vector domains are checked.

The assembled eight-element straight stationary rod uses existing synthetic
parameters: L=1 m, EA=1000 N, line mass=0.2 kg/m, tip mass=0.1 kg, root axial
stiffness=400 N/m, damping=2 N s/m and inertance=0.03 kg. The head principal axes
and zero rotation make axial motion invariant for this control. Conservative
axial eigenvectors define bases with 1, 2, 4 and all 9 axial modes. Projected
damping retains its full coupled matrix; no modal damping diagonal is assumed.
The other 45 nodal directions are unexcited in this special fixture only.

For a collocated axial tip force/displacement, the selected angular frequencies
are 0, 10, 25, 50, 75, 100, 140 and 200 rad/s. Complex relative error is
`abs(Hr-H)/abs(H)`, magnitude error is `abs(abs(Hr)/abs(H)-1)`, and phase error
is the absolute principal angle of Hr/H. Both amplitudes exceed the
explicit fixture floor at these samples; phase at a transfer zero would not be
defined. The full 25 rad/s result also agrees with the existing SI tip FRF API.

| Axial modes | Max sampled complex error | Max magnitude error | Max phase error (rad) |
| ----------- | ------------------------- | ------------------- | --------------------- |
| 1           | 1.4938974                 | 1.3407381           | 2.4145414             |
| 2           | 0.1035115                 | 0.1030886           | 0.0542904             |
| 4           | 0.0051187                 | 0.0051092           | 0.0027228             |
| 9           | 1.37e-14                  | 1.31e-14            | 4.30e-15              |

These are Windows Python 3.12.10 numerical results, not measured club data.
The predeclared four-mode checks were 5% sampled complex error and 0.05 rad
sampled phase error; the observations did not set or relax these thresholds.
All nine axial modes reproduce the full driven axial response to roundoff.
A separate two-oscillator counterexample has <1% low-frequency error while
missing >99% of the higher-resonance response when that mode is omitted.

The initial 28 focused cases pass in 16.15 s. An additional overflow/tiny-load
control brings the combined Galerkin/transient/operating run to 81 passes in
27.68 s. Optional local JUnit collection initially produced four xunit2/property
compatibility warnings; the final run uses `-o junit_family=xunit1` and has none.
The broader Linux suite passes 888 tests in 245.03 s, with two optional CAD
skips and three unavailable-plugin configuration warnings, before the last
overflow test was added. That test passes separately on Linux (59.65 s including
collection). The explicit amplitude-floor check for both phase operands also
passes in the four-case rod rerun (11.29 s). Actual pre-push mypy
and repository Ruff 0.14.10 pass (3,797 formatted files). The API snapshot adds
only one empty-export module; unrelated snapshot formatting was compared as
parsed JSON and restored. Canonical inventory adds the new module and test
links, with no source changes to other modules. All nine final governance gates
pass, with the two existing release blockers retained. Normal publication
remains. New source is 100 lines; every new source/test function is at most 50.

The unmerged branch's classifier still labels this calculation non-calculation;
that known false negative is #5101 / PR #5103. It must be integrated and the
inventory regenerated before combined delivery. Structural gates confer no
scientific, textbook or publication approval; no hand-edited label is used.

## Remaining qualification

This is sampled modal convergence at one mesh, not a continuous-band error
bound or mesh-converged physical response. Dense/adaptive peak and antiresonance
checks, input/output-specific error budgets, mesh and modal refinement together,
shear/rotary inertia validity, rotating loaded states, time-domain excitation,
moving nonlinear boundary work and contact remain required. A small projected
residual or stable reduced eigenvalues cannot qualify omitted directions.
Radiation, calibrated grip/shaft measurements and blinded sweetness evidence
remain separate. No acoustic-equipment or player-effect claim is created.
