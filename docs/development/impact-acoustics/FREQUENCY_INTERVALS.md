# Between-Sample Frequency Bounds: Tools #5072

A sampled frequency-response plot can miss a narrow resonance. The private
`_shaft_frequency_interval` kernel adds a conditional bound over a declared
interval of a constant finite-dimensional pencil. It composes the existing
`DampedPencil` and plant validation, retaining separate gyroscopic and damping
terms, nonsymmetric stiffness and the caller's coordinate scales. It neither
chooses modes nor identifies a physical bandwidth.

## Derivation and assumptions

For exp(+i*omega*t), use the already length-scaled dynamic stiffness

```text
D(omega) = K - omega^2 M + i*omega*G + i*omega*C.
omega = omega0 + delta, |delta| <= h, omega0 >= h >= 0.
D1 = -2*omega0*M + i*G + i*C.
D(omega) = D0 + delta*D1 - delta^2*M.
```

Let X be the computed center inverse, retained in the result as owned complex
tuples. Assume a uniform additive dynamic-stiffness uncertainty E(omega), with
spectral norm at most epsilon throughout the interval. This assumption must be
supplied in the same coordinates and units as D. Zero epsilon prescribes the
nominal coefficient model; it is not evidence of zero material or measurement
uncertainty. Frequency-dependent material/grip laws require an independently
bounded discrepancy from this constant-coefficient model.

Define the following Frobenius-norm upper estimate for the spectral norm of
I-X[D(omega)+E(omega)]:

```text
q = ||I-X*D0||_F + h*||X*D1||_F + h^2*||X*M||_F
    + epsilon*||X||_F.
```

If q < 1, the Neumann-series argument gives, in exact arithmetic,

```text
||(D+E)^-1||_2     <= ||X||_F/(1-q) = R,
||(D+E)^-1-X||_2   <= q*R.
```

To see the second statement, write F=I-X(D+E), then
(D+E)^-1=(I-F)^-1 X and subtract X. The center inverse defect is included;
it is not silently treated as zero. This result does not require symmetric
stiffness, modal diagonalization or stability of the time-domain ODE.

The underlying inverse-perturbation argument is described in David Bindel's
Cornell CS 4220 lecture, January 30, 2026, section “Norms and Neumann series.”
The second-order frequency expansion and uncertainty/response application
above are the present derivation.
[Primary lecture](https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-01-30.html).

## Ports, units and phase

For one fixed force column b and displacement observation row l, let
H0=l X b. The scalar complex transfer lies in the disk centered on H0 with
radius r=||l||\_2 ||b||\_2 qR, subject to the assumptions above. Its magnitude
differs from |H0| by at most r. Only if r<|H0| can the phase difference be
bounded by asin(r/|H0|). If the disk contains zero, no phase qualification
follows: an antiresonance must not be hidden by phase unwrapping or an amplitude
floor presented as measured noise.

Use work-conjugate mappings: for q_SI=S*y, an SI force maps to S.T*f and an
SI observation maps through S. The existing gripped-shaft test uses a transverse
unit point force and its collocated displacement, with b=S.T\*f. Each transfer
entry retains its own units. An unscaled norm mixing torque and force ports
is not a dimensionless error metric. The kernel's inverse norm is coordinate
dependent; it is not an impact-quality or acoustic-amplitude score.

For full/reduced transfer comparison, both responses need interval disks.
The triangle inequality bounds their difference by the center discrepancy plus
both radii. A reduced-model bound alone cannot bound omitted full-system modes.
Mesh and continuum errors remain separate, as BENDING_CONTINUUM.md demonstrates.

## Numerical contract and verification

The input interval, uniform pencil-error assumption and contraction ceiling
are explicit. The ceiling must be strictly below one. An interval exceeding
that ceiling is refused; neither endpoint interpolation nor added damping
substitutes for resolution. Singular centers and nonfinite arithmetic are
refused. Hypot-based Frobenius norms avoid naive squaring underflow/overflow;
unrepresentable positive products are refused. Inputs are not modified.

This is floating evaluation of an exact-arithmetic theorem, reported as
`conditional-numerical`. The residual is itself computed, and norm, assembly,
matrix-product and final-expression rounding are not outward-rounded. Thus it
is not a machine-certified enclosure or a substitute for verified arithmetic.
The supplied epsilon is a model-error assumption, not an automatically inferred
rounding certificate. `stability_status` remains `unqualified` even when a
frequency interval resolves for a negative-stiffness unstable oscillator.

TDD began with the missing module. The first 17 controls passed. An additional
failing ownership test required retaining the actual computed X used by the
bound, returned only as fresh arrays. The final 23 controls include:

- A scalar damped oscillator against a separate complex reciprocal and the
  independently evaluated Neumann expression, with positive/negative real and
  imaginary coefficient perturbations.
- An undamped pole at 2 rad/s inside an interval with finite endpoint samples:
  the interval is refused without changing its tolerances.
- Coupled nonsymmetric stiffness and gyroscopic terms checked against direct
  complex inverses; original invalid mass is still refused.
- A deliberately inaccurate center inverse: its 0.1 defect contributes to q
  and the resulting nonzero center difference bound.
- Strict scalar domains, uncertainty-induced refusal, singular-center refusal,
  frequency overflow and positive-product underflow.
- The existing four-element finite-grip bending model at 8 +/- 0.001 rad/s,
  with complex, magnitude and phase checks using the work-conjugate port disk.

All 23 Windows checks pass in 3.30 s. The prior combined interval/Galerkin run
passes 47 checks in 3.32 s, before the final shaft-port integration case.
Pinned Ruff and actual mypy 1.13 pass for the new source. The broader Linux golf/signal/API suite passes 920 tests in 220.12 s, with
two optional CAD skips and three unavailable-plugin configuration warnings.
Repository Ruff 0.14.10 passes (3,800 formatted files), all nine final manual
gates pass, and the API baseline adds only one private empty-export entry;
all existing API entries remain identical. Normal publication remains.

The current inventory classifier still labels this calculation non-calculation.
Its canonical entry is preserved with publication blocked; classifier PR #5103
is a delivery prerequisite, not scientific approval. No manual authority or
physical qualification follows from generated inventory metadata.

## Remaining qualification

This single-interval primitive does not yet implement automatic band subdivision,
peak localization, continuous full/reduced mesh comparison or a calibrated
acoustic-band study. Refusal can mean a conservative bound, poor scaling,
insufficient interval refinement, large uncertainty or a genuine singularity;
it is not a diagnosis of instability. Small intervals may be required near a
lightly damped pole. A finite pole-free frequency band does not prove autonomous
stability, nonlinear validity or absence of instability outside retained modes.

Rotating/moving operation, nonlinear boundary work, flexible contact, head
radiation, physical parameter identification and blinded listening evidence
remain open. No measured player, grip, launch or sweetness result is added.
