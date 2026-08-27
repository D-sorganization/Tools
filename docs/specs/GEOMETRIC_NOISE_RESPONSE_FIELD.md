# Geometric Noise-Response Fields

## Scope and Evidence Boundary

The geometric noise-response field compares two properties of a modeled swing
ensemble at every registered point and time coordinate:

1. absolute spatial scatter of eligible perturbed positions; and
2. modeled positional response to one declared input perturbation.

These are model-scenario quantities. They do not identify an anatomical force
source, establish a human control strategy, measure joint work or energy
transfer, or authorize coaching advice. The implementation supports the
falsifiable software and mechanics questions in [Tools epic
#4142](https://github.com/D-sorganization/Tools/issues/4142), specifically
[R12.3 issue #4765](https://github.com/D-sorganization/Tools/issues/4765).

## Governed Input Contract

Each response estimate consumes a baseline and perturbed
`TraceResamplingResult` produced by
`swing-trace-time-linear-contiguous/v1`. The input requires identical stable
point IDs, coordinate frame, time grid, ordered trial identities, and governed
execution metadata. It therefore inherits the resampler's rules:

- no extrapolation outside a trial's valid domain;
- no interpolation across an invalid interval;
- leading and trailing missingness remain unavailable;
- one-sample islands do not become interpolated trajectories;
- hit, no-impact, and numerical-failure rows retain their identities; and
- approximate impact-marker alignment error remains separate from physical
  event time.

Any frame, point, trial-order, input-registry, source-layout, plan, registry,
policy, or resume-contract drift fails closed.

## Estimands

For input (j), paired trial (r), sample (s), and point (p), let
(mathbf{y}^{(0)}_{rsp}) and (mathbf{y}^{(1)}_{rsp}) be the baseline and
perturbed Cartesian positions. Let

\[
z*{rj}=\frac{x^{(1)}*{rj}-x^{(0)}\_{rj}}{\sigma_j},
\]

where (sigma_j) is the standard deviation implied by the declared
perturbation distribution and its declared scale. It is not estimated from
the observed output range. The signed paired one-at-a-time response is the
through-origin vector coefficient

\[
\boldsymbol{\beta}_{jsp}=
\frac{\sum_{r\in C*{jsp}}z*{rj}
\left(\mathbf{y}^{(1)}_{rsp}-\mathbf{y}^{(0)}_{rsp}\right)}
{\sum*{r\in C*{jsp}}z\_{rj}^{2}},
\]

where (C*{jsp}) contains only trials for which both traces are available at
that cell. The nonnegative response magnitude is
(lVert\boldsymbol{\beta}*{jsp}\rVert_2); the signed Cartesian components
remain available and must not be replaced by the magnitude.

The matched absolute scatter uses the perturbed positions from the same paired
cohort:

\[
S^{\mathrm{matched}}_{jsp}=
\sqrt{\frac{1}{|C_{jsp}|}\sum*{r\in C*{jsp}}
\lVert\mathbf{y}^{(1)}_{rsp}-\bar{\mathbf{y}}^{(1)}_{jsp}\rVert_2^2}.
\]

The field also retains an all-eligible scatter computed from every available
perturbed row. Reporting both exposes changes caused by the response
denominator rather than silently conflating them with geometry.

## Interpretation

| Proposition                                  | Quantity That Bears on It                   | What the Quantity Does Not Establish                    |
| -------------------------------------------- | ------------------------------------------- | ------------------------------------------------------- |
| Geometry is spatially concentrated           | Absolute RMS scatter                        | Insensitivity to a declared input                       |
| Geometry responds to a declared perturbation | Signed response and magnitude               | Biological causation or active control                  |
| A local OAT response is small                | Paired OAT coefficient                      | Small simultaneous nonlinear response                   |
| A region is robust to missingness            | Matched and all-eligible counts and scatter | Robustness in unobserved trials                         |
| An input changes clubhead position           | Pointwise position response                 | Energy transfer, momentum redistribution, or joint work |
| A model outcome is stable                    | A separately declared outcome analysis      | Human repeatability or universal technique              |

“Quiet geometry,” “noise responsiveness,” “sensitivity,” “causal control,”
“joint work,” and “outcome robustness” are therefore different propositions.
They require different estimands and cannot be used as synonyms.

## Adequacy and Unsupported Designs

Every cell retains an integer availability count and one typed adequacy state.
Non-estimable response values remain `NaN`; they are never imputed as zero.

- fewer than two paired rows: `insufficient-pairs`;
- zero normalized perturbation energy: `zero-perturbation`;
- bounded input under the v1 estimator: `unsupported-bounded-input`;
- discrete input: `unsupported-discrete-input`; and
- correlated or grouped input: `unsupported-correlated-input`.

The paired OAT v1 estimator is qualified only for declared continuous,
unbounded, independently attributed perturbations. A grouped estimator,
nonlinear response surface, or causal design must use another method ID and
receive separate qualification.

## Falsification Protocol

The field is rejected if any of the following checks fail:

- a rigid translation changes centered scatter or paired response;
- a rotation does not rotate signed components while preserving magnitudes;
- a position-unit scaling does not scale every geometric output consistently;
- zero perturbation is reported as zero robustness rather than non-estimable;
- an affine golden system does not recover its analytical coefficient;
- an interaction-only countermodel is mislabeled as an OAT main effect;
- equal scatter under different declared scales fails to yield the expected
  normalized-response contrast;
- equal response with different residual geometry fails to separate response
  from scatter;
- missing-data classes alter identities or bridge unavailable intervals;
- a correlated design receives independent attribution;
- serial, chunked, and resumed records differ; or
- any source or execution identity changes without invalidating the
  fingerprint.

The exhaustive capability authority is
[`rate_of_closure_r12_3_noise_response_capabilities.v1.json`](../audits/rate_of_closure_r12_3_noise_response_capabilities.v1.json).
Only two current double-pendulum source/adapter cells are verified. Ten other
source/adapter cells remain explicitly unavailable; no result is fabricated
for those cells.

## Plot and Review Surface

`iter_position_noise_response_plot_rows` emits one immutable row per
input/time/point cell. Each row places the signed response, response magnitude,
matched scatter, all-eligible scatter, both counts, adequacy, frame, units,
method, normalization, source layout, adapter, scientific boundary, and final
field fingerprint together. Plotting code can filter or reshape these rows
without reading or duplicating the source trace tensors.

Reviewers should inspect response and scatter side by side, display unavailable
cells, and stratify by adequacy before interpreting a pattern. A visually smooth
field is not evidence that omitted interactions, model-form error, fatigue,
contact compliance, shaft dynamics, or participant variability are negligible.

## Reproducible Verification

From the Tools repository root with `PYTHONPATH` including `src`:

```powershell
python -m pytest src/shared/python/swing_sim/variation/tests -k "response or dispersion or resampling" -n 0 -q
python -m pytest tests/rate_of_closure -k "response or dispersion or variation_geometry" -n 0 -q
python -m ruff check src/shared/python/swing_sim/variation src/rate_of_closure tests/rate_of_closure
python -m ruff format --check src/shared/python/swing_sim/variation src/rate_of_closure tests/rate_of_closure
python -m scripts.check_design_manual_governance
python -m scripts.build_tools_module_inventory --check
```

The analytical fixtures are synthetic software-verification evidence. They are
not substitutes for governed participant data.
