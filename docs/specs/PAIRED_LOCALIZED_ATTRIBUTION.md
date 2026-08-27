# Paired Localized Source-To-Downstream Attribution

## Purpose

This contract reports the downstream difference between two otherwise matched
model trials after one declared source parameter is changed. The source can
include an exact control point and half-open time window. Targets can be an
exact state coordinate on the governed trace grid, a registered impact scalar,
or a registered shot scalar.

The result is a model-scenario paired response. It is not a rank association,
global main effect, anatomical force attribution, human causal estimate, or
coaching recommendation.

## Required Identity

Every record binds the model, source adapter, attribution adapter, coordinate
frame, trace grid, variation plan, variable registry, execution document, and
source revision. Baseline and perturbed sides must have identical identity and
distinct trial IDs. Any drift fails before a response is calculated.

## Calculation

For pair (i), target (j), and source value (x), the signed response is

\[
\Delta y*{ij}=y^{\mathrm{perturbed}}*{ij}-y^{\mathrm{baseline}}\_{ij}.
\]

The local response per source unit is

\[
r*{ij}=\frac{\Delta y*{ij}}
{x^{\mathrm{perturbed}}\_i-x^{\mathrm{baseline}}\_i}.
\]

This quotient is only a finite paired secant at the declared baseline and
perturbation. It is not promoted to a derivative or global sensitivity when
the model is nonlinear. Zero source differences are rejected.

## Availability And Falsifiability

Missing, nonfinite, unsupported, no-impact, and numerical-failure observations
remain typed unavailable cells. They are never replaced with zero. State
targets remain available through a no-impact trial when the requested trace
sample exists; impact and shot targets do not.

Version 1 supports only independent, continuous, unbounded, one-at-a-time
source designs. Bounded, discrete, correlated, grouped, and simultaneous
interaction designs fail closed because this method cannot identify their
independent source contribution.

Analytical affine, sign-reversal, zero-response, nonlinear, missingness,
identity-drift, observational-confounding, and serial/chunk/resume falsifiers
qualify the implementation. The complete matrix of supported and unavailable
source layouts, adapters, and target metrics is recorded in
`docs/audits/rate_of_closure_r13_3_paired_attribution_capabilities.v1.json`.

## Reviewer Exports

Rows retain source locus, pair IDs, source values and delta, target metric and
locus, typed availability, baseline and perturbed values, signed response,
response magnitude, local response per source unit, and method identity.
Selectors filter existing immutable records without recomputing the analysis.
JSON snapshots and CSV exports preserve deterministic precision under bounded
pair, target, observation, and archive-size caps.
