# Explicit waveform calibration and shared uncertainty

Review #5157 continues #5074 after complex-FRF PR #5156. This is an explicit
affine numerical conversion y = g\*x + b with declared validity and provenance.
It preserves the legacy WaveformRecording/raw_data_hash API. Caller declarations
and content hashes do not authenticate calibration or establish a measured
physical/acoustic effect.

The [VIM calibration definition](https://jcgm.bipm.org/vim/en/2.39.html)
distinguishes calibration from verification. First-order uncertainty uses the
[NIST TN 1297 covariance law](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty).
For independent raw-sample errors and calibration parameters, the intended
covariance is g² diag(u_x²) + J_c C_gb J_c^T, with row J_c[i] = [x_i, 1].
Common gain/offset uncertainty is correlated across samples and must not be
divided by sample count as independent noise. Unknown components stay unknown.
The implementation also provides exact independent-block second moments.

## Model and derivation

Let Y_i = G X_i + B, where g, x_i and b denote means. The joint block (G, B)
must be independent of the complete indication vector X. Its gain and offset
may be correlated; raw indication errors are mutually independent in this API.
Finite second moments are required. No Gaussian distribution is required.
Writing C_x = diag(u_x^2), C_gb for the gain/offset covariance, and J_c[i] =
[x_i, 1], direct expansion of E[Y_i Y_j] - E[Y_i] E[Y_j] gives

```text
E[Y] = g*x + b
C_y_exact = (g^2 + u_g^2) C_x + J_c C_gb J_c^T
C_y_first_order = g^2 C_x + J_c C_gb J_c^T
```

The exact law retains the product u_g^2 C_x omitted by linearization. This is
our derivation for the declared affine model, not a general exact replacement
for uncertainty propagation. Correlation between indications and calibration
parameters invalidates this factorization; filtered, temporally correlated
indication errors require a richer input covariance contract.

`UncertaintyPropagation.FIRST_ORDER` is the default; callers may select
`EXACT_INDEPENDENT`. The method is retained in content identity. For an
independent two-point gain G in {1, 3} and indication X in {0.5, 1.5}, equal
probabilities give mean Y = 2 and variance 2.25; first order gives 2. This
complete, non-Gaussian ensemble is an independent regression oracle.

Gain has output-unit/input-unit dimensions; offset and converted samples have
output units. Covariance has output-unit squared dimensions. The weighted-sum
API requires dimensionless weights and returns standard uncertainty in output
units. It preserves shared calibration covariance: averaging 100 indications
does not divide a common offset standard uncertainty by ten.

The shared component uses factors a_i = x_i*u_g + rho*u_b and
b_i = sqrt((1-rho)(1+rho))\*u_b, including perfect correlations. Hypotenuse
accumulation avoids unnecessary squaring in standard-uncertainty calculations.
Finite output is required; representable standard uncertainty does not promise
that its squared covariance is representable. There is no general floating-point
error bound or coverage-probability claim.

## Integration and contracts

`waveform_calibration` is an explicit submodule; existing package imports and
legacy `WaveformRecording` signatures/hash semantics remain unchanged. It reuses
real-sample validation, immutable byte ownership, spectral finite-output checks
and the existing exact variation digest. No computation is copied into a consumer.

Raw acquisition retains channel and sensor-chain identity, declared unit, setup
digest, sample clock/rate/first time, timing evidence and source-file digest.
Calibration retains gain/offset, parameter uncertainty, units, indication range,
flat-response frequency band, validity interval/clock and certificate/method
references. Clock resolution, entire sample time extent, units, indication range,
finite values, immutable arrays and record types are checked unconditionally.
Bounds are inclusive; domains must have positive width. Negative nonzero gain
supports polarity reversal. Zero gain is refused.

The returned valid band is the intersection of the declared calibration band
and [0, sample_rate/2]. This does not filter the signal, qualify anti-aliasing,
or apply a frequency-dependent sensitivity or phase response. A timestamp is
relative to its named clock and does not establish UTC or synchronization.

Missing uncertainty stays `None`; explicit zero uncertainty is a distinct input.
Calibration-only covariance is separately accessible. Combined covariance is
unknown unless both the calibration and raw indication components are supplied.
No confidence interval, timing uncertainty or sensor-noise estimate is inferred
from the variation of an impact waveform.

## Identity and evidence limits

Version `swing_sim.calibrated_waveform/1` binds all constructor inputs, propagation
method and converted sample digest. Metadata float64 values use the shared exact
encoding; arrays use exact little-endian float64 bytes. The older 11-decimal
canonical JSON would erase small calibration differences and is not used here.
Tests change gain by one representable step even when zero indications leave
nominal output bytes unchanged. The calibration identity still changes.

Hashes establish content identity, not authenticity. Referenced setup, timing,
raw-file and certificate digests are not evidence that those files were supplied
or that their contents agree with decoded indications. `SourceKind.MEASURED`
remains a caller declaration; a synthetic source or synthetic calibration always
keeps the result synthetic. The [VIM definition](https://jcgm.bipm.org/vim/en/2.39.html)
provides terminology, not authentication of these records.

## Qualification and remaining work

Tests were written before implementation: missing-module RED for conversion
and identity, and a later missing-enum RED for exact moments are retained.
Controls cover independent covariance expansion, the complete discrete ensemble,
common-error averaging, perfect correlation, polarity, unknown components,
immutability, exact identity, legacy hash compatibility and boundary refusals.
Source-specific results are recorded in `CALIBRATION_RESULTS.json`.

This step supplies reusable numerical calibration records. Still required under
#5074/#5075 and UpstreamDrift #9703/#9704: authenticated ingestion and calibration
authority; clock/phase and correlated-noise models; uncertainty through spectral
estimators; consumer integration with pinned provider identity; calibrated
held-out recordings; observer/radiation models; and blinded preference studies.
It does not demonstrate that a player stiffens the club, transfers more ball
energy, or produces a sweeter sound. Those causal hypotheses require the
matched-state dynamics and measurement designs in the parent epics.
