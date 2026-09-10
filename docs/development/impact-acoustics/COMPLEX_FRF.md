# Complex transfer, coherence and impact sound

Tools #5155 is a bounded numerical continuation of IA-T5 #5074, under #5068.
The existing recording, signed-lag and spectral boundaries from merged #5106
remain canonical. This addition estimates a scalar complex transfer from
recordings. It does not identify a golf club, authenticate calibration, predict
radiated sound, or establish why one player's impact sounds better.

## Why phase belongs in the acoustic program

A modal radiation model has the schematic frequency-domain form
`p(omega, observer) = sum_r A_r(omega, observer) q_r(omega)`.
Both radiation coefficients and modal amplitudes are complex. Equal individual
magnitudes can produce very different sums through constructive or destructive
interference. A magnitude-only transfer cannot reconstruct this sum, a causal
impulse response, or the relative motion of a face, hosel and shaft.
This is a model-level inference, not an experimental explanation of sweetness.

Grip contact can change modal boundary impedance, damping and mode shapes. The
guitar muting analogy motivates testing those mechanisms, but does not establish
their magnitude during golf impact. A measured change also includes force-pulse
shape, impact location, ball, microphone position, acoustic reflections, sensor
phase, synchronization and player-dependent initial conditions. Separate those
causes using the matched-state and measured-data protocols in the parent epics.
Prestress/rotation may change structural transfer; this estimator itself has no
centrifugal-stiffening coefficient and makes no universal stiffening claim.

## Definitions, dimensions and implementation

Let x be the input and y the response, with already interpreted sample units
u_x and u_y. The API preserves these declarations and applies no sensitivity.
For each complete segment j, remove its mean and multiply by the symmetric
Hann window w of length L. Use stride floor(L/2), an unnormalized real FFT,
no tail padding, and the same segmentation for both records. Define

```
Axx[k] = mean_j |X_j[k]|^2
Ayy[k] = mean_j |Y_j[k]|^2
Axy[k] = mean_j conj(X_j[k]) Y_j[k]
D = sample_rate_hz * sum_n w[n]^2
Pxx[k] = a[k] Axx[k] / D
Pyy[k] = a[k] Ayy[k] / D
H1[k] = Axy[k] / Axx[k]
gamma2[k] = |Axy[k]|^2 / (Axx[k] Ayy[k])
```

The one-sided multiplier a is one at DC and even-L Nyquist, two elsewhere.
Pxx has units u_x²/Hz, Pyy u_y²/Hz, H1 u_y/u_x, and gamma2 is dimensionless.
The conjugation convention agrees with
[SciPy CSD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html);
the coherence definition agrees with
[SciPy coherence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html).
For the FFT's negative-exponent convention, a delayed response
`y[n] = g*x[n-d]` has transfer `g*exp(-i*omega*d)` in a compatible stationary
or periodic experiment. Finite windowing/leakage remains an estimation error.

`H1Settings` declares L and separate finite nonnegative input/response PSD
floors. Strictly greater input PSD supports H1; both PSDs must exceed their
floors for coherence. Zero floors request only algebraic positivity. These
floors are not inferred signal-to-noise or confidence thresholds. Retain the
settings alongside results in experiment records; the current result is a
numerical value, not a versioned acquisition/evidence wire.

`H1Estimate` contains immutable frequency and `H1Bin` tuples, complete-segment
count and input/response unit declarations. Unsupported ratios are `None`.
Silence has zero PSD and undefined H1/coherence. With supported input and a
zero response, H1 is zero and coherence is undefined. Zero complex H1 has no
meaningful phase. No unwrap or group-delay inference is performed.

Coherence normalization divides by square roots in stages to avoid directly
forming the overflow/underflow-prone product of auto-spectra. Cauchy-Schwarz
implies gamma2 <= 1. Overshoot of at most 64 binary64 epsilons is clipped at
one; a larger or nonfinite violation raises. This tolerance concerns arithmetic
only. PSD/cross-spectrum overflow is refused before support masking.

The legacy magnitude-only function shares the pair boundary and input/cross
spectra, retains its signature and zero-excitation refusal, and does not acquire
a new requirement that the unused response auto-spectrum be finite.

## Identification limits and required experiments

Two complete segments are required. A single-segment coherence would be one
where defined regardless of whether a useful transfer was identified. Even
with many segments, overlapping windows are correlated; segment count is not
effective degrees of freedom and this API supplies no confidence interval.

Under the model Y=H X+noise, the cross-spectrum identity gives H1=H only when
the input and output-noise cross term vanishes and input measurement error is
negligible. Correlated disturbances, input noise, multiple unobserved forces,
nonlinearity and time variation violate that inference. High coherence alone
does not establish causation, linearity, calibration or physical validity.

For an isolated impact, cutting one nonstationary force pulse into overlapping
Welch segments is not equivalent to an ensemble of repeated independent hits.
Use a justified stationary excitation or explicitly controlled ensemble/modal
test; assess window bias and leakage, repeatability and held-out transfer.
Full transient force-to-pressure identification remains a separate task.

A response-clock offset tau contributes a factor exp(-i*omega*tau), just like
a physical delay. Equal array lengths and sample rates do not establish common
time origin, clock drift, sensor phase, anti-aliasing or bandwidth. The existing
correlation heuristic must not silently align phase-sensitive identification.
Versioned calibration and acquisition identity, uncertainty, calibrated
observer transfer/radiation and blinded loudness-controlled preference trials
remain open. No synthetic test below supplies any of those missing data.

## Verification and integration

TDD retains the missing-module collection failure and both API-surface failures.
Independent controls include explicit SciPy odd/even Welch/CSD settings,
known delayed-tone phase/gain, output noise, strict bin floors, silence, zero
response, invalid immutable construction, tiny/huge signal scales, finite
arithmetic refusal and preservation of the legacy extreme-amplitude domain.
Predetermined comparison limits are in `tests/test_complex_frf.py`.

All 107 Windows ingestion/study-report/API controls pass (7.67 s), then pass
with 94.68% package coverage (7.09 s) above the unchanged 20% floor. Four production
modules pass NumPy-aware changed-file mypy; the actual isolated pre-push mypy
hook also passes. Every existing API symbol and signature remains unchanged;
three module entries and four package exports are additions.

WSL currently fails even `/bin/true` with getpwuid/filesystem I/O errors after
host disk exhaustion. No Linux result is claimed for this addition. The
separate load-history PR #5154 retains its already completed Windows/Linux
source-specific evidence. Reproducible temporary TAR recovery is recorded in
the session's `impact-reproducible-archive-recovery*.json`: verified SHA-256,
original mtime and retained Git tree recreate the exact archives using
`git archive --format=tar --mtime=@<mtime> <tree>`. JUnit and study records remain.

All nine final governance gates and root Ruff pass (3,863 files). The two
existing manual release blockers remain. Next: normal publication and protected
review, then
qualified acquisition identity, uncertainty and physical acoustic experiments.
