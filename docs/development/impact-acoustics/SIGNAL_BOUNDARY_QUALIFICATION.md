# IA-T5 Waveform and Spectral Numerical Boundaries

This is a partial repair under Tools #5074, reusing the ingestion package
merged in #5084. No recording, calibration, radiation model, golf experiment
or blinded preference result is created. The full cross-repo program stays open.

## Reproduced Defects and Contract Changes

The initial synthetic boundary run has 16 failures and seven passes (9.85 s).
Complex samples lose their imaginary part, Boolean and object samples enter as
numbers, sample arrays remain writable, negative lags wrap around, a two-point
Hann produces NaNs, zero excitation permits an undefined H1, zero ringdown
returns NaN parameters, and overflowing PSD returns nonfinite values. An
independent SciPy comparison also exposes PSD's whole-record detrending versus
H1's segment detrending. These are software counterexamples, not observations
about a player or club.

The repair keeps the existing public signatures and magnitude-only H1 result.
Intentional behavior changes are strict real input, immutable owned samples,
signed linear lag, per-segment PSD detrending, and refusal of undefined or
nonfinite estimates. Existing valid sample bytes and legacy hashes are preserved.
Consumers must not depend on mutating a recording, circular lag indices,
undefined spectral bins or the former global-detrending PSD.

## Samples and Alignment

The shared private sample boundary accepts finite one-dimensional real numeric
arrays and real Python sequences. It refuses complex, Boolean, string and
object arrays before float conversion. A recording stores a fresh float64 view
over immutable bytes, so changes to the caller's array and re-enabling write
flags cannot alter its recorded samples. This does not authenticate provenance.

Alignment uses full linear cross-correlation of response against reference,
with zero extension outside each record. Lag k is positive for a delayed
response x[n-k]; negative lags represent advances. There is no circular wrap.
The maximum must be finite, positive and unique. Exact ties and silence are
refused. No normalization, detrending, polarity reversal, sub-sample delay,
clock drift, overlap threshold or uncertainty is inferred. A unique peak can
still be unreliable in noise or reverberation; this helper is a heuristic.
The sign and finite-record convention follow the
[SciPy correlation definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html).

## Shared Spectral Preparation and Units

For window length L, stride floor(L/2), segment j and sample n, define
z_j[n] = (x_j[n] - mean(x_j)) w[n], with a symmetric Hann window w.
Only full segments are retained. Both PSD and H1 reuse the same preparation.
L must be an integer from three through the recording length: the symmetric
Hann at L=2 has zero energy. An explicit window avoids dependence on a
library's periodic-versus-symmetric default.

Let X_j be the unnormalized rFFT of z_j. The one-sided PSD is
P[k] = a[k] mean_j(|X_j[k]|^2) / (fs sum_n(w[n]^2)), where a=2 for paired
positive/negative bins and a=1 at DC and the even-L Nyquist bin. Its units are
sample-unit squared per hertz. Integrating bins therefore yields window-weighted
detrended segment power, not an exact assertion about the original transient's
mean square. Odd/even windows and changing segment offsets are checked against
[SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
with every relevant setting explicit, to relative 1e-12 and absolute 1e-18.

H1 uses mean(conj(X_j) Y_j) / mean(|X_j|^2), then returns its magnitude for
compatibility. The common density normalization and one-sided factors cancel.
The cross-spectrum convention agrees with
[SciPy CSD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html).
A zero excitation bin is undefined and refuses the whole legacy result;
no arbitrary damping, epsilon divisor or fabricated zero transfer is inserted.
Positive finite excitation is only an algebraic condition. Noise floors,
coherence, phase, uncertainty and a supported-bin mask need an explicit richer
contract before identification claims. PSD of a zero record is valid zero;
H1 or modal decay from that record is not identifiable.

## Evidence and Remaining Work

All 41 ingestion/spectral tests pass (24.04 s), including the 23 new boundary
cases. Actual pre-push mypy passes for the new/modified calculation modules and
boundary tests. All nine API tests pass; only two empty private-module entries
are added, with no public signature changes. The final Linux ingestion/report/
API run passes all 69 tests (152.47 s), with three optional-plugin configuration
warnings. All nine final manual/inventory gates pass, as do pinned repository
Ruff and actual six-file pre-push mypy. Normal protected delivery remains pending.

The first Linux run times out in the inherited oscillator fixture's 16,384-point
direct convolution. Its full linear convolution now uses
[SciPy FFT convolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html),
with an independent 256-sample causal prefix checked against direct convolution.
The fixture includes the time-step quadrature factor and correctly describes
force excitation of a unit-mass oscillator. Neither the resonance tolerance
nor the 60-second test limit is relaxed. Existing ringdown controls remain green.

The legacy raw_data_hash still omits sensitivity and full acquisition identity;
its docstring now states this boundary. A versioned provenance and
calibration contract remains required rather than silently changing archived
hashes. Sample units are caller declarations; sensitivity is not applied here.
Nyquist margin does not establish anti-aliasing or sensor bandwidth. The
single-mode Hilbert estimate retains its finite-record, modal-isolation,
noise-floor and fit-selection limits. It supplies no general modal diagnosis.

Next: versioned calibration/acquisition identity; synchronized timebase and
supported-bin complex FRF/coherence/uncertainty; qualified radiation and observer
transfer; held-out physical validation and blinded sweetness study. Preserve
Tools #5075 report boundaries and UpstreamDrift's exact provider-pin integration.
Scientific-import inventory correction is separately owned by PR #5103; merge
that authority and regenerate this branch before final integrated delivery.
Generated candidate labels must not be hand-edited into scientific approval.
