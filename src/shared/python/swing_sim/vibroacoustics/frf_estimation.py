"""Complex Welch H1 with explicit PSD support; numerical estimation only.

Use this alongside the legacy magnitude-only estimate. Relative sensor phase,
timebase synchronization and calibration must be independently qualified before
physical identification. Neither synthetic signals nor coherence predict sound.
"""

from __future__ import annotations

import numpy as np

from ._frf_contracts import H1Bin, H1Estimate, H1Settings
from ._spectral_frames import finite_output
from ._spectral_pair import spectral_pair
from .measurement import WaveformRecording

# Cauchy-Schwarz may exceed unity by floating-point rounding, not by physics.
_COHERENCE_ROUNDOFF = 64 * np.finfo(float).eps


def _coherence(spectra: tuple[float, float, complex]) -> float:
    """Normalize in stages to avoid overflow in Sxx*Syy or |Sxy|²."""
    sxx, syy, sxy = spectra
    value = float((abs(sxy) / np.sqrt(sxx) / np.sqrt(syy)) ** 2)
    if not np.isfinite(value) or value > 1 + _COHERENCE_ROUNDOFF:
        raise ValueError("coherence violates finite Cauchy-Schwarz bound")
    return min(value, 1.0)


def _estimate_bin(
    spectra: tuple[float, float, complex], scale: float, settings: H1Settings
) -> H1Bin:
    """Keep unsupported ratios absent; refuse numerical failure before masking."""
    sxx, syy, sxy = spectra
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        powers = finite_output(np.array([sxx, syy]) * scale)
        input_psd, response_psd = map(float, powers)
        h1 = complex(sxy / sxx) if input_psd > settings.minimum_input_psd else None
        coherence = (
            _coherence(spectra)
            if h1 is not None and response_psd > settings.minimum_response_psd
            else None
        )
    return H1Bin(input_psd, response_psd, h1, coherence)


def estimate_complex_frf_h1(
    force_recording: WaveformRecording,
    response_recording: WaveformRecording,
    *,
    settings: H1Settings,
) -> H1Estimate:
    """Return H1=mean(conj(X)*Y)/mean(|X|²) and magnitude-squared coherence.

    Require equal rates/lengths, explicit settings and at least two complete
    segments. Reuse symmetric Hann, segment means and floor(L/2) stride. Each
    PSD must exceed its caller-declared floor strictly; H1 needs input support,
    coherence needs both. Unsupported bins are None. PSD units are declared
    input²/Hz and response²/Hz; H1 units are response/input. No sensitivity is
    applied, alignment inferred, tail padded, phase unwrapped, or uncertainty
    estimated. Finite arithmetic and immutable output are postconditions.
    """
    if not isinstance(settings, H1Settings):
        raise TypeError("settings must be H1Settings")
    pair = spectral_pair(force_recording, response_recording, settings.segment_length)
    count = int(pair.input_fft.shape[0])
    if count < 2:
        raise ValueError("at least two complete spectral segments are required")
    sxx, syy, sxy = pair.cross_spectra()
    factors = np.full(sxx.size, 2.0 / pair.density_divisor)
    factors[0] *= 0.5
    if settings.segment_length % 2 == 0:
        factors[-1] *= 0.5
    bins = tuple(
        _estimate_bin((float(x), float(y), complex(xy)), float(scale), settings)
        for x, y, xy, scale in zip(sxx, syy, sxy, factors, strict=True)
    )
    return H1Estimate(
        tuple(map(float, pair.frequencies)),
        bins,
        count,
        (force_recording.unit, response_recording.unit),
    )


__all__ = ["H1Settings", "H1Bin", "H1Estimate", "estimate_complex_frf_h1"]
