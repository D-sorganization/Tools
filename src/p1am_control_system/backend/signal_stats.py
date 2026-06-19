"""Pure noise/variability statistics for power-supply feedback signals.

DC-arc "arcing" shows up as a sudden rise in the AC noise riding on an
otherwise steady DC current/voltage feedback signal. This module quantifies
that noise from a short window of samples so the integration layer can raise an
operator-facing arc warning when a chosen metric crosses a tunable threshold.

Design notes:
    - DbC: every public entry point validates its inputs (``TypeError`` for
      wrong types, ``ValueError`` for out-of-range) and documents its
      pre/postconditions.
    - LOD: this module is pure. It imports nothing from the DB, PLC, or FastAPI
      layers and pulls only from the stdlib (``statistics``/``math``) — no
      numpy — so it stays light enough to run at 10 Hz on a Raspberry Pi and is
      unit-testable in isolation.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, Field

from shared.python.compatibility import StrEnum

NOISE_DEFAULT_WINDOW = 100
NOISE_DEFAULT_THRESHOLD = 0.0


class NoiseMetric(StrEnum, Enum):
    """Which scalar the arc threshold is evaluated against."""

    STD = "std"
    PEAK_TO_PEAK = "peak_to_peak"
    RMS = "rms"
    CV = "cv"


class NoiseStats(BaseModel):
    """Noise/variability summary of one window of feedback samples."""

    sample_count: int = Field(
        ge=0, description="Number of samples the stats were computed from."
    )
    mean: float = Field(description="Arithmetic mean (the DC level) of the window.")
    std: float = Field(
        ge=0.0,
        description=(
            "Sample standard deviation (ddof=1); 0.0 when fewer than 2 samples."
        ),
    )
    peak_to_peak: float = Field(
        ge=0.0, description="max - min of the window; 0.0 when empty."
    )
    rms_about_mean: float = Field(
        ge=0.0,
        description=(
            "Population deviation RMS sqrt(mean((x-mean)^2)) — the AC content "
            "of the signal; 0.0 when fewer than 2 samples."
        ),
    )
    coeff_of_variation: float = Field(
        ge=0.0,
        description=(
            "std / |mean|, the AC/DC noise ratio; 0.0 when mean==0 or fewer "
            "than 2 samples."
        ),
    )
    metric: NoiseMetric = Field(
        description="Which metric the arc threshold was evaluated against."
    )
    metric_value: float = Field(
        ge=0.0, description="Value of the selected metric for this window."
    )
    threshold: float | None = Field(
        default=None,
        description="Arc threshold in effect (None disables arc detection).",
    )
    arcing: bool = Field(
        description="True iff threshold is not None and metric_value > threshold."
    )


def _is_real_number(value: object) -> bool:
    """True for an int/float that is NOT a bool (bool is not numeric here)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compute_noise(
    samples: Sequence[int | float],
    *,
    metric: NoiseMetric | str = NoiseMetric.STD,
    threshold: float | None = None,
) -> NoiseStats:
    """Quantify the noise of a window of feedback samples for arc detection.

    Args:
        samples: A sequence (list/tuple) of int|float feedback readings.
        metric: Which metric to evaluate the arc threshold against. Accepts a
            ``NoiseMetric`` or its string value (e.g. ``"std"``).
        threshold: Arc threshold for the chosen metric. ``None`` disables arc
            detection (``arcing`` is always False). Otherwise must be a finite
            number.

    Returns:
        NoiseStats with the full set of noise metrics, the selected metric and
        its value, the threshold in effect, and the resulting ``arcing`` flag.

    Raises:
        TypeError: If ``samples`` is not a sequence (a str is rejected), if any
            element is non-numeric (``bool`` counts as non-numeric), if
            ``metric`` is not a ``NoiseMetric``/``str``, or if ``threshold`` is
            neither ``None`` nor a real number.
        ValueError: If ``metric`` is not a valid ``NoiseMetric`` value, or if
            ``threshold`` is given but not finite (NaN/inf).

    Preconditions:
        - ``samples`` is a list/tuple of finite-or-not real numbers.
        - ``metric`` resolves to a member of ``NoiseMetric``.
        - ``threshold`` is ``None`` or a finite real number.

    Postconditions:
        - ``sample_count == len(samples)``.
        - Empty input -> all numeric stats 0.0 and ``arcing`` False.
        - A single sample -> ``mean`` is that value, every spread stat is 0.0.
        - ``std`` uses ddof=1 (statistics.stdev) for n>=2, else 0.0.
        - ``rms_about_mean`` is the population deviation RMS.
        - ``coeff_of_variation`` is ``std / abs(mean)`` (0.0 if mean==0/n<2).
        - ``metric_value`` equals the selected metric's value.
        - ``arcing`` is ``threshold is not None and metric_value > threshold``.
    """
    # Reject str/bytes up front: they are sequences but not sample windows.
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TypeError(
            f"samples must be a list/tuple sequence, got {type(samples).__name__}"
        )
    if not isinstance(metric, (NoiseMetric, str)):
        raise TypeError(
            f"metric must be a NoiseMetric or str, got {type(metric).__name__}"
        )
    try:
        metric = NoiseMetric(metric)
    except ValueError as exc:
        valid = ", ".join(m.value for m in NoiseMetric)
        raise ValueError(f"metric must be one of [{valid}], got {metric!r}") from exc

    if threshold is not None:
        if not _is_real_number(threshold):
            raise TypeError(
                "threshold must be None or a real number, got "
                f"{type(threshold).__name__}"
            )
        if not math.isfinite(threshold):
            raise ValueError(f"threshold must be finite, got {threshold!r}")

    values: list[float] = []
    for index, element in enumerate(samples):
        if not _is_real_number(element):
            raise TypeError(
                f"samples[{index}] must be int|float, got {type(element).__name__}"
            )
        values.append(float(element))

    sample_count = len(values)
    if sample_count == 0:
        mean = 0.0
        std = 0.0
        peak_to_peak = 0.0
        rms_about_mean = 0.0
        coeff_of_variation = 0.0
    else:
        mean = statistics.fmean(values)
        peak_to_peak = max(values) - min(values)
        if sample_count >= 2:
            std = statistics.stdev(values)
            rms_about_mean = math.sqrt(
                statistics.fmean([(x - mean) ** 2 for x in values])
            )
            coeff_of_variation = std / abs(mean) if mean != 0.0 else 0.0
        else:
            std = 0.0
            rms_about_mean = 0.0
            coeff_of_variation = 0.0

    metric_values: dict[NoiseMetric, float] = {
        NoiseMetric.STD: std,
        NoiseMetric.PEAK_TO_PEAK: peak_to_peak,
        NoiseMetric.RMS: rms_about_mean,
        NoiseMetric.CV: coeff_of_variation,
    }
    metric_value = metric_values[metric]
    arcing = threshold is not None and metric_value > threshold

    return NoiseStats(
        sample_count=sample_count,
        mean=mean,
        std=std,
        peak_to_peak=peak_to_peak,
        rms_about_mean=rms_about_mean,
        coeff_of_variation=coeff_of_variation,
        metric=metric,
        metric_value=metric_value,
        threshold=threshold,
        arcing=arcing,
    )
