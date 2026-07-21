"""Enumerations for the Data Explorer analysis suite.

Split into their own module (no business logic, no heavy imports) so the
request/response models, the numeric kernels, the service and the router can all
share one source of truth without circular imports. Mirrors the established
``*_models``/``*_enums`` split used by the temperature and performance
subsystems — and, importantly, keeps these ``StrEnum`` members *imported* (hence
``Any``) wherever they are used as defaults, side-stepping the
``mypy --follow-imports=skip`` quirk that flags ``StrEnum`` members as ``str``.
"""

from __future__ import annotations

from shared.python.compatibility import StrEnum


class FilterType(StrEnum):
    """A signal-conditioning filter that maps one series to a same-length series.

    Frequency-domain members (``FFT_*``) replace IIR Butterworth from the desktop
    tool: they are zero-phase and pure-``numpy``, so the standalone Pi backend
    needs no ``scipy`` dependency.
    """

    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    MEDIAN = "median"
    GAUSSIAN = "gaussian"
    SAVGOL = "savgol"
    HAMPEL = "hampel"
    ZSCORE = "zscore"
    FFT_LOWPASS = "fft_lowpass"
    FFT_HIGHPASS = "fft_highpass"
    FFT_BANDPASS = "fft_bandpass"
    INTEGRATE = "integrate"
    DIFFERENTIATE = "differentiate"


class AggMethod(StrEnum):
    """Aggregation applied to each bin when resampling onto a coarser grid."""

    MEAN = "mean"
    MEDIAN = "median"
    FIRST = "first"
    LAST = "last"
    MIN = "min"
    MAX = "max"
    SUM = "sum"


class CorrelationMethod(StrEnum):
    """Correlation coefficient family for the correlation matrix."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"


class SpectrumMethod(StrEnum):
    """Power-spectrum estimator."""

    FFT = "fft"
    WELCH = "welch"


class WindowKind(StrEnum):
    """Tapering window applied before the FFT to reduce spectral leakage."""

    NONE = "none"
    HANNING = "hanning"
    HAMMING = "hamming"
    BLACKMAN = "blackman"


class TrendlineKind(StrEnum):
    """Curve family fit to an (x, y) scatter for the trendline overlay."""

    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    POWER = "power"


class ExportFormat(StrEnum):
    """Serialization format for a processed-dataset export."""

    CSV = "csv"
    JSON = "json"


class SourceKind(StrEnum):
    """Where a dataset's raw columns come from."""

    HISTORIAN = "historian"
    INLINE = "inline"
