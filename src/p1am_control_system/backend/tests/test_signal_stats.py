"""Unit tests for the pure signal noise/variability stats module.

Pure pydantic + stdlib, so no importorskip and no live PLC/DB is needed — these
must run in CI. Covers a hand-computed known sequence, the constant-signal
floor, arc-threshold behaviour across every metric, metric selection (incl. a
plain string metric), the empty/single-sample edge cases, and DbC validation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_stats import (  # noqa: E402
    NOISE_DEFAULT_THRESHOLD,
    NOISE_DEFAULT_WINDOW,
    NoiseMetric,
    NoiseStats,
    compute_noise,
)

# Hand-computed reference window:
#   mean = 5.0, sample std (ddof=1) = sqrt(32/7) ~= 2.1380899,
#   peak_to_peak = 7.0, population rms about mean = sqrt(4) = 2.0,
#   coeff_of_variation = std / 5.0 ~= 0.4276180.
KNOWN = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
KNOWN_STD = math.sqrt(32.0 / 7.0)


def test_known_sequence_mean_and_spread() -> None:
    stats = compute_noise(KNOWN)
    assert stats.sample_count == 8
    assert stats.mean == pytest.approx(5.0)
    assert stats.std == pytest.approx(KNOWN_STD)
    assert stats.peak_to_peak == pytest.approx(7.0)
    assert stats.rms_about_mean == pytest.approx(2.0)
    assert stats.coeff_of_variation == pytest.approx(KNOWN_STD / 5.0)


def test_known_sequence_returns_noisestats() -> None:
    stats = compute_noise(KNOWN)
    assert isinstance(stats, NoiseStats)
    # Default metric is STD with no threshold -> detection disabled.
    assert stats.metric is NoiseMetric.STD
    assert stats.metric_value == pytest.approx(KNOWN_STD)
    assert stats.threshold is None
    assert stats.arcing is False


def test_constant_signal_has_zero_noise() -> None:
    stats = compute_noise([3.3, 3.3, 3.3, 3.3])
    assert stats.mean == pytest.approx(3.3)
    assert stats.std == 0.0
    assert stats.peak_to_peak == 0.0
    assert stats.rms_about_mean == 0.0
    assert stats.coeff_of_variation == 0.0


def test_constant_signal_not_arcing_even_with_threshold() -> None:
    stats = compute_noise([3.3] * 10, metric=NoiseMetric.RMS, threshold=0.0)
    assert stats.metric_value == 0.0
    # 0.0 is not strictly greater than the 0.0 threshold.
    assert stats.arcing is False


def test_noisy_signal_over_threshold_arcs() -> None:
    stats = compute_noise(KNOWN, metric=NoiseMetric.STD, threshold=1.0)
    assert stats.threshold == pytest.approx(1.0)
    assert stats.metric_value == pytest.approx(KNOWN_STD)
    assert stats.arcing is True


def test_noisy_signal_under_threshold_does_not_arc() -> None:
    stats = compute_noise(KNOWN, metric=NoiseMetric.STD, threshold=10.0)
    assert stats.arcing is False


def test_threshold_none_disables_arc_detection() -> None:
    stats = compute_noise(KNOWN, metric=NoiseMetric.STD, threshold=None)
    assert stats.threshold is None
    assert stats.arcing is False


def test_threshold_equal_to_value_is_not_arcing() -> None:
    # arcing requires strictly greater-than, not >=.
    stats = compute_noise(KNOWN, metric=NoiseMetric.PEAK_TO_PEAK, threshold=7.0)
    assert stats.metric_value == pytest.approx(7.0)
    assert stats.arcing is False


def test_metric_std_selected() -> None:
    stats = compute_noise(KNOWN, metric=NoiseMetric.STD)
    assert stats.metric is NoiseMetric.STD
    assert stats.metric_value == pytest.approx(KNOWN_STD)


def test_metric_peak_to_peak_selected() -> None:
    stats = compute_noise(KNOWN, metric=NoiseMetric.PEAK_TO_PEAK)
    assert stats.metric is NoiseMetric.PEAK_TO_PEAK
    assert stats.metric_value == pytest.approx(7.0)


def test_metric_rms_selected() -> None:
    stats = compute_noise(KNOWN, metric=NoiseMetric.RMS)
    assert stats.metric is NoiseMetric.RMS
    assert stats.metric_value == pytest.approx(2.0)


def test_metric_cv_selected() -> None:
    stats = compute_noise(KNOWN, metric=NoiseMetric.CV)
    assert stats.metric is NoiseMetric.CV
    assert stats.metric_value == pytest.approx(KNOWN_STD / 5.0)


def test_metric_accepts_plain_string() -> None:
    stats = compute_noise(KNOWN, metric="peak_to_peak")
    assert stats.metric is NoiseMetric.PEAK_TO_PEAK
    assert stats.metric_value == pytest.approx(7.0)


def test_metric_selection_changes_metric_value() -> None:
    std_v = compute_noise(KNOWN, metric=NoiseMetric.STD).metric_value
    p2p_v = compute_noise(KNOWN, metric=NoiseMetric.PEAK_TO_PEAK).metric_value
    rms_v = compute_noise(KNOWN, metric=NoiseMetric.RMS).metric_value
    cv_v = compute_noise(KNOWN, metric=NoiseMetric.CV).metric_value
    assert std_v == pytest.approx(KNOWN_STD)
    assert p2p_v == pytest.approx(7.0)
    assert rms_v == pytest.approx(2.0)
    assert cv_v == pytest.approx(KNOWN_STD / 5.0)
    assert len({round(v, 6) for v in (std_v, p2p_v, rms_v, cv_v)}) == 4


def test_empty_window_is_all_zero_and_not_arcing() -> None:
    stats = compute_noise([], threshold=0.0)
    assert stats.sample_count == 0
    assert stats.mean == 0.0
    assert stats.std == 0.0
    assert stats.peak_to_peak == 0.0
    assert stats.rms_about_mean == 0.0
    assert stats.coeff_of_variation == 0.0
    assert stats.metric_value == 0.0
    assert stats.arcing is False


def test_single_sample_mean_only() -> None:
    stats = compute_noise([42.0], metric=NoiseMetric.CV, threshold=0.0)
    assert stats.sample_count == 1
    assert stats.mean == pytest.approx(42.0)
    assert stats.std == 0.0
    assert stats.peak_to_peak == 0.0
    assert stats.rms_about_mean == 0.0
    assert stats.coeff_of_variation == 0.0
    assert stats.arcing is False


def test_zero_mean_signal_has_zero_cv() -> None:
    # Symmetric about zero -> mean 0 -> CV guarded to 0.0 despite real spread.
    stats = compute_noise([-2.0, 2.0, -2.0, 2.0], metric=NoiseMetric.CV)
    assert stats.mean == pytest.approx(0.0)
    assert stats.std > 0.0
    assert stats.coeff_of_variation == 0.0


def test_integer_samples_accepted() -> None:
    stats = compute_noise([2, 4, 4, 4, 5, 5, 7, 9])
    assert stats.mean == pytest.approx(5.0)
    assert stats.std == pytest.approx(KNOWN_STD)


def test_tuple_samples_accepted() -> None:
    stats = compute_noise((2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0))
    assert stats.peak_to_peak == pytest.approx(7.0)


def test_module_constants() -> None:
    assert NOISE_DEFAULT_WINDOW == 100
    assert NOISE_DEFAULT_THRESHOLD == 0.0


def test_dbc_non_sequence_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        compute_noise(42)  # type: ignore[arg-type]


def test_dbc_string_sequence_rejected() -> None:
    with pytest.raises(TypeError):
        compute_noise("1234")  # type: ignore[arg-type]


def test_dbc_non_numeric_element_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        compute_noise([1.0, "x", 3.0])  # type: ignore[list-item]


def test_dbc_bool_element_rejected() -> None:
    with pytest.raises(TypeError):
        compute_noise([1.0, True, 3.0])


def test_dbc_bad_metric_string_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        compute_noise(KNOWN, metric="variance")


def test_dbc_bad_metric_type_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        compute_noise(KNOWN, metric=123)  # type: ignore[arg-type]


def test_dbc_non_finite_threshold_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        compute_noise(KNOWN, threshold=float("nan"))
    with pytest.raises(ValueError):
        compute_noise(KNOWN, threshold=float("inf"))


def test_dbc_wrong_type_threshold_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        compute_noise(KNOWN, threshold="0.5")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compute_noise(KNOWN, threshold=True)  # type: ignore[arg-type]
