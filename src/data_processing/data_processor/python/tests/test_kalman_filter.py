"""Tests for data_processor.core.kalman_filter.

Covers the previously-untested public surface (issue #3691) and guards the
robustness fixes in sibling issues:

- #3692 covariance PSD/symmetry validation and singular innovation covariance
- #3693 shared Gaussian log-likelihood helper used by KF/EKF/UKF
- #3694 consistent NaN/missing-measurement handling across KF/EKF/UKF
- #3695 boundary/dimension validation that survives ``python -O``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from data_processor.core.kalman_filter import (
    ExtendedKalmanFilter,
    KalmanFilter,
    KalmanFilterConfig,
    KalmanFilterResult,
    KalmanFilterType,
    UnscentedKalmanFilter,
    _gaussian_log_likelihood,
    _is_psd_symmetric,
    apply_kalman_filter,
    estimate_kalman_params,
    kalman_smooth,
)

pytestmark = pytest.mark.unit


def _random_walk(n: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(n)) + rng.randn(n) * 0.3


class TestGaussianLogLikelihoodHelper:
    """#3693 — the shared log-likelihood block."""

    def test_matches_manual_formula(self) -> None:
        y = np.array([0.5, -0.2])
        S = np.array([[2.0, 0.3], [0.3, 1.0]])
        sign, logdet = np.linalg.slogdet(S)
        expected = -0.5 * (
            len(y) * np.log(2 * np.pi) + logdet + y @ np.linalg.inv(S) @ y
        )
        assert _gaussian_log_likelihood(y, S) == pytest.approx(expected)

    def test_singular_covariance_returns_neg_inf(self) -> None:
        # #3692 — a singular S must surface as -inf, not silent nan.
        y = np.array([1.0, 1.0])
        S = np.zeros((2, 2))
        assert _gaussian_log_likelihood(y, S) == float("-inf")

    def test_indefinite_covariance_returns_neg_inf(self) -> None:
        y = np.array([1.0])
        S = np.array([[-1.0]])
        assert _gaussian_log_likelihood(y, S) == float("-inf")


class TestPsdSymmetricHelper:
    """#3692 — covariance PSD/symmetry test."""

    def test_identity_is_psd(self) -> None:
        assert _is_psd_symmetric(np.eye(3))

    def test_asymmetric_rejected(self) -> None:
        assert not _is_psd_symmetric(np.array([[1.0, 2.0], [0.0, 1.0]]))

    def test_negative_definite_rejected(self) -> None:
        assert not _is_psd_symmetric(np.array([[-1.0, 0.0], [0.0, -1.0]]))

    def test_non_square_rejected(self) -> None:
        assert not _is_psd_symmetric(np.zeros((2, 3)))


class TestKalmanFilterConstruction:
    """#3695 — boundary validation at construction time."""

    def test_requires_config(self) -> None:
        with pytest.raises(ValueError):
            KalmanFilter(None)

    def test_rejects_non_psd_process_noise(self) -> None:
        cfg = KalmanFilterConfig(
            state_dim=1,
            measurement_dim=1,
            process_noise=np.array([[-1.0]]),
        )
        with pytest.raises(ValueError):
            KalmanFilter(cfg)

    def test_rejects_asymmetric_initial_covariance(self) -> None:
        cfg = KalmanFilterConfig(
            state_dim=2,
            measurement_dim=1,
            initial_covariance=np.array([[1.0, 2.0], [0.0, 1.0]]),
        )
        with pytest.raises(ValueError):
            KalmanFilter(cfg)


class TestKalmanFilterConfigKwargs:
    """#3745 — KalmanFilterConfig rejects unknown/typo'd keyword arguments."""

    def test_accepts_known_kwargs(self) -> None:
        cfg = KalmanFilterConfig(state_dim=3, measurement_dim=2)
        assert cfg.state_dim == 3
        assert cfg.measurement_dim == 2

    def test_obs_dim_alias_still_accepted(self) -> None:
        cfg = KalmanFilterConfig(state_dim=2, obs_dim=4)
        assert cfg.measurement_dim == 4

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {"meas_noise": 0.1},
            {"state_dimension": 3},
            {"transition": np.eye(2)},
        ],
    )
    def test_rejects_unknown_kwarg(self, bad_kwargs: dict) -> None:
        with pytest.raises(ValueError, match="Unknown KalmanFilterConfig argument"):
            KalmanFilterConfig(**bad_kwargs)


class TestNonlinearFilterStateDimValidation:
    """#3745 — EKF/UKF require a positive-integer state_dim."""

    @pytest.mark.parametrize("bad", [0, -1])
    def test_ekf_rejects_non_positive_state_dim(self, bad: int) -> None:
        with pytest.raises(ValueError, match="state_dim must be a positive integer"):
            ExtendedKalmanFilter(state_dim=bad)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_ukf_rejects_non_positive_state_dim(self, bad: int) -> None:
        with pytest.raises(ValueError, match="state_dim must be a positive integer"):
            UnscentedKalmanFilter(
                state_dim=bad,
                measurement_dim=1,
                f=lambda x, u: x,
                h=lambda x: x,
                Q=np.eye(1),
                R=np.eye(1),
            )


class TestKalmanFilterRun:
    """Standard Kalman filter happy-path + validation (#3691, #3695)."""

    def _config(self) -> KalmanFilterConfig:
        return KalmanFilterConfig(
            state_dim=1,
            measurement_dim=1,
            process_noise=0.01,
            measurement_noise=0.25,
        )

    def test_filter_returns_result(self) -> None:
        kf = KalmanFilter(self._config())
        sig = _random_walk(40)
        result = kf.filter(sig.reshape(-1, 1))
        assert isinstance(result, KalmanFilterResult)
        assert result.filtered_states.shape == (40, 1)
        assert np.isfinite(result.log_likelihood)

    def test_filter_rejects_none(self) -> None:
        kf = KalmanFilter(self._config())
        with pytest.raises(ValueError):
            kf.filter(None)

    def test_filter_rejects_empty(self) -> None:
        kf = KalmanFilter(self._config())
        with pytest.raises(ValueError):
            kf.filter(np.empty((0, 1)))

    def test_filter_rejects_wrong_measurement_dim(self) -> None:
        cfg = KalmanFilterConfig(state_dim=2, measurement_dim=2)
        kf = KalmanFilter(cfg)
        # 3 columns where 2 expected -- and not transposable to 2.
        with pytest.raises(ValueError):
            kf.filter(np.zeros((5, 3)))

    def test_filter_rejects_mismatched_control_inputs(self) -> None:
        cfg = KalmanFilterConfig(
            state_dim=1, measurement_dim=1, control_matrix=np.array([[1.0]])
        )
        kf = KalmanFilter(cfg)
        with pytest.raises(ValueError):
            kf.filter(np.zeros((5, 1)), control_inputs=np.zeros((3, 1)))

    def test_nan_measurement_marks_innovation(self) -> None:
        # #3694 — missing measurement yields NaN innovation, prior carried.
        kf = KalmanFilter(self._config())
        sig = _random_walk(20).reshape(-1, 1)
        sig[5] = np.nan
        result = kf.filter(sig)
        assert np.isnan(result.innovations[5, 0])
        assert np.all(np.isfinite(result.filtered_states))

    def test_smoother_runs(self) -> None:
        kf = KalmanFilter(self._config())
        sig = _random_walk(30).reshape(-1, 1)
        result = kf.filter(sig)
        smoothed = kf.smooth(result)
        assert smoothed.smoothed_states is not None
        assert smoothed.smoothed_states.shape == (30, 1)

    def test_smoother_rejects_none(self) -> None:
        kf = KalmanFilter(self._config())
        with pytest.raises(ValueError):
            kf.smooth(None)


class TestExtendedKalmanFilter:
    """#3691/#3694 — EKF public surface + NaN handling."""

    def _ekf(self) -> ExtendedKalmanFilter:
        return ExtendedKalmanFilter(
            state_dim=1,
            measurement_dim=1,
            f=lambda x, u=None: x,
            h=lambda x: x,
            Q=np.array([[0.01]]),
            R=np.array([[0.25]]),
        )

    def test_filter_runs(self) -> None:
        ekf = self._ekf()
        result = ekf.filter(_random_walk(30).reshape(-1, 1))
        assert result.filtered_states.shape == (30, 1)
        assert np.isfinite(result.log_likelihood)

    def test_requires_functions(self) -> None:
        ekf = ExtendedKalmanFilter(state_dim=1, measurement_dim=1)
        with pytest.raises(ValueError):
            ekf.filter(np.zeros((5, 1)))

    def test_nan_measurement_marks_innovation(self) -> None:
        # #3694 — EKF must now mark NaN like the standard KF (was zeros before).
        ekf = self._ekf()
        sig = _random_walk(20).reshape(-1, 1)
        sig[7] = np.nan
        result = ekf.filter(sig)
        assert np.isnan(result.innovations[7, 0])
        assert np.isnan(result.innovation_covariances[7, 0, 0])


class TestUnscentedKalmanFilter:
    """#3691/#3694 — UKF public surface + NaN handling."""

    def _ukf(self) -> UnscentedKalmanFilter:
        return UnscentedKalmanFilter(
            state_dim=1,
            measurement_dim=1,
            f=lambda x, u=None: x,
            h=lambda x: x,
            Q=np.array([[0.01]]),
            R=np.array([[0.25]]),
        )

    def test_filter_runs(self) -> None:
        ukf = self._ukf()
        result = ukf.filter(_random_walk(30).reshape(-1, 1))
        assert result.filtered_states.shape == (30, 1)
        assert np.isfinite(result.log_likelihood)

    def test_nan_measurement_marks_innovation(self) -> None:
        ukf = self._ukf()
        sig = _random_walk(20).reshape(-1, 1)
        sig[9] = np.nan
        result = ukf.filter(sig)
        assert np.isnan(result.innovations[9, 0])
        assert np.isnan(result.innovation_covariances[9, 0, 0])


class TestConvenienceFunctions:
    """#3691 — apply_kalman_filter, kalman_smooth, estimate_kalman_params."""

    def test_apply_kalman_filter_adds_columns(self) -> None:
        df = pd.DataFrame({"s": _random_walk(40)})
        out = apply_kalman_filter(df, "s")
        for suffix in (
            "_kf_filtered",
            "_kf_smoothed",
            "_kf_std",
            "_kf_lower",
            "_kf_upper",
        ):
            assert f"s{suffix}" in out.columns

    def test_apply_kalman_filter_rejects_none(self) -> None:
        with pytest.raises(ValueError):
            apply_kalman_filter(None, "s")

    def test_kalman_smooth_smooths(self) -> None:
        sig = _random_walk(50)
        smoothed = kalman_smooth(sig)
        assert smoothed.shape == sig.shape
        # Smoothing should reduce variance of the first difference.
        assert np.var(np.diff(smoothed)) < np.var(np.diff(sig))

    def test_kalman_smooth_rejects_none(self) -> None:
        with pytest.raises(ValueError):
            kalman_smooth(None)

    def test_estimate_kalman_params_positive(self) -> None:
        q, r = estimate_kalman_params(_random_walk(100))
        assert q > 0
        assert r > 0

    def test_estimate_kalman_params_short_signal_defaults(self) -> None:
        q, r = estimate_kalman_params(np.array([1.0, 2.0, 3.0]))
        assert (q, r) == (1.0, 1.0)

    def test_estimate_kalman_params_handles_nan(self) -> None:
        sig = _random_walk(100)
        sig[::7] = np.nan
        q, r = estimate_kalman_params(sig)
        assert np.isfinite(q) and np.isfinite(r)

    def test_estimate_kalman_params_rejects_none(self) -> None:
        with pytest.raises(ValueError):
            estimate_kalman_params(None)


class TestKalmanFilterType:
    def test_enum_values(self) -> None:
        assert KalmanFilterType.STANDARD.value == "standard"
        assert KalmanFilterType.EXTENDED.value == "extended"
        assert KalmanFilterType.UNSCENTED.value == "unscented"
