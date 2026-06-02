import numpy as np
import plot_engine.trendline as trendline_module
import pytest
from plot_engine.trendline import (
    _linear,
    _polynomial,
    _r_squared,
    compute_trendline,
)


def test_linear_trendline_filters_nan_pairs_and_predicts_requested_points() -> None:
    result = compute_trendline(
        np.array([0.0, 1.0, 2.0, np.nan]),
        np.array([-1.0, 1.0, 3.0, 99.0]),
        trend_type="linear",
        n_points=5,
    )

    assert result.trend_type == "linear"
    assert result.coefficients == pytest.approx([2.0, -1.0])
    assert result.equation == "y = 2x - 1"
    assert result.r_squared == pytest.approx(1.0)
    assert np.array_equal(result.x_pred, np.linspace(0.0, 2.0, 5))
    assert result.y_pred == pytest.approx(np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))


def test_compute_trendline_rejects_insufficient_valid_points() -> None:
    with pytest.raises(
        ValueError,
        match="At least 2 valid data points required for trendline",
    ):
        compute_trendline(
            np.array([1.0, np.nan]),
            np.array([2.0, 3.0]),
        )


def test_compute_trendline_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown trend type: logarithmic"):
        compute_trendline(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            trend_type="logarithmic",  # type: ignore[arg-type]
        )


def test_polynomial_trendline_caps_degree_to_available_data() -> None:
    result = compute_trendline(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 4.0, 9.0]),
        trend_type="polynomial",
        degree=10,
        n_points=3,
    )

    assert result.trend_type == "polynomial"
    assert len(result.coefficients) == 3
    assert result.equation.startswith("y =")
    assert "x^2" in result.equation
    assert result.r_squared == pytest.approx(1.0)
    assert result.y_pred == pytest.approx(np.array([1.0, 4.0, 9.0]))


def test_polynomial_trendline_reports_zero_equation_for_zero_coefficients() -> None:
    result = compute_trendline(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.0, 0.0]),
        trend_type="polynomial",
        degree=2,
    )

    assert result.equation == "y = 0"
    assert result.r_squared == 0.0


def test_exponential_trendline_fits_positive_values() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 3.0 * np.exp(0.5 * x)

    result = compute_trendline(x, y, trend_type="exponential", n_points=4)

    assert result.trend_type == "exponential"
    assert result.coefficients == pytest.approx([3.0, 0.5], rel=1e-5)
    assert result.equation.startswith("y = 3")
    assert "exp(0.5x)" in result.equation
    assert result.r_squared == pytest.approx(1.0)
    assert result.y_pred == pytest.approx(y, rel=1e-5)


def test_exponential_trendline_rejects_too_few_positive_y_values() -> None:
    with pytest.raises(
        ValueError,
        match="Exponential fit requires at least 2 positive y values",
    ):
        compute_trendline(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, -1.0, 2.0]),
            trend_type="exponential",
        )


def test_exponential_trendline_falls_back_to_log_fit_on_curve_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_runtime_error(*_args: object, **_kwargs: object) -> tuple[object, object]:
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(trendline_module, "curve_fit", raise_runtime_error)
    x = np.array([0.0, 1.0, 2.0])
    y = 2.0 * np.exp(0.25 * x)

    result = compute_trendline(x, y, trend_type="exponential", n_points=3)

    assert result.coefficients == pytest.approx([2.0, 0.25])
    assert result.y_pred == pytest.approx(y)


def test_power_trendline_filters_to_positive_x_and_y_values() -> None:
    result = compute_trendline(
        np.array([-1.0, 0.0, 1.0, 2.0, 4.0]),
        np.array([10.0, 10.0, 2.0, 16.0, 128.0]),
        trend_type="power",
        n_points=6,
    )

    assert result.trend_type == "power"
    assert result.coefficients == pytest.approx([2.0, 3.0])
    assert result.equation == "y = 2 * x^3"
    assert result.r_squared == pytest.approx(1.0)
    assert np.all(result.x_pred > 0)
    assert result.y_pred == pytest.approx(2.0 * result.x_pred**3)


def test_power_trendline_rejects_too_few_positive_pairs() -> None:
    with pytest.raises(
        ValueError,
        match="Power fit requires at least 2 positive x and y values",
    ):
        compute_trendline(
            np.array([-1.0, 0.0, 2.0]),
            np.array([1.0, 2.0, -3.0]),
            trend_type="power",
        )


def test_r_squared_validates_y_and_handles_constant_series() -> None:
    with pytest.raises(ValueError, match="y must be provided"):
        _r_squared(None, np.array([1.0]))  # type: ignore[arg-type]

    assert _r_squared(np.array([2.0, 2.0]), np.array([2.0, 2.0])) == 0.0


def test_private_fit_helpers_validate_x() -> None:
    with pytest.raises(ValueError, match="x must be provided"):
        _linear(None, np.array([1.0, 2.0]), np.array([1.0, 2.0]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="x must be provided"):
        _polynomial(
            None,  # type: ignore[arg-type]
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            degree=1,
        )
