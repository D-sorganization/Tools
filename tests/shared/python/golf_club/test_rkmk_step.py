"""Independent temporal and work controls for the shared local-chart step."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from shared.python.golf_club._rkmk_step import RkmkStepModel, rkmk_step


@dataclass(frozen=True)
class _State:
    position: np.ndarray
    twists: np.ndarray


@dataclass(frozen=True)
class _Response:
    acceleration: np.ndarray
    power: float


def _shift(state: _State, coordinates: np.ndarray, twists: np.ndarray) -> _State:
    # This reference problem uses only commuting translations.
    return _State(state.position + coordinates[:, :3], twists)


def _initial() -> _State:
    return _State(np.array([[1.0, 0, 0]]), np.zeros((1, 6)))


def _oscillator(state: _State, time_s: float) -> _Response:
    acceleration = np.zeros((1, 6))
    acceleration[:, :3] = -4 * state.position
    return _Response(acceleration, time_s**3)


def _model(
    evaluate: Callable[[_State, float], _Response] = _oscillator,
) -> RkmkStepModel[_State, _Response]:
    return RkmkStepModel(
        _shift,
        evaluate,
        lambda state: state.twists,
        lambda response: response.acceleration,
    )


def test_harmonic_motion_and_cubic_quadrature_refine_against_closed_form() -> None:
    # Chosen before numerical runs: finest absolute error 2e-7; fourth-order
    # ratios 14..18. The cubic power integral has exact quadrature 0.3**4/4.
    errors = []
    for steps in (3, 6, 12):
        state = _initial()
        response = _oscillator(state, 0)
        work = 0.0
        times = np.linspace(0, 0.3, steps + 1)
        for start, end in zip(times[:-1], times[1:], strict=True):
            cell = (float(start), float(start + (end - start) / 2), float(end))
            state, response, weighted = rkmk_step(_model(), state, response, cell)
            work += (end - start) * sum(
                item.power * weight for item, weight in weighted
            )
        expected = np.array([np.cos(0.6), -2 * np.sin(0.6)])
        observed = np.array([state.position[0, 0], state.twists[0, 0]])
        errors.append(float(np.linalg.norm(observed - expected)))
        assert work == pytest.approx(0.3**4 / 4, abs=2e-17)
    assert errors[-1] < 2e-7
    assert 14 < errors[0] / errors[1] < 18
    assert 14 < errors[1] / errors[2] < 18


def test_cached_start_and_owned_final_use_four_new_responses() -> None:
    calls = []

    def evaluate(state: _State, time_s: float) -> _Response:
        calls.append(time_s)
        return _oscillator(state, time_s)

    initial = _initial()
    position, twists = initial.position.copy(), initial.twists.copy()
    final, response, _ = rkmk_step(
        _model(evaluate), initial, _oscillator(initial, 0), (0, 0.05, 0.1)
    )
    assert calls == [0.05, 0.05, 0.1, 0.1]
    np.testing.assert_array_equal(initial.position, position)
    np.testing.assert_array_equal(initial.twists, twists)
    assert response.power == pytest.approx(0.1**3)
    assert not np.shares_memory(final.twists, initial.twists)


@pytest.mark.parametrize("bad", [(0, 0, 0.1), (0, 0.03, 0.1), (0.1, 0.05, 0)])
def test_invalid_time_cell_is_refused_before_evaluation(bad: tuple) -> None:
    def evaluate(state: _State, time_s: float) -> _Response:
        pytest.fail("invalid cell reached the constitutive response")

    initial = _initial()
    with pytest.raises(ValueError, match="time cell"):
        rkmk_step(_model(evaluate), initial, _oscillator(initial, 0), bad)


@pytest.mark.parametrize("bad", [np.full((1, 6), np.nan), np.zeros((1, 5))])
def test_invalid_acceleration_is_refused_without_a_partial_step(
    bad: np.ndarray,
) -> None:
    initial = _initial()
    with pytest.raises(ValueError, match="acceleration"):
        rkmk_step(_model(), initial, _Response(bad, 0), (0, 0.05, 0.1))


def test_constitutive_domain_failure_propagates_without_clipping() -> None:
    def evaluate(state: _State, time_s: float) -> _Response:
        raise ValueError("explicit constitutive domain")

    initial = _initial()
    with pytest.raises(ValueError, match="explicit constitutive domain"):
        rkmk_step(_model(evaluate), initial, _oscillator(initial, 0), (0, 0.05, 0.1))
