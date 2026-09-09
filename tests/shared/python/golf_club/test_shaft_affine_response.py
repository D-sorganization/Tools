"""Independent forced-oscillator and strict-domain controls for Tools #5072."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from shared.python.golf_club import _shaft_affine_response as affine
from shared.python.golf_club._shaft_autonomous_decay import (
    DecayControls,
    DecayEnvelope,
    assess_autonomous_decay,
)
from shared.python.golf_club._shaft_damped_spectrum import DampedPencil
from shared.python.golf_club._shaft_spectrum import SpectrumScales


def _scales(time: float = 1.0) -> SpectrumScales:
    return SpectrumScales(1.0, time, 1e-12, 1e-9)


def _controls(error: float = 0.0) -> affine.ForcedResponseControls:
    return affine.ForcedResponseControls(DecayControls(1e-10, 1e-10, 0.0), error)


def _pencil() -> DampedPencil:
    return DampedPencil([[1.0]], [[0.0]], [[2.0]], [[1.0]])


@pytest.mark.parametrize("time_scale", [0.25, 1.0, 2.0])
def test_residual_sign_and_time_scale_bound_analytic_forced_oscillator(
    time_scale: float,
) -> None:
    # q''+2q'+q=3, q(0)=q'(0)=0; no solver or eigenvector oracle.
    result = affine.assess_affine_response(
        _pencil(), [-3.0], _scales(time_scale), _controls()
    )
    assert result.status == "numerically_supported"
    assert result.scope == "constant_affine_regular_linear_ode"
    assert result.residual == (-3.0,)
    assert result.input_per_s == (0.0, 3.0 * time_scale)
    envelope = result.envelope
    assert envelope is not None
    for time in np.linspace(0, 30, 151):
        q = 3 * (1 - (1 + time) * np.exp(-time))
        v = 3 * time * np.exp(-time)
        bound = envelope.bound(time, 0.0)
        assert bound.initial_state_term == 0.0
        assert np.hypot(q, time_scale * v) <= bound.total + 1e-14
    assert envelope.bound(30.0, 0.0).input_term > 2.9


def test_zero_residual_preserves_the_existing_homogeneous_envelope() -> None:
    controls = _controls()
    free = assess_autonomous_decay(_pencil(), _scales(), controls.decay)
    forced = affine.assess_affine_response(_pencil(), [0.0], _scales(), controls)
    assert forced.homogeneous == free
    assert forced.envelope is not None and free.envelope is not None
    value = forced.envelope.bound(0.7, 2.0)
    expected = (
        2 * free.envelope.norm_prefactor * np.exp(-0.7 * free.envelope.decay_rate_s_inv)
    )
    assert value.initial_state_term == pytest.approx(expected)
    assert value.input_term == 0.0
    assert value.total == value.initial_state_term


def test_bounded_input_error_survives_an_exactly_balanced_nominal_model() -> None:
    result = affine.assess_affine_response(_pencil(), [0.0], _scales(), _controls(0.4))
    assert result.envelope is not None
    assert result.input_per_s == (0.0, 0.0)
    assert result.envelope.input_norm_bound_s_inv == 0.4
    assert result.envelope.bound(1.0, 0.0).input_term > 0.0


def test_initial_and_forcing_terms_use_the_declared_operator_error() -> None:
    exact = affine.assess_affine_response(_pencil(), [-1.0], _scales(), _controls())
    robust_controls = replace(_controls(), decay=DecayControls(1e-10, 1e-10, 0.01))
    robust = affine.assess_affine_response(
        _pencil(), [-1.0], _scales(), robust_controls
    )
    assert exact.envelope is not None and robust.envelope is not None
    nominal = exact.envelope.bound(2.0, 1.0)
    uncertain = robust.envelope.bound(2.0, 1.0)
    assert uncertain.initial_state_term > nominal.initial_state_term
    assert uncertain.input_term > nominal.input_term


def test_neutral_model_does_not_receive_a_forced_response_bound() -> None:
    neutral = DampedPencil([[1.0]], [[0.0]], [[0.0]], [[0.0]])
    result = affine.assess_affine_response(neutral, [-1.0], _scales(), _controls())
    assert result.status == "not_established"
    assert result.envelope is None


def test_residual_is_copied_and_tiny_nonzero_forcing_is_retained() -> None:
    residual = np.array([1e-200])
    result = affine.assess_affine_response(_pencil(), residual, _scales(), _controls())
    residual[:] = 0
    assert result.residual == (1e-200,)
    assert result.envelope is not None
    assert result.envelope.input_norm_bound_s_inv == 1e-200
    assert result.envelope.bound(1.0, 0.0).input_term > 0
    with pytest.raises(FrozenInstanceError):
        result.residual = (0.0,)


def test_convolution_retains_the_small_time_limit_without_cancellation() -> None:
    envelope = affine.ForcedEnvelope(DecayEnvelope(2.0, 1e-200), 3.0)
    value = envelope.bound(1e-200, 0.0)
    assert value.input_term == pytest.approx(6e-200, abs=0, rel=1e-14)
    zero = envelope.bound(0.0, 2.0)
    assert zero.total == 4.0 and zero.input_term == 0.0


@pytest.mark.parametrize("bad", [True, "1", 1j, float("nan"), float("inf"), -1.0])
def test_forcing_error_rejects_invalid_values(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _controls(bad)


@pytest.mark.parametrize("bad", [[True], ["1"], [1j], [float("nan")], [[1.0]], []])
def test_residual_contracts_are_checked_before_assessment(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        affine.assess_affine_response(_pencil(), bad, _scales(), _controls())


@pytest.mark.parametrize("bad", [True, "1", 1j, float("nan"), float("inf"), -1.0])
def test_bound_requires_finite_nonnegative_real_time_and_initial_norm(
    bad: object,
) -> None:
    envelope = affine.ForcedEnvelope(DecayEnvelope(2.0, 1.0), 3.0)
    with pytest.raises((TypeError, ValueError)):
        envelope.bound(bad, 0.0)
    with pytest.raises((TypeError, ValueError)):
        envelope.bound(1.0, bad)


@pytest.mark.parametrize("prefactor,rate", [(0.5, 1), (1, 0), (1, -1), (np.inf, 1)])
def test_invalid_decay_envelopes_cannot_produce_forced_bounds(
    prefactor: float,
    rate: float,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        affine.ForcedEnvelope(DecayEnvelope(prefactor, rate), 1.0)


def test_nonrepresentable_response_is_refused() -> None:
    envelope = affine.ForcedEnvelope(DecayEnvelope(2.0, 1.0), 1e308)
    with pytest.raises(ValueError):
        envelope.bound(10.0, 1e308)


def test_large_input_has_zero_contribution_at_zero_time() -> None:
    envelope = affine.ForcedEnvelope(DecayEnvelope(2.0, 1.0), 1e308)
    assert envelope.bound(0.0, 0.0).total == 0.0
    assert envelope.bound(1e-300, 0.0).total == pytest.approx(2e8)


def test_nonzero_residual_cannot_underflow_to_a_zero_input_claim() -> None:
    pencil = DampedPencil([[1e100]], [[0.0]], [[2e100]], [[1e100]])
    result = affine.assess_affine_response(pencil, [1e-300], _scales(), _controls())
    assert result.status == "not_established"
    assert result.envelope is None
    assert "underflow" in result.reason


@pytest.mark.parametrize(
    "forcing,time,initial", [(0.0, 1000.0, 1.0), (1e-300, 1e-100, 0.0)]
)
def test_unrepresentable_positive_bound_terms_are_not_reported_as_zero(
    forcing: float, time: float, initial: float
) -> None:
    envelope = affine.ForcedEnvelope(DecayEnvelope(1.0, 1.0), forcing)
    with pytest.raises(ValueError, match="underflow"):
        envelope.bound(time, initial)
