"""Independent Lyapunov, transient and refusal controls for Tools #5072."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from shared.python.golf_club import _shaft_autonomous_decay as decay
from shared.python.golf_club._shaft_damped_spectrum import DampedPencil

from .test_shaft_spectrum import _scales


def _controls() -> decay.DecayControls:
    return decay.DecayControls(1e-10, 1e-10, 0.0)


def _scalar(damping: float = 1.0) -> DampedPencil:
    return DampedPencil([[1]], [[0]], [[damping]], [[1]])


@pytest.mark.parametrize(
    ("damping", "expected"),
    [(1.0, [[1.5, 0.5], [0.5, 1.0]]), (2.0, [[1.5, 0.5], [0.5, 0.5]])],
)
def test_scalar_storage_matches_independent_linear_equations(
    damping: float,
    expected: list,
) -> None:
    result = decay.assess_autonomous_decay(
        _scalar(damping), replace(_scales(), time_s=1), _controls()
    )
    assert result.status == "numerically_supported"
    assert result.scope == "constant_homogeneous_regular_linear_ode"
    evidence, envelope = result.evidence, result.envelope
    assert evidence is not None and envelope is not None
    np.testing.assert_allclose(evidence.storage_matrix, expected, atol=1e-13)
    np.testing.assert_allclose(evidence.dissipation_matrix, np.eye(2), atol=1e-13)
    low, high = np.linalg.eigvalsh(expected)
    assert envelope.norm_prefactor == pytest.approx(np.sqrt(high / low))
    assert envelope.decay_rate_s_inv == pytest.approx(1 / (2 * high))
    assert evidence.relative_residual < 1e-13


@pytest.mark.parametrize("time", [0.1, 2.0])
def test_physical_time_conversion_of_same_dimensionless_generator(time: float) -> None:
    pencil = DampedPencil([[1]], [[0]], [[1 / time]], [[1 / time**2]])
    result = decay.assess_autonomous_decay(
        pencil, replace(_scales(), time_s=time), _controls()
    )
    assert result.envelope is not None and result.evidence is not None
    np.testing.assert_allclose(result.evidence.storage_matrix, [[1.5, 0.5], [0.5, 1]])
    high = (2.5 + np.sqrt(1.25)) / 2
    assert result.envelope.decay_rate_s_inv == pytest.approx(1 / (2 * high * time))


def test_nonnormal_circulatory_model_bounds_independent_transient_growth() -> None:
    pencil = DampedPencil(np.eye(2), np.zeros((2, 2)), 2 * np.eye(2), [[1, 4], [0, 1]])
    result = decay.assess_autonomous_decay(
        pencil, replace(_scales(), time_s=1), _controls()
    )
    assert result.envelope is not None
    t = np.linspace(0, 20, 401)
    # q2=e^-t(1+t), q1=e^-t(-2t²-2t³/3); initial [q1,q2,v1,v2]=[0,1,0,0].
    p = -2 * t**2 - 2 * t**3 / 3
    state = np.exp(-t) * np.array([p, 1 + t, -4 * t - 2 * t**2 - p, -t])
    norm = np.linalg.norm(state, axis=0)
    envelope = result.envelope
    assert max(norm) > 1.5
    assert np.all(
        norm <= envelope.norm_prefactor * np.exp(-envelope.decay_rate_s_inv * t)
    )


@pytest.mark.parametrize("stiffness", [0.0, 1.0, -1.0])
def test_absence_of_decay_is_not_reported_as_proven_instability(
    stiffness: float,
) -> None:
    pencil = DampedPencil([[1]], [[0]], [[0]], [[stiffness]])
    result = decay.assess_autonomous_decay(pencil, _scales(), _controls())
    assert result.status == "not_established"
    assert result.envelope is None


def test_passive_damping_does_not_certify_the_destabilized_gyroscope() -> None:
    pencil = DampedPencil(np.eye(2), [[0, -3], [3, 0]], 0.1 * np.eye(2), -np.eye(2))
    result = decay.assess_autonomous_decay(pencil, _scales(), _controls())
    assert result.status == "not_established"
    assert result.envelope is None


@pytest.mark.parametrize("coupling, supported", [(0.0, False), (-1.0, True)])
def test_semidefinite_loss_requires_coupling_to_every_undamped_mode(
    coupling: float,
    supported: bool,
) -> None:
    pencil = DampedPencil(
        np.eye(2), np.zeros((2, 2)), np.diag([1, 0]), [[2, coupling], [coupling, 2]]
    )
    result = decay.assess_autonomous_decay(
        pencil, replace(_scales(), time_s=1), _controls()
    )
    assert (result.envelope is not None) is supported


def test_declared_operator_error_reduces_decay_and_can_prevent_qualification() -> None:
    scales = replace(_scales(), time_s=1)
    exact = decay.assess_autonomous_decay(_scalar(), scales, _controls())
    robust = decay.assess_autonomous_decay(
        _scalar(), scales, replace(_controls(), generator_error_bound=0.01)
    )
    refused = decay.assess_autonomous_decay(
        _scalar(), scales, replace(_controls(), generator_error_bound=1)
    )
    assert exact.envelope is not None and robust.envelope is not None
    assert robust.evidence is not None
    assert robust.envelope.decay_rate_s_inv < exact.envelope.decay_rate_s_inv
    high = np.linalg.eigvalsh([[1.5, 0.5], [0.5, 1]])[-1]
    assert robust.evidence.robust_dissipation_margin == pytest.approx(
        1 - 2 * high * 0.01
    )
    assert refused.envelope is None


def test_unresolved_positive_storage_is_not_promoted() -> None:
    controls = replace(_controls(), definiteness_rcond_floor=0.5)
    result = decay.assess_autonomous_decay(
        _scalar(), replace(_scales(), time_s=1), controls
    )
    assert result.envelope is None


def test_unresolved_positive_robust_margin_is_not_promoted() -> None:
    high = np.linalg.eigvalsh([[1.5, 0.5], [0.5, 1]])[-1]
    controls = replace(_controls(), generator_error_bound=(1 - 1e-12) / (2 * high))
    result = decay.assess_autonomous_decay(
        _scalar(), replace(_scales(), time_s=1), controls
    )
    assert result.evidence is not None
    assert 0 < result.evidence.robust_dissipation_margin < 1e-10
    assert result.envelope is None


@pytest.mark.parametrize(
    "bad", [np.eye(2), np.full((2, 2), np.nan), np.zeros((1, 1)), np.eye(2) * 1j]
)
def test_inaccurate_or_invalid_solver_output_never_produces_an_envelope(
    bad: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decay, "solve_continuous_lyapunov", lambda a, q: bad)
    result = decay.assess_autonomous_decay(_scalar(), _scales(), _controls())
    assert result.envelope is None
    assert result.status == "not_established"


def test_results_do_not_alias_inputs_or_mutable_solver_arrays() -> None:
    source = np.array([[1.0]])
    pencil = DampedPencil(source, [[0]], [[1]], [[1]])
    result = decay.assess_autonomous_decay(
        pencil, replace(_scales(), time_s=1), _controls()
    )
    source[0, 0] = 99
    assert result.evidence is not None
    assert result.evidence.storage_matrix[0][0] == pytest.approx(1.5)
    with pytest.raises(TypeError):
        result.evidence.storage_matrix[0][0] = 99
    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"


@pytest.mark.parametrize("field", ["residual_tolerance", "definiteness_rcond_floor"])
@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, np.inf, np.nan])
def test_control_domains_are_enforced(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(_controls(), **{field: value})


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
def test_operator_error_bound_is_finite_and_nonnegative(value: float) -> None:
    with pytest.raises(ValueError):
        replace(_controls(), generator_error_bound=value)


@pytest.mark.parametrize("value", [True, "0.1", 1j])
def test_controls_refuse_nonreal_or_coerced_values(value: object) -> None:
    with pytest.raises(TypeError):
        replace(_controls(), generator_error_bound=value)


def test_model_and_control_types_are_checked() -> None:
    with pytest.raises(TypeError):
        decay.assess_autonomous_decay(None, _scales(), _controls())
    with pytest.raises(TypeError):
        decay.assess_autonomous_decay(_scalar(), None, _controls())
    with pytest.raises(TypeError):
        decay.assess_autonomous_decay(_scalar(), _scales(), None)


@pytest.mark.parametrize("mass", [0.0, -1.0])
def test_ode_path_does_not_regularize_singular_or_negative_mass(mass: float) -> None:
    with pytest.raises(ValueError):
        decay.assess_autonomous_decay(
            DampedPencil([[mass]], [[0]], [[1]], [[1]]), _scales(), _controls()
        )


def test_time_scale_overflow_is_refused() -> None:
    with pytest.raises(ValueError):
        decay.assess_autonomous_decay(
            _scalar(), replace(_scales(), time_s=1e308), _controls()
        )
