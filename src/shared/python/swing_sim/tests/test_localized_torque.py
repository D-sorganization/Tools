"""Localized additive commanded-torque contracts and integration tests."""

from __future__ import annotations

import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.localized_torque import (
    add_localized_offsets,
    require_offsets_within_duration,
)
from shared.python.swing_sim.run_config import (
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
    DoublePendulumRunConfig,
    LocalizedTorqueOffset,
)
from shared.python.swing_sim.swing_source import DoublePendulumSwing
from shared.python.swing_sim.types import PendulumState

from .test_prescribed_run import _prescribed_swing, _profile

pytestmark = pytest.mark.unit


def _localized_swing(
    offsets: tuple[LocalizedTorqueOffset, ...], **kwargs: object
) -> DoublePendulumSwing:
    defaults: dict[str, object] = {
        "duration": 0.1,
        "dt": 0.001,
        "backend": "python",
        "gravity_m_s2": 0.0,
        "initial_state": PendulumState(0.0, 0.0, 0.0, 0.0),
        "run_config": DoublePendulumRunConfig(commanded_torque_offsets=offsets),
    }
    defaults.update(kwargs)
    return DoublePendulumSwing(**defaults)  # type: ignore[arg-type]


def test_locus_is_exactly_half_open_and_rejects_invalid_sample_times() -> None:
    offset = LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.02, 0.04), 3.5)

    assert offset.is_active(0.02)
    assert offset.is_active(0.039999999)
    assert not offset.is_active(0.04)
    assert not offset.is_active(0.01)
    for invalid in (True, "0.03", float("nan")):
        with pytest.raises(ContractViolationError, match="sample time"):
            offset.is_active(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"joint_id": "swing.wrist"},
        {"joint_id": "joint.elbow"},
        {"time_window_s": (0.04, 0.02)},
        {"time_window_s": None},
        {"time_window_s": (0.02,)},
        {"time_window_s": (-0.01, 0.02)},
        {"time_window_s": (0.02, float("inf"))},
        {"torque_nm": float("nan")},
        {"torque_nm": True},
    ],
)
def test_locus_rejects_invalid_contract(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "joint_id": SHOULDER_JOINT_ID,
        "time_window_s": (0.02, 0.04),
        "torque_nm": 3.5,
    }
    values.update(kwargs)
    with pytest.raises(ContractViolationError):
        LocalizedTorqueOffset(**values)  # type: ignore[arg-type]


def test_offsets_add_to_passive_and_prescribed_commands() -> None:
    offsets = (
        LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.02, 0.04), 3.0),
        LocalizedTorqueOffset(WRIST_JOINT_ID, (0.02, 0.04), -2.0),
    )
    passive = _localized_swing(offsets)
    prescribed = _prescribed_swing(
        run_config=DoublePendulumRunConfig.prescribed(
            _profile().profile_id,
            commanded_torque_offsets=offsets,
        )
    )

    assert passive.joint_torques_at(0.019) == pytest.approx(
        {SHOULDER_JOINT_ID: 0.0, WRIST_JOINT_ID: 0.0}
    )
    assert passive.joint_torques_at(0.02) == pytest.approx(
        {SHOULDER_JOINT_ID: 3.0, WRIST_JOINT_ID: -2.0}
    )
    assert passive.joint_torques_at(0.04) == pytest.approx(
        {SHOULDER_JOINT_ID: 0.0, WRIST_JOINT_ID: 0.0}
    )
    assert prescribed.joint_torques_at(0.03) == pytest.approx(
        {SHOULDER_JOINT_ID: 23.0, WRIST_JOINT_ID: -7.0}
    )


@pytest.mark.parametrize(
    "base",
    [(True, 0.0), ("2.0", 0.0), (float("nan"), 0.0), (0.0,)],
)
def test_add_helper_rejects_coercive_nonfinite_or_malformed_base(
    base: object,
) -> None:
    offset = LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.02, 0.04), 1.0)
    with pytest.raises(ContractViolationError, match="base_torques_nm"):
        add_localized_offsets(base, (offset,), 0.03)  # type: ignore[arg-type]


@pytest.mark.parametrize("offsets", [["bad"], "bad", (None,)])
def test_public_helpers_reject_malformed_offset_collections(offsets: object) -> None:
    with pytest.raises(ContractViolationError, match="offsets"):
        add_localized_offsets((0.0, 0.0), offsets, 0.03)  # type: ignore[arg-type]
    with pytest.raises(ContractViolationError, match="offsets"):
        require_offsets_within_duration(offsets, 0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize("offsets", [None, 7, "bad", {"offset": "bad"}])
def test_run_config_rejects_malformed_offset_collections_with_contract_error(
    offsets: object,
) -> None:
    with pytest.raises(ContractViolationError, match="commanded_torque_offsets"):
        DoublePendulumRunConfig(  # type: ignore[arg-type]
            commanded_torque_offsets=offsets
        )


@pytest.mark.parametrize("duration", [True, "0.1", float("nan"), 0.0])
def test_duration_helper_rejects_invalid_raw_domain(duration: object) -> None:
    offset = LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.02, 0.04), 1.0)
    with pytest.raises(ContractViolationError, match="duration"):
        require_offsets_within_duration((offset,), duration)  # type: ignore[arg-type]


def test_command_is_evaluated_at_every_rk4_substep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[float] = []
    original_is_active = LocalizedTorqueOffset.is_active

    def capture_active(offset: LocalizedTorqueOffset, time_s: float) -> bool:
        observed.append(time_s)
        return original_is_active(offset, time_s)

    monkeypatch.setattr(LocalizedTorqueOffset, "is_active", capture_active)
    _localized_swing(
        (LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.0, 0.001), 2.0),),
        duration=0.001,
        dt=0.001,
    )

    assert observed == [0.0, 0.0005, 0.0005, 0.001]


def test_auto_uses_python_and_explicit_rust_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shared.python.swing_sim import _rust_facade

    monkeypatch.setattr(_rust_facade, "rust_available", lambda: True)
    monkeypatch.setattr(
        _rust_facade,
        "simulate_rust",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("localized commands reached Rust")
        ),
    )
    offsets = (LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.0, 0.01), 2.0),)

    assert _localized_swing(offsets, backend="auto").backend == "python"
    with pytest.raises(ContractViolationError, match="localized.*Rust"):
        _localized_swing(offsets, backend="rust")
