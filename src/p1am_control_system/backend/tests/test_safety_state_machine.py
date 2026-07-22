"""Direct unit tests for the shared ``SafetyStateMachine`` base.

The two production controllers (temperature, power-supply) share this
scaffolding; their own suites already exercise it end-to-end. Here we test the
base in isolation through a minimal concrete subclass so the shared mechanics
(``_safe_finite``, the E-stop latch, permissive arm/disarm, trip latch +
acknowledge, the force-off predicate, and the ``_on_*`` reset hooks) are pinned
down independently of either control law.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_state_machine import SafetyStateMachine  # noqa: E402

from shared.python.compatibility import StrEnum  # noqa: E402


class _State(StrEnum):
    IDLE = "idle"
    ARMED = "armed"
    RUNNING = "running"
    TRIPPED = "tripped"


class _Fake(SafetyStateMachine[_State]):
    """Minimal concrete machine that records which reset hooks fired."""

    def __init__(self) -> None:
        super().__init__(
            idle=_State.IDLE,
            armed=_State.ARMED,
            running=_State.RUNNING,
            tripped=_State.TRIPPED,
            logger=logging.getLogger("test.safety_state_machine"),
        )
        self.setpoint = 0.0
        self.disarms = 0
        self.estop_engaged_calls = 0
        self.estop_cleared_calls = 0
        self.ack_calls = 0

    def _on_disarm(self) -> None:
        self.setpoint = 0.0
        self.disarms += 1

    def _on_estop_engaged(self) -> None:
        self.setpoint = 0.0
        self.estop_engaged_calls += 1

    def _on_estop_cleared(self) -> None:
        self.setpoint = 0.0
        self.estop_cleared_calls += 1

    def _on_trip_acknowledged(self) -> None:
        self.setpoint = 0.0
        self.ack_calls += 1

    def _estop_log_message(self) -> str:
        return "E-STOP engaged — fake actuator latched off"

    # Test-only helpers to drive the machine into runtime states.
    def arm_and_run(self) -> None:
        self._apply_permissive(True)
        self._state = _State.RUNNING
        self.setpoint = 5.0

    def force_trip(self, key: str) -> None:
        self._trips.add(key)
        self._latch_trips(log_context="ctx")


@pytest.fixture
def machine() -> _Fake:
    return _Fake()


# -- _safe_finite -------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12.5, 12.5),
        (0, 0.0),
        (-3, -3.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (True, 0.0),
        (False, 0.0),
        ("junk", 0.0),
        (None, 0.0),
    ],
)
def test_safe_finite_coerces(value: object, expected: float) -> None:
    assert SafetyStateMachine._safe_finite(value) == expected  # type: ignore[arg-type]


# -- construction -------------------------------------------------------


def test_initial_state_is_idle_disarmed(machine: _Fake) -> None:
    assert machine.state is _State.IDLE
    assert machine.permissive is False
    assert machine.estopped is False
    assert machine.trips == []


# -- permissive arm / disarm -------------------------------------------


def test_permissive_true_arms_from_idle(machine: _Fake) -> None:
    machine._apply_permissive(True)
    assert machine.state is _State.ARMED
    assert machine.permissive is True


def test_permissive_false_disarms_and_resets(machine: _Fake) -> None:
    machine.arm_and_run()
    machine._apply_permissive(False)
    assert machine.state is _State.IDLE
    assert machine.permissive is False
    assert machine.setpoint == 0.0
    assert machine.disarms == 1


def test_permissive_does_not_leave_tripped(machine: _Fake) -> None:
    machine.arm_and_run()
    machine.force_trip("HH")
    assert machine.state is _State.TRIPPED
    machine._apply_permissive(True)
    assert machine.state is _State.TRIPPED
    # A trip latch is not cleared by a permissive change.
    assert machine.trips == ["HH"]


def test_permissive_rejects_non_bool(machine: _Fake) -> None:
    with pytest.raises(TypeError):
        machine._apply_permissive(1)  # type: ignore[arg-type]


# -- E-stop one-way latch ----------------------------------------------


def test_engage_estop_latches_and_disarms(machine: _Fake) -> None:
    machine.arm_and_run()
    machine.engage_estop()
    assert machine.estopped is True
    assert machine.permissive is False
    assert machine.state is _State.IDLE
    assert machine.setpoint == 0.0
    assert machine.estop_engaged_calls == 1


def test_permissive_ignored_while_estopped(machine: _Fake) -> None:
    machine.engage_estop()
    machine._apply_permissive(True)
    assert machine.permissive is False
    assert machine.state is _State.IDLE


def test_clear_estop_releases_and_resets(machine: _Fake) -> None:
    machine.arm_and_run()
    machine.engage_estop()
    machine.clear_estop()
    assert machine.estopped is False
    assert machine.permissive is False
    assert machine.state is _State.IDLE
    assert machine.estop_cleared_calls == 1


def test_clear_estop_noop_when_not_latched(machine: _Fake) -> None:
    machine.clear_estop()
    assert machine.estopped is False
    assert machine.estop_cleared_calls == 0


# -- trip latch + acknowledge ------------------------------------------


def test_latch_trips_flips_to_tripped(machine: _Fake) -> None:
    machine.arm_and_run()
    machine.force_trip("HH")
    assert machine.state is _State.TRIPPED
    assert machine.trips == ["HH"]


def test_latch_trips_noop_without_trips(machine: _Fake) -> None:
    machine.arm_and_run()
    machine._latch_trips(log_context="ctx")
    assert machine.state is _State.RUNNING


def test_acknowledge_trip_returns_to_armed_when_permissive(machine: _Fake) -> None:
    machine.arm_and_run()
    machine.force_trip("HH")
    assert machine.acknowledge_trip() is True
    assert machine.state is _State.ARMED
    assert machine.trips == []
    assert machine.setpoint == 0.0
    assert machine.ack_calls == 1


def test_acknowledge_trip_returns_to_idle_without_permissive(machine: _Fake) -> None:
    # Trip while running, then drop permissive is not possible without leaving
    # TRIPPED, so force the flag directly to model a latched-but-disarmed case.
    machine.arm_and_run()
    machine.force_trip("HH")
    machine._permissive = False
    assert machine.acknowledge_trip() is True
    assert machine.state is _State.IDLE


def test_acknowledge_trip_noop_when_not_tripped(machine: _Fake) -> None:
    machine.arm_and_run()
    assert machine.acknowledge_trip() is False
    assert machine.ack_calls == 0


# -- force-off predicate ------------------------------------------------


def test_force_off_true_unless_cleanly_running(machine: _Fake) -> None:
    # IDLE / disarmed -> force off.
    assert machine._should_force_actuator_off() is True
    machine.arm_and_run()
    # Cleanly RUNNING + permissive + no trip/estop -> do not force off.
    assert machine._should_force_actuator_off() is False


@pytest.mark.parametrize("mutate", ["estop", "trip", "disarm"])
def test_force_off_each_unsafe_condition(machine: _Fake, mutate: str) -> None:
    machine.arm_and_run()
    assert machine._should_force_actuator_off() is False
    if mutate == "estop":
        machine.engage_estop()
    elif mutate == "trip":
        machine.force_trip("HH")
    else:
        machine._apply_permissive(False)
    assert machine._should_force_actuator_off() is True
