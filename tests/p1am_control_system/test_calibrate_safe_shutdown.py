"""Regression tests for issue #3997 — calibration must not leave AOs energized.

``calibration/calibrate.py`` drives the P1AM analog outputs to up to 100 %
(20 mA) through pass-through PIDs. Teardown used to unmap the PIDs while
leaving the firmware's output routing pointing at the (now frozen) tag, so the
firmware kept writing the last commanded value every scan and the AO stayed at
20 mA indefinitely.

These are unit tests against a fake ``PLC`` that records the exact command
order, because the real CLI talks to live hardware. The property under test is
the *ordering*: command zero, read back to confirm zero, and only then unmap.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_CALIBRATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "p1am_control_system"
    / "calibration"
    / "calibrate.py"
)


def _load_calibrate() -> ModuleType:
    """Import calibrate.py by path (the calibration dir is not a package)."""
    pytest.importorskip("pymodbus")
    spec = importlib.util.spec_from_file_location("p1am_calibrate", _CALIBRATE_PATH)
    assert spec is not None, f"cannot build a module spec for {_CALIBRATE_PATH}"
    loader = spec.loader
    assert loader is not None, f"no loader for {_CALIBRATE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


calibrate = _load_calibrate()


class FakePLC:
    """Records the command order issued by the calibration CLI.

    Emulates the firmware's pass-through behaviour: writing a PID setpoint
    moves that PID's ``cv_tag``. ``sticky`` reproduces the #3997 hardware
    symptom where the AO tag refuses to follow the commanded zero.
    """

    def __init__(self, *, sticky: bool = False) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.tags: dict[int, float] = {tag: 0.0 for tag in calibrate.AO_TAG.values()}
        self.sticky = sticky
        self.closed = False

    # -- calibration surface -------------------------------------------------
    def connect(self) -> None:
        self.calls.append(("connect",))

    def close(self) -> None:
        self.calls.append(("close",))
        self.closed = True

    def set_pid_setpoint(self, pid_index: int, setpoint: float) -> None:
        self.calls.append(("set_pid_setpoint", pid_index, setpoint))
        if self.sticky:
            return
        for channel, pid in calibrate.PID_FOR_AO.items():
            if pid == pid_index:
                self.tags[calibrate.AO_TAG[channel]] = setpoint

    def read_tag(self, tag_id: int) -> float:
        self.calls.append(("read_tag", tag_id))
        return self.tags.get(tag_id, 0.0)

    def configure_pid(
        self,
        pid_index: int,
        pv_tag: int,
        cv_tag: int,
        setpoint: float,
        kp: float,
        ki: float,
        kd: float,
    ) -> None:
        self.calls.append(("configure_pid", pid_index, pv_tag, cv_tag))

    def set_output_routing(self, channel: int, tag_id: int) -> None:
        self.calls.append(("set_output_routing", channel, tag_id))

    def set_input_routing(self, slot: int, tag_id: int) -> None:
        self.calls.append(("set_input_routing", slot, tag_id))

    def write_coil(self, address: int, value: bool) -> None:
        self.calls.append(("write_coil", address, value))

    # -- helpers -------------------------------------------------------------
    def index_of(self, call: tuple[Any, ...]) -> int:
        assert call in self.calls, f"{call!r} never issued; calls={self.calls!r}"
        return self.calls.index(call)

    def energize(self, percent: float) -> None:
        for tag in calibrate.AO_TAG.values():
            self.tags[tag] = percent


@pytest.fixture(autouse=True)
def _no_dwell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the hardware settle dwells so the unit tests stay fast."""
    monkeypatch.setattr(calibrate.time, "sleep", lambda _seconds: None)


def test_teardown_zeroes_and_confirms_before_unmapping() -> None:
    plc = FakePLC()
    plc.energize(100.0)

    calibrate.cmd_teardown(object(), plc)

    for channel, pid_index in calibrate.PID_FOR_AO.items():
        ao_tag = calibrate.AO_TAG[channel]
        zero_cmd = plc.index_of(("set_pid_setpoint", pid_index, 0.0))
        confirm = plc.index_of(("read_tag", ao_tag))
        unmap_pid = plc.index_of(
            ("configure_pid", pid_index, calibrate.UNMAPPED_TAG, calibrate.UNMAPPED_TAG)
        )
        unmap_route = plc.index_of(
            ("set_output_routing", channel, calibrate.UNMAPPED_TAG)
        )

        assert zero_cmd < confirm, "AO must be commanded to 0% before read-back"
        assert confirm < unmap_pid, "AO zero must be CONFIRMED before unmapping the PID"
        assert unmap_pid < unmap_route

    assert all(value == 0.0 for value in plc.tags.values())


def test_teardown_unmaps_routing_and_raises_when_output_will_not_zero() -> None:
    """A stuck AO must still surrender the channel to the firmware safe path."""
    plc = FakePLC(sticky=True)
    plc.energize(100.0)

    with pytest.raises(SystemExit) as exc_info:
        calibrate.cmd_teardown(object(), plc)

    assert "0%" in str(exc_info.value) or "0 %" in str(exc_info.value)
    for channel in calibrate.AO_TAG:
        # Unmapping the CHANNEL is what makes the firmware write 0 % itself.
        assert ("set_output_routing", channel, calibrate.UNMAPPED_TAG) in plc.calls


def test_main_drives_outputs_to_zero_when_a_command_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #3997: a Modbus SystemExit mid-sweep left the AO parked."""
    plc = FakePLC()

    def _boom(_args: Any, target: Any) -> None:
        target.set_pid_setpoint(calibrate.PID_FOR_AO[0], 75.0)
        raise SystemExit("write_registers(202, [0, 17024]) failed: ModbusIOException")

    monkeypatch.setattr(calibrate, "PLC", lambda host, port: plc)
    monkeypatch.setitem(calibrate._COMMANDS, "sweep", _boom)

    with pytest.raises(SystemExit):
        calibrate.main(["sweep", "--channel", "0"])

    for channel, pid_index in calibrate.PID_FOR_AO.items():
        zero_cmd = plc.index_of(("set_pid_setpoint", pid_index, 0.0))
        confirm = plc.index_of(("read_tag", calibrate.AO_TAG[channel]))
        assert zero_cmd < confirm < plc.index_of(("close",))
    assert plc.closed is True
    assert all(value == 0.0 for value in plc.tags.values())


def test_main_drives_outputs_to_zero_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plc = FakePLC()

    def _interrupt(_args: Any, target: Any) -> None:
        target.set_pid_setpoint(calibrate.PID_FOR_AO[1], 100.0)
        raise KeyboardInterrupt

    monkeypatch.setattr(calibrate, "PLC", lambda host, port: plc)
    monkeypatch.setitem(calibrate._COMMANDS, "sweep", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        calibrate.main(["sweep", "--channel", "1"])

    assert all(value == 0.0 for value in plc.tags.values())
    assert plc.closed is True


def test_main_leaves_a_successful_ao_command_energized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator needs the AO held at the commanded percent to meter it."""
    plc = FakePLC()
    monkeypatch.setattr(calibrate, "PLC", lambda host, port: plc)

    assert calibrate.main(["ao", "--channel", "0", "--percent", "100"]) == 0

    assert plc.tags[calibrate.AO_TAG[0]] == 100.0
    assert ("set_pid_setpoint", calibrate.PID_FOR_AO[0], 0.0) not in plc.calls


def test_main_reports_but_does_not_mask_the_original_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the emergency zero itself fails, the original error must surface."""
    plc = FakePLC(sticky=True)
    plc.energize(100.0)

    def _boom(_args: Any, _plc: Any) -> None:
        raise SystemExit("original modbus failure")

    monkeypatch.setattr(calibrate, "PLC", lambda host, port: plc)
    monkeypatch.setitem(calibrate._COMMANDS, "sweep", _boom)

    with caplog.at_level("ERROR", logger="p1am.calibration"):
        with pytest.raises(SystemExit, match="original modbus failure"):
            calibrate.main(["sweep", "--channel", "0"])

    assert any("still be energized" in record.message for record in caplog.records)
    assert plc.closed is True


def test_zero_analog_outputs_rejects_a_plc_without_the_calibration_surface() -> None:
    with pytest.raises(TypeError):
        calibrate.zero_analog_outputs(object())
