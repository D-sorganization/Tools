"""Regression tests for overlay-state synchronization on model switch.

The bug being fixed: when the user has overlay toggles checked (e.g.
"Mobility Ellipsoids") and switches to a different model via the
toolstrip dropdown, the new model's pendulum widget never receives the
current state, so the overlays vanish until the user cycles the
checkbox.

Contract under test
-------------------
``apply_toolstrip_overlay_state(toolstrip, pendulum)`` is the single
function that maps every overlay-bearing toolstrip control onto the
matching ``set_*`` method on a pendulum widget. It is the LOD/DRY
focal point: the model-switch handler calls it once and the new
widget is fully in sync.
"""

from __future__ import annotations

from typing import Any

import pytest
from PyQt6.QtWidgets import QWidget

from double_pendulum_golf.gui.overlay_state import apply_toolstrip_overlay_state
from double_pendulum_golf.gui.toolstrip_widget import ToolStrip


class _RecordingPendulum(QWidget):
    """Stand-in for PendulumWidget that records every set_* call.

    Implements the entire contract that ``apply_toolstrip_overlay_state``
    is allowed to call (LOD: the helper does not reach into anything
    else). Each setter just stores its argument.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: dict[str, Any] = {}

    def set_show_forces(self, v: bool) -> None:
        self.calls["set_show_forces"] = bool(v)

    def set_show_zero_torque_forces(self, v: bool) -> None:
        self.calls["set_show_zero_torque_forces"] = bool(v)

    def set_show_mob_ellipsoids(self, v: bool) -> None:
        self.calls["set_show_mob_ellipsoids"] = bool(v)

    def set_show_force_ellipsoids(self, v: bool) -> None:
        self.calls["set_show_force_ellipsoids"] = bool(v)

    def set_show_com(self, v: bool) -> None:
        self.calls["set_show_com"] = bool(v)

    def set_show_torque_vectors(self, v: bool) -> None:
        self.calls["set_show_torque_vectors"] = bool(v)

    def set_show_moment_of_force(self, v: bool) -> None:
        self.calls["set_show_moment_of_force"] = bool(v)

    def set_show_sum_moments(self, v: bool) -> None:
        self.calls["set_show_sum_moments"] = bool(v)

    def set_3d_mode(self, v: bool) -> None:
        self.calls["set_3d_mode"] = bool(v)

    def set_force_scale(self, v: float) -> None:
        self.calls["set_force_scale"] = float(v)

    def set_mob_ellipsoid_scale(self, v: float) -> None:
        self.calls["set_mob_ellipsoid_scale"] = float(v)

    def set_force_ellipsoid_scale(self, v: float) -> None:
        self.calls["set_force_ellipsoid_scale"] = float(v)


# ── DbC: argument validation ──────────────────────────────────────────


def test_rejects_none_toolstrip(qapp) -> None:
    pw = _RecordingPendulum()
    with pytest.raises(ValueError, match="toolstrip"):
        apply_toolstrip_overlay_state(None, pw)  # type: ignore[arg-type]


def test_rejects_none_pendulum(qapp) -> None:
    ts = ToolStrip()
    with pytest.raises(ValueError, match="pendulum"):
        apply_toolstrip_overlay_state(ts, None)  # type: ignore[arg-type]


# ── Boolean overlays propagate from toolstrip to pendulum ────────────


def test_mobility_ellipsoid_checked_state_pushed(qapp) -> None:
    """Bug fix: the original symptom — Mobility Ellipsoids checked
    on the toolstrip but the new pendulum widget didn't get the state."""
    ts = ToolStrip()
    ts.chk_mob.setChecked(True)
    pw = _RecordingPendulum()

    apply_toolstrip_overlay_state(ts, pw)

    assert pw.calls.get("set_show_mob_ellipsoids") is True


def test_force_ellipsoid_checked_state_pushed(qapp) -> None:
    ts = ToolStrip()
    ts.chk_force_ell.setChecked(True)
    pw = _RecordingPendulum()

    apply_toolstrip_overlay_state(ts, pw)

    assert pw.calls.get("set_show_force_ellipsoids") is True


def test_unchecked_overlays_pushed_as_false(qapp) -> None:
    """When the user unchecks something on the old model, the new
    model must also see ``False`` — the helper is total, not partial."""
    ts = ToolStrip()
    ts.chk_mob.setChecked(False)
    ts.chk_forces.setChecked(False)
    pw = _RecordingPendulum()

    apply_toolstrip_overlay_state(ts, pw)

    assert pw.calls.get("set_show_mob_ellipsoids") is False
    assert pw.calls.get("set_show_forces") is False


def test_every_overlay_checkbox_is_pushed(qapp) -> None:
    """Pre/post: after one call, every checkbox state has been
    propagated to the pendulum widget exactly once."""
    ts = ToolStrip()
    ts.chk_forces.setChecked(True)
    ts.chk_zero_torque.setChecked(True)
    ts.chk_mob.setChecked(True)
    ts.chk_force_ell.setChecked(True)
    ts.chk_com.setChecked(True)
    ts.chk_torque.setChecked(True)
    ts.chk_mof.setChecked(True)
    ts.chk_sum_moments.setChecked(True)
    ts.chk_3d.setChecked(True)
    pw = _RecordingPendulum()

    apply_toolstrip_overlay_state(ts, pw)

    expected_setters = {
        "set_show_forces",
        "set_show_zero_torque_forces",
        "set_show_mob_ellipsoids",
        "set_show_force_ellipsoids",
        "set_show_com",
        "set_show_torque_vectors",
        "set_show_moment_of_force",
        "set_show_sum_moments",
        "set_3d_mode",
    }
    for setter in expected_setters:
        assert setter in pw.calls, f"{setter} was not called"
        assert pw.calls[setter] is True


# ── Scale sliders also propagate ──────────────────────────────────────


def _expected_scale(slider: Any) -> float:
    """Compute the display scale a slider should emit, divisor-aware."""
    divisor = slider.property("scale_divisor") or 10
    return slider.value() / float(divisor)


def test_force_scale_pushed(qapp) -> None:
    ts = ToolStrip()
    ts._sld_force.setValue(25)
    pw = _RecordingPendulum()

    apply_toolstrip_overlay_state(ts, pw)

    assert pw.calls.get("set_force_scale") == pytest.approx(
        _expected_scale(ts._sld_force)
    )


def test_mob_ellipsoid_scale_pushed(qapp) -> None:
    ts = ToolStrip()
    # Use a value that's valid for whatever divisor the slider uses
    ts._sld_mob.setValue(50)
    pw = _RecordingPendulum()

    apply_toolstrip_overlay_state(ts, pw)

    assert pw.calls.get("set_mob_ellipsoid_scale") == pytest.approx(
        _expected_scale(ts._sld_mob)
    )


def test_force_ellipsoid_scale_pushed(qapp) -> None:
    ts = ToolStrip()
    ts._sld_force_ell.setValue(30)
    pw = _RecordingPendulum()

    apply_toolstrip_overlay_state(ts, pw)

    assert pw.calls.get("set_force_ellipsoid_scale") == pytest.approx(
        _expected_scale(ts._sld_force_ell)
    )


# ── LOD: missing setters are skipped silently ────────────────────────


class _MinimalPendulum(QWidget):
    """A pendulum widget that supports only one of the setters.

    Used to verify that the helper does not crash when a model widget
    happens to lack a particular setter (e.g. a future read-only model).
    """

    def __init__(self) -> None:
        super().__init__()
        self.mob_calls: list[bool] = []

    def set_show_mob_ellipsoids(self, v: bool) -> None:
        self.mob_calls.append(bool(v))


def test_helper_skips_missing_setters_without_crashing(qapp) -> None:
    ts = ToolStrip()
    ts.chk_mob.setChecked(True)
    pw = _MinimalPendulum()

    # Must not raise
    apply_toolstrip_overlay_state(ts, pw)

    # The one setter that exists should still get called
    assert pw.mob_calls == [True]
