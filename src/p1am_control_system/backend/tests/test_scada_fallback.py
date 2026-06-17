"""Tests for the pure-Python ``scada_fallback`` shim and its wiring in ``main``.

These guard the fix for issue #3515: the backend used to hard-import
``from tools_core import scada`` with no fallback, so it raised
``ModuleNotFoundError`` whenever the Rust wheel was not installed. The backend
now falls back to ``scada_fallback`` and must import cleanly either way.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest
import scada_fallback


class TestMovingAverageFallback:
    def test_centered_window(self) -> None:
        # Matches the centered ("same") semantics of the Rust kernel.
        result = scada_fallback.moving_average([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        assert len(result) == 5
        # Interior points are simple 3-point means.
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)
        assert result[3] == pytest.approx(4.0)

    def test_empty(self) -> None:
        assert scada_fallback.moving_average([], 3) == []

    def test_window_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            scada_fallback.moving_average([1.0, 2.0], 0)


class TestExponentialSmoothingFallback:
    def test_seeded_with_first_value(self) -> None:
        result = scada_fallback.exponential_smoothing([10.0, 20.0, 30.0], 0.5)
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(0.5 * 20.0 + 0.5 * 10.0)
        assert result[2] == pytest.approx(0.5 * 30.0 + 0.5 * result[1])

    def test_empty(self) -> None:
        assert scada_fallback.exponential_smoothing([], 0.5) == []

    def test_alpha_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            scada_fallback.exponential_smoothing([1.0], 0.0)
        with pytest.raises(ValueError, match="alpha"):
            scada_fallback.exponential_smoothing([1.0], 1.5)


class TestAlarmEngineFallback:
    def _engine(self) -> scada_fallback.AlarmEngine:
        return scada_fallback.AlarmEngine(
            {"T1": {"lolo": 10.0, "low": 20.0, "high": 80.0, "hihi": 90.0}}
        )

    def test_state_transitions_and_events(self) -> None:
        engine = self._engine()
        assert engine.update_tag("T1", 50.0) == []  # Normal, no event
        events = engine.update_tag("T1", 85.0)  # -> High
        assert len(events) == 1
        assert events[0]["current_state"] == scada_fallback.AlarmState.HIGH
        active = engine.get_active_alarms()
        assert len(active) == 1
        assert active[0]["severity"] == 1

    def test_acknowledge(self) -> None:
        engine = self._engine()
        engine.update_tag("T1", 95.0)  # HiHi
        assert engine.acknowledge_alarm("T1", "OperatorA") is True
        state = engine.get_alarm_state("T1")
        assert state["acknowledged"] is True
        assert state["acknowledged_by"] == "OperatorA"

    def test_acknowledge_normal_returns_false(self) -> None:
        engine = self._engine()
        engine.update_tag("T1", 50.0)
        assert engine.acknowledge_alarm("T1", "OperatorA") is False

    def test_too_many_tags_rejected(self) -> None:
        limits = {
            f"T{i}": {"lolo": 0.0, "low": 1.0, "high": 2.0, "hihi": 3.0}
            for i in range(33)
        }
        with pytest.raises(ValueError, match="32 tags"):
            scada_fallback.AlarmEngine(limits)

    def test_non_monotonic_limits_rejected(self) -> None:
        with pytest.raises(ValueError, match="monotonic|lolo <= low"):
            scada_fallback.AlarmEngine(
                {"T1": {"lolo": 90.0, "low": 20.0, "high": 80.0, "hihi": 10.0}}
            )

    def test_unregistered_tag_raises(self) -> None:
        engine = self._engine()
        with pytest.raises(KeyError):
            engine.update_tag("UNKNOWN", 1.0)


def test_backend_imports_without_tools_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` must import even when ``tools_core`` is unavailable.

    Simulate the wheel being absent by blocking the import, then reload
    ``main`` and assert it falls back to the pure-Python ``scada_fallback``
    symbols.
    """
    pytest.importorskip("sqlmodel")
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tools_core" or name.startswith("tools_core."):
            raise ModuleNotFoundError("No module named 'tools_core'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, "tools_core", raising=False)
    monkeypatch.delitem(sys.modules, "main", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    main = importlib.import_module("main")
    main = importlib.reload(main)

    # The fallback helpers must be wired up and functional.
    assert main.moving_average([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]
    smoothed = main.exponential_smoothing([10.0, 20.0], 0.5)
    assert smoothed[0] == pytest.approx(10.0)
    engine = main.AlarmEngine(
        {"T1": {"lolo": 0.0, "low": 1.0, "high": 2.0, "hihi": 3.0}}
    )
    assert engine.get_active_alarms() == []
