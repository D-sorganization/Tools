"""Unit tests for the global performance-mode controller."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("pydantic")

from performance import (  # noqa: E402
    PerformanceConfig,
    PerformanceController,
    PerformanceMode,
)


class TestPerformanceController:
    def test_default_mode_uses_fast_interval(self) -> None:
        c = PerformanceController(0.1, 2.0)
        assert c.mode == PerformanceMode.PERFORMANCE
        assert c.poll_interval_s == 0.1

    def test_lightweight_mode_uses_slow_interval(self) -> None:
        c = PerformanceController(0.1, 2.0, mode=PerformanceMode.LIGHTWEIGHT)
        assert c.poll_interval_s == 2.0

    def test_set_mode_switches_interval(self) -> None:
        c = PerformanceController(0.1, 2.0)
        c.set_mode(PerformanceMode.LIGHTWEIGHT)
        assert c.mode == PerformanceMode.LIGHTWEIGHT
        assert c.poll_interval_s == 2.0
        c.set_mode(PerformanceMode.PERFORMANCE)
        assert c.poll_interval_s == 0.1

    def test_set_mode_rejects_non_enum(self) -> None:
        c = PerformanceController(0.1, 2.0)
        with pytest.raises(TypeError):
            c.set_mode("lightweight")  # type: ignore[arg-type]

    def test_init_validates_intervals(self) -> None:
        with pytest.raises(ValueError):
            PerformanceController(0.0, 2.0)
        with pytest.raises(ValueError):
            PerformanceController(0.1, -1.0)
        with pytest.raises(TypeError):
            PerformanceController("fast", 2.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            PerformanceController(0.1, True)  # bool is not an accepted numeric
        with pytest.raises(ValueError):
            PerformanceController(float("inf"), 2.0)

    def test_init_rejects_non_enum_mode(self) -> None:
        with pytest.raises(TypeError):
            PerformanceController(0.1, 2.0, mode="performance")  # type: ignore[arg-type]

    def test_config_reports_mode_and_interval(self) -> None:
        c = PerformanceController(0.1, 2.0, mode=PerformanceMode.LIGHTWEIGHT)
        cfg = c.config()
        assert isinstance(cfg, PerformanceConfig)
        assert cfg.mode == PerformanceMode.LIGHTWEIGHT
        assert cfg.poll_interval_s == 2.0
        assert cfg.model_dump()["mode"] == "lightweight"

    def test_mode_enum_values(self) -> None:
        assert PerformanceMode.PERFORMANCE == "performance"
        assert PerformanceMode.LIGHTWEIGHT == "lightweight"
