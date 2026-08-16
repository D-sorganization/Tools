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
        assert c.broadcast_interval_s == 0.1

    def test_lightweight_mode_uses_slow_interval(self) -> None:
        c = PerformanceController(0.1, 2.0, mode=PerformanceMode.LIGHTWEIGHT)
        assert c.broadcast_interval_s == 2.0

    def test_set_mode_switches_interval(self) -> None:
        c = PerformanceController(0.1, 2.0)
        c.set_mode(PerformanceMode.LIGHTWEIGHT)
        assert c.mode == PerformanceMode.LIGHTWEIGHT
        assert c.broadcast_interval_s == 2.0
        c.set_mode(PerformanceMode.PERFORMANCE)
        assert c.broadcast_interval_s == 0.1

    def test_controller_no_longer_exposes_a_scan_period(self) -> None:
        """#4008: nothing may read a control period off the browser-driven mode."""
        c = PerformanceController(0.1, 2.0)
        assert not hasattr(c, "poll_interval_s")

    def test_mode_only_decimates_the_broadcast(self) -> None:
        c = PerformanceController(0.1, 2.0, scan_interval_s=0.1)
        assert c.broadcast_every_n == 1
        c.set_mode(PerformanceMode.LIGHTWEIGHT)
        assert c.broadcast_every_n == 20
        assert c.scan_interval_s == 0.1

    def test_broadcast_decimation_is_never_below_one(self) -> None:
        c = PerformanceController(0.1, 0.05, scan_interval_s=1.0)
        assert c.broadcast_every_n == 1

    def test_scan_interval_defaults_to_the_fast_interval(self) -> None:
        c = PerformanceController(0.1, 2.0)
        assert c.scan_interval_s == 0.1

    def test_scan_interval_is_validated(self) -> None:
        with pytest.raises(ValueError):
            PerformanceController(0.1, 2.0, scan_interval_s=0.0)
        with pytest.raises(TypeError):
            PerformanceController(0.1, 2.0, scan_interval_s="fast")

    def test_set_mode_rejects_non_enum(self) -> None:
        c = PerformanceController(0.1, 2.0)
        with pytest.raises(TypeError):
            c.set_mode("lightweight")

    def test_init_validates_intervals(self) -> None:
        with pytest.raises(ValueError):
            PerformanceController(0.0, 2.0)
        with pytest.raises(ValueError):
            PerformanceController(0.1, -1.0)
        with pytest.raises(TypeError):
            PerformanceController("fast", 2.0)
        with pytest.raises(TypeError):
            PerformanceController(0.1, True)  # bool is not an accepted numeric
        with pytest.raises(ValueError):
            PerformanceController(float("inf"), 2.0)

    def test_init_rejects_non_enum_mode(self) -> None:
        with pytest.raises(TypeError):
            PerformanceController(0.1, 2.0, mode="performance")

    def test_config_reports_mode_and_interval(self) -> None:
        c = PerformanceController(0.1, 2.0, mode=PerformanceMode.LIGHTWEIGHT)
        cfg = c.config()
        assert isinstance(cfg, PerformanceConfig)
        assert cfg.mode == PerformanceMode.LIGHTWEIGHT
        # poll_interval_s remains the HMI's *frame* period (the only thing the
        # mode may change); the control period is reported separately.
        assert cfg.poll_interval_s == 2.0
        assert cfg.broadcast_interval_s == 2.0
        assert cfg.scan_interval_s == 0.1
        assert cfg.broadcast_every_n == 20
        assert cfg.model_dump()["mode"] == "lightweight"

    def test_config_carries_loop_health_counters(self) -> None:
        c = PerformanceController(0.1, 2.0)
        cfg = c.config(scan_overruns=3, historian_write_failures=2)
        assert cfg.scan_overruns == 3
        assert cfg.historian_write_failures == 2

    def test_config_rejects_negative_counters(self) -> None:
        c = PerformanceController(0.1, 2.0)
        with pytest.raises(ValueError):
            c.config(scan_overruns=-1)

    def test_mode_enum_values(self) -> None:
        assert PerformanceMode.PERFORMANCE == "performance"
        assert PerformanceMode.LIGHTWEIGHT == "lightweight"
