"""Tests for the power-supply controller's signal-noise / arc integration.

The controller accumulates every tick's current and voltage feedback into a
bounded rolling window and surfaces NoiseStats (plus an `arcing` flag) in its
status snapshot, so the HMI can quantify how noisy a DC-arc signal is.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from power_supply import PowerSupplyController
from power_supply_models import PowerSupplyConfig
from signal_stats import NoiseMetric


def _running_controller(**cfg_kwargs: object) -> PowerSupplyController:
    ctl = PowerSupplyController(PowerSupplyConfig(**cfg_kwargs))
    ctl.set_permissive(True)
    ctl.set_current_setpoint(10.0)
    return ctl


class TestNoiseAccumulation:
    def test_status_reports_window_sample_count(self) -> None:
        ctl = _running_controller()
        for i in range(5):
            ctl.tick(
                measured_current_a=float(i),
                measured_voltage_v=10.0,
                measured_temp_c=20.0,
                now=float(i),
            )
        status = ctl.status()
        assert status.current_noise.sample_count == 5
        assert status.voltage_noise.sample_count == 5

    def test_steady_signal_has_zero_noise_and_not_arcing(self) -> None:
        ctl = _running_controller()
        for i in range(10):
            ctl.tick(
                measured_current_a=25.0,
                measured_voltage_v=12.0,
                measured_temp_c=20.0,
                now=float(i),
            )
        status = ctl.status()
        assert status.current_noise.std == 0.0
        assert status.current_noise.peak_to_peak == 0.0
        assert status.arcing is False

    def test_window_is_bounded_by_noise_window(self) -> None:
        ctl = _running_controller(noise_window=4)
        for i in range(20):
            ctl.tick(
                measured_current_a=float(i),
                measured_voltage_v=10.0,
                measured_temp_c=20.0,
                now=float(i),
            )
        # Only the last 4 samples are retained.
        assert ctl.status().current_noise.sample_count == 4


class TestArcDetection:
    def test_noisy_current_over_threshold_flags_arcing(self) -> None:
        # Alternating current feedback => large std; tiny threshold => arcing.
        ctl = _running_controller(
            noise_metric=NoiseMetric.STD, current_arc_threshold=1.0
        )
        for i in range(10):
            ctl.tick(
                measured_current_a=0.0 if i % 2 == 0 else 40.0,
                measured_voltage_v=12.0,
                measured_temp_c=20.0,
                now=float(i),
            )
        status = ctl.status()
        assert status.current_noise.arcing is True
        assert status.arcing is True

    def test_no_threshold_never_arcs(self) -> None:
        ctl = _running_controller()  # thresholds default None
        for i in range(10):
            ctl.tick(
                measured_current_a=0.0 if i % 2 == 0 else 40.0,
                measured_voltage_v=0.0 if i % 2 else 30.0,
                measured_temp_c=20.0,
                now=float(i),
            )
        status = ctl.status()
        assert status.current_noise.arcing is False
        assert status.voltage_noise.arcing is False
        assert status.arcing is False

    def test_voltage_arc_alone_sets_overall_arcing(self) -> None:
        ctl = _running_controller(
            noise_metric=NoiseMetric.PEAK_TO_PEAK, voltage_arc_threshold=5.0
        )
        for i in range(10):
            ctl.tick(
                measured_current_a=25.0,  # steady current
                measured_voltage_v=0.0 if i % 2 == 0 else 30.0,  # noisy voltage
                measured_temp_c=20.0,
                now=float(i),
            )
        status = ctl.status()
        assert status.current_noise.arcing is False
        assert status.voltage_noise.arcing is True
        assert status.arcing is True


class TestWindowResizePreservesRecent:
    def test_update_config_resizes_and_keeps_recent(self) -> None:
        ctl = _running_controller(noise_window=100)
        for i in range(10):
            ctl.tick(
                measured_current_a=float(i),
                measured_voltage_v=10.0,
                measured_temp_c=20.0,
                now=float(i),
            )
        # Shrinking the window keeps only the most-recent samples.
        ctl.update_config(ctl.config.model_copy(update={"noise_window": 3}))
        assert ctl.status().current_noise.sample_count == 3
