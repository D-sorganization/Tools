"""Canonical signal-quality contracts shared across every SCADA layer."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_quality import (  # noqa: E402
    SignalFrameFactory,
    SignalQuality,
    SignalSample,
)

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017
NOW = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)


def test_signal_sample_requires_complete_aware_provenance() -> None:
    sample = SignalSample(
        value=12.5,
        source_timestamp=NOW - timedelta(milliseconds=5),
        server_timestamp=NOW,
        quality=SignalQuality.GOOD,
        diagnostic_reason=None,
        sequence=7,
        source="synthetic.driver",
    )

    assert sample.value == 12.5
    assert sample.age_seconds(NOW + timedelta(seconds=1)) == pytest.approx(1.005)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_signal_sample_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        SignalSample(
            value=value,
            source_timestamp=NOW,
            server_timestamp=NOW,
            quality=SignalQuality.GOOD,
            diagnostic_reason=None,
            sequence=1,
            source="synthetic.driver",
        )


def test_degraded_quality_requires_diagnostic_reason() -> None:
    with pytest.raises(ValueError, match="diagnostic_reason"):
        SignalSample(
            value=1.0,
            source_timestamp=NOW,
            server_timestamp=NOW,
            quality=SignalQuality.STALE,
            diagnostic_reason=None,
            sequence=1,
            source="synthetic.driver",
        )


def test_frame_factory_sequences_good_stale_and_simulated_scans() -> None:
    clock_values = iter([NOW, NOW + timedelta(seconds=1), NOW + timedelta(seconds=2)])
    factory = SignalFrameFactory(clock=lambda: next(clock_values))

    good = factory.good({"TAG_0": 2.0}, source="synthetic.driver")
    stale = factory.stale(good.values, source="synthetic.driver", reason="read_timeout")
    simulated = factory.simulated({"TAG_0": 3.0}, source="synthetic.simulator")

    assert [frame.sequence for frame in (good, stale, simulated)] == [1, 2, 3]
    assert good.samples["TAG_0"].quality is SignalQuality.GOOD
    assert stale.samples["TAG_0"].quality is SignalQuality.STALE
    assert stale.samples["TAG_0"].source_timestamp == good.server_timestamp
    assert simulated.samples["TAG_0"].quality is SignalQuality.SIMULATED
    assert good.alarm_eligible is True
    assert stale.alarm_eligible is False


def test_factory_rejects_empty_or_malformed_tag_maps() -> None:
    factory = SignalFrameFactory(clock=lambda: NOW)
    with pytest.raises(ValueError, match="at least one"):
        factory.good({}, source="synthetic.driver")
    with pytest.raises(TypeError, match="dict"):
        factory.good([])  # type: ignore[arg-type]
