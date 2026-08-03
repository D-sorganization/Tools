"""F10 contracts for maintainable asset-health advisories."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from asset_health import (
    AdvisoryCode,
    AssetHealthPolicy,
    AssetHealthService,
    AssetObservation,
)


def _observation(
    at: datetime,
    value: float,
    *,
    reference: float = 10.0,
    command: bool = True,
    feedback: bool = True,
    running: bool = True,
) -> AssetObservation:
    return AssetObservation(
        observed_at=at,
        value=value,
        reference=reference,
        command=command,
        feedback=feedback,
        running=running,
    )


def test_report_detects_calibration_drift_flatline_and_mismatch_as_advisories() -> None:
    start = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    observations = tuple(
        _observation(start + timedelta(seconds=index * 10), 15.0, feedback=False)
        for index in range(7)
    )
    service = AssetHealthService(
        AssetHealthPolicy(
            drift_limit=2.0,
            flatline_duration=timedelta(seconds=30),
            flatline_span=0.01,
            mismatch_duration=timedelta(seconds=30),
            noise_standard_deviation=3.0,
        ),
        now=lambda: start + timedelta(minutes=2),
    )

    report = service.assess(
        "SYNTHETIC.FEED.PUMP",
        observations,
        calibration_due_at=start - timedelta(days=1),
    )

    assert {advisory.code for advisory in report.advisories} == {
        AdvisoryCode.CALIBRATION_DUE,
        AdvisoryCode.DRIFT,
        AdvisoryCode.FLATLINE,
        AdvisoryCode.COMMAND_FEEDBACK_MISMATCH,
    }
    assert all(
        advisory.classification == "maintenance_advisory"
        for advisory in report.advisories
    )
    assert all(advisory.authoritative_trip is False for advisory in report.advisories)
    assert report.counters.runtime_seconds == 60
    assert report.counters.start_count == 1
    assert report.statistics.sample_count == 7


def test_noisy_signal_and_device_statistics_are_reproducible() -> None:
    start = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    values = (0.0, 20.0, 0.0, 20.0, 0.0)
    observations = tuple(
        _observation(start + timedelta(seconds=index), value, reference=value)
        for index, value in enumerate(values)
    )
    service = AssetHealthService(
        AssetHealthPolicy(
            drift_limit=2.0,
            flatline_duration=timedelta(seconds=30),
            flatline_span=0.01,
            mismatch_duration=timedelta(seconds=30),
            noise_standard_deviation=5.0,
        ),
        now=lambda: start + timedelta(seconds=5),
    )

    report = service.assess(
        "SYNTHETIC.REACTOR.TEMPERATURE",
        observations,
        calibration_due_at=start + timedelta(days=1),
    )

    assert [advisory.code for advisory in report.advisories] == [
        AdvisoryCode.NOISY_SIGNAL
    ]
    assert report.statistics.minimum == 0
    assert report.statistics.maximum == 20
    assert report.statistics.mean == 8
    assert report.statistics.standard_deviation > 5


def test_start_counter_distinguishes_transitions_from_runtime() -> None:
    start = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    observations = (
        _observation(start, 10, running=False),
        _observation(start + timedelta(seconds=10), 10, running=True),
        _observation(start + timedelta(seconds=20), 10, running=True),
        _observation(start + timedelta(seconds=30), 10, running=False),
        _observation(start + timedelta(seconds=40), 10, running=True),
    )
    service = AssetHealthService(
        AssetHealthPolicy(), now=lambda: start + timedelta(seconds=40)
    )

    report = service.assess(
        "SYNTHETIC.FEED.PUMP",
        observations,
        calibration_due_at=start + timedelta(days=1),
    )

    assert report.counters.start_count == 2
    assert report.counters.runtime_seconds == 20
