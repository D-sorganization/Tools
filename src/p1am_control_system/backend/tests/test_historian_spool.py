"""Unit tests for the SpoolingSink store-and-forward decorator (issue #5541, H2).

Enforces:
1. Protocol contract: SpoolingSink implements HistorianSink.
2. Downstream failure / recovery: downstream raising -> records land in spool;
   downstream recovering -> replayed in original order exactly once.
3. Restart simulation: new instance replays an existing spool from checkpoint.
4. Resilience: corrupt/truncated final spool record skipped with logged
   warning, no crash.
5. Backpressure / safety: spool full -> drop counter increments and submit()
   latency stays bounded.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from historian import HistorianRecord
from historian_sink import HistorianSink
from historian_spool import SpoolingSink


class FlakyDownstreamSink:
    """Downstream test double that can be toggled to raise or succeed."""

    def __init__(self) -> None:
        """Initialize mock downstream sink state."""
        self.submitted: list[HistorianRecord] = []
        self.should_fail: bool = False
        self.closed: bool = False
        self._drained: int = 0

    def submit(self, record: HistorianRecord) -> bool:
        """Submit a record or simulate failure/closure."""
        if self.should_fail:
            raise ConnectionError("downstream connection lost")
        if self.closed:
            return False
        self.submitted.append(record)
        return True

    async def drain_once(self) -> int:
        if self.should_fail:
            raise ConnectionError("downstream drain failed")
        count = len(self.submitted)
        self._drained += count
        return count

    def stats(self) -> Mapping[str, int]:
        """Return diagnostic metrics mapping."""
        return {
            "submitted": len(self.submitted),
            "drained": self._drained,
        }

    def close(self) -> None:
        """Close the mock sink."""
        self.closed = True


def test_spooling_sink_satisfies_protocol(tmp_path: Path) -> None:
    """Verify SpoolingSink conforms to the HistorianSink protocol."""
    downstream = FlakyDownstreamSink()
    spool_path = tmp_path / "historian.spool"
    sink = SpoolingSink(downstream, spool_path=spool_path)

    assert isinstance(sink, HistorianSink)
    stats = sink.stats()
    assert isinstance(stats, Mapping)
    assert "dropped_records" in stats
    assert "spooled_records" in stats
    assert "replayed_records" in stats

    sink.close()
    assert downstream.closed is True
    # Idempotent close
    sink.close()


@pytest.mark.asyncio
async def test_downstream_failure_and_recovery_replay_in_order(
    tmp_path: Path,
) -> None:
    downstream = FlakyDownstreamSink()
    spool_path = tmp_path / "historian.spool"
    sink = SpoolingSink(downstream, spool_path=spool_path)

    rec1 = HistorianRecord(tags={"t1": 1.0}, events=("e1",), quality="live")
    rec2 = HistorianRecord(tags={"t2": 2.0}, events=(), quality="live")
    rec3 = HistorianRecord(tags={"t3": 3.0}, events=("e3",), quality="live")

    # Downstream healthy: rec1 goes directly to downstream
    assert sink.submit(rec1) is True
    assert downstream.submitted == [rec1]

    # Downstream fails: rec2 and rec3 land in the spool
    downstream.should_fail = True
    assert sink.submit(rec2) is True
    assert sink.submit(rec3) is True
    # Downstream has only rec1
    assert len(downstream.submitted) == 1

    # Check stats reflect spooled count
    stats = sink.stats()
    assert stats["spooled_records"] == 2
    assert stats["replayed_records"] == 0

    # Downstream recovers: drain_once replays spooled records in order
    downstream.should_fail = False
    replayed = await sink.drain_once()
    assert replayed == 2
    assert downstream.submitted == [rec1, rec2, rec3]

    # Replayed exactly once: subsequent drain_once does not duplicate
    replayed_second = await sink.drain_once()
    assert replayed_second == 0
    assert downstream.submitted == [rec1, rec2, rec3]

    sink.close()


@pytest.mark.asyncio
async def test_restart_simulation_replays_from_checkpoint(
    tmp_path: Path,
) -> None:
    downstream = FlakyDownstreamSink()
    spool_path = tmp_path / "historian.spool"
    checkpoint_path = tmp_path / "historian.checkpoint"

    # Instance 1: spools records while downstream is failing
    sink1 = SpoolingSink(
        downstream, spool_path=spool_path, checkpoint_path=checkpoint_path
    )
    downstream.should_fail = True

    rec1 = HistorianRecord(tags={"t1": 10.0}, quality="live")
    rec2 = HistorianRecord(tags={"t2": 20.0}, quality="live")
    sink1.submit(rec1)
    sink1.submit(rec2)
    sink1.close()

    # Simulate restart: create Instance 2 with recovered downstream
    downstream2 = FlakyDownstreamSink()

    sink2 = SpoolingSink(
        downstream2, spool_path=spool_path, checkpoint_path=checkpoint_path
    )
    replayed = await sink2.drain_once()
    assert replayed == 2
    assert downstream2.submitted == [rec1, rec2]

    sink2.close()


@pytest.mark.asyncio
async def test_corrupted_trailing_record_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    downstream = FlakyDownstreamSink()
    spool_path = tmp_path / "historian.spool"
    checkpoint_path = tmp_path / "historian.checkpoint"

    sink = SpoolingSink(
        downstream, spool_path=spool_path, checkpoint_path=checkpoint_path
    )
    downstream.should_fail = True

    rec1 = HistorianRecord(tags={"ok": 1.0}, quality="live")
    sink.submit(rec1)

    # Corrupt the trailing bytes in the spool file (simulating incomplete flush / crash)
    with open(spool_path, "ab") as f:
        f.write(b"\xaa\x55\x00\x00\x00\xffCORRUPT_BYTES_NO_HEADER")

    downstream.should_fail = False
    with caplog.at_level(logging.WARNING):
        replayed = await sink.drain_once()

    # rec1 was successfully parsed and replayed despite corrupt trailer
    assert replayed == 1
    assert downstream.submitted == [rec1]
    # Warning was logged about corrupt/truncated record
    assert any(
        "corrupt" in msg.lower() or "truncated" in msg.lower()
        for msg in caplog.messages
    )

    sink.close()


def test_spool_full_increments_dropped_records_with_bounded_latency(
    tmp_path: Path,
) -> None:
    """Verify submit() drops records and remains bounded in latency when full."""
    downstream = FlakyDownstreamSink()
    downstream.should_fail = True
    spool_path = tmp_path / "historian.spool"

    # Very small spool budget (e.g. 200 bytes)
    sink = SpoolingSink(downstream, spool_path=spool_path, max_bytes=200)

    rec = HistorianRecord(tags={"temperature": 75.5}, quality="live")
    # First record fits
    assert sink.submit(rec) is True

    # Fill until spool drops
    start = time.perf_counter()
    dropped = False
    for _ in range(50):
        if not sink.submit(rec):
            dropped = True
            break
    elapsed = time.perf_counter() - start

    assert dropped is True
    stats = sink.stats()
    assert stats["dropped_records"] > 0
    # submit() must remain fast and non-blocking
    assert elapsed < 0.5

    sink.close()


def test_argument_validation(tmp_path: Path) -> None:
    """Verify constructor validates types and values of all arguments."""
    downstream = FlakyDownstreamSink()
    spool_path = tmp_path / "historian.spool"

    with pytest.raises(TypeError, match="downstream must implement HistorianSink"):
        SpoolingSink(cast(Any, "not_a_sink"), spool_path=spool_path)

    with pytest.raises(TypeError, match="max_bytes must be an int"):
        SpoolingSink(downstream, spool_path=spool_path, max_bytes=cast(Any, "invalid"))

    with pytest.raises(ValueError, match="max_bytes must be >= 1"):
        SpoolingSink(downstream, spool_path=spool_path, max_bytes=0)

    with pytest.raises(TypeError, match="batch_max must be an int"):
        SpoolingSink(downstream, spool_path=spool_path, batch_max=cast(Any, True))

    with pytest.raises(ValueError, match="batch_max must be >= 1"):
        SpoolingSink(downstream, spool_path=spool_path, batch_max=0)

    with pytest.raises(TypeError, match="to_thread must be callable"):
        SpoolingSink(downstream, spool_path=spool_path, to_thread=cast(Any, "invalid"))


def test_submit_type_checking(tmp_path: Path) -> None:
    """Verify submit() rejects objects that are not HistorianRecord instances."""
    downstream = FlakyDownstreamSink()
    sink = SpoolingSink(downstream, spool_path=tmp_path / "historian.spool")

    with pytest.raises(TypeError, match="record must be a HistorianRecord"):
        sink.submit(cast(Any, "not_a_record"))

    sink.close()


@pytest.mark.asyncio
async def test_closed_sink_rejects_submit_and_drain(tmp_path: Path) -> None:
    """Verify closed sink rejects new submissions and returns 0 from drain."""
    downstream = FlakyDownstreamSink()
    sink = SpoolingSink(downstream, spool_path=tmp_path / "historian.spool")

    sink.close()
    rec = HistorianRecord(tags={"t": 1.0}, quality="live")
    assert sink.submit(rec) is False
    assert await sink.drain_once() == 0


@pytest.mark.asyncio
async def test_downstream_failure_mid_replay_preserves_checkpoint(
    tmp_path: Path,
) -> None:
    """Verify downstream failure during replay preserves checkpoint for retry."""
    downstream = FlakyDownstreamSink()
    spool_path = tmp_path / "historian.spool"
    checkpoint_path = tmp_path / "historian.checkpoint"

    sink = SpoolingSink(
        downstream, spool_path=spool_path, checkpoint_path=checkpoint_path
    )
    downstream.should_fail = True

    rec1 = HistorianRecord(tags={"t1": 1.0}, quality="live")
    rec2 = HistorianRecord(tags={"t2": 2.0}, quality="live")
    rec3 = HistorianRecord(tags={"t3": 3.0}, quality="live")

    sink.submit(rec1)
    sink.submit(rec2)
    sink.submit(rec3)

    # Downstream accepts rec1, then fails on rec2
    downstream.should_fail = False
    original_submit = downstream.submit

    def submit_fail_on_rec2(record: HistorianRecord) -> bool:
        """Helper to inject transient error on record 2."""
        if record.tags and record.tags.get("t2") is not None:
            raise ConnectionError("downstream transient failure")
        return original_submit(record)

    downstream.submit = submit_fail_on_rec2  # type: ignore[method-assign]

    replayed_first = await sink.drain_once()
    assert replayed_first == 1
    assert downstream.submitted == [rec1]

    # Downstream recovers completely
    downstream.submit = original_submit  # type: ignore[method-assign]
    replayed_second = await sink.drain_once()
    assert replayed_second == 2
    assert downstream.submitted == [rec1, rec2, rec3]

    sink.close()


def test_corrupted_checkpoint_file_handled_gracefully(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify corrupted checkpoint file content defaults to offset 0 with warning."""
    downstream = FlakyDownstreamSink()
    spool_path = tmp_path / "historian.spool"
    checkpoint_path = tmp_path / "historian.checkpoint"

    checkpoint_path.write_text("NOT_A_VALID_INTEGER\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        sink = SpoolingSink(
            downstream, spool_path=spool_path, checkpoint_path=checkpoint_path
        )

    # Starts cleanly with checkpoint 0 and logs warning
    assert sink._checkpoint == 0
    assert any("failed to read checkpoint" in msg.lower() for msg in caplog.messages)

    sink.close()


def test_historian_record_to_dict_and_from_dict_contract() -> None:
    """Verify HistorianRecord serialization roundtrip and DbC validation."""
    record = HistorianRecord(
        tags={"temp": 21.5, "pressure": 101.3},
        events=("ALARM_TRIP",),
        quality="live",
    )
    data = record.to_dict()
    assert isinstance(data, dict)
    assert data["tags"] == {"temp": 21.5, "pressure": 101.3}
    assert data["quality"] == "live"

    restored = HistorianRecord.from_dict(data)
    assert restored == record

    # DbC validation on deserialization
    with pytest.raises((ValueError, TypeError)):
        HistorianRecord.from_dict({"tags": {"bad": "not_a_float"}})

    with pytest.raises((ValueError, TypeError)):
        HistorianRecord.from_dict({"tags": {"bad": float("nan")}})

    with pytest.raises((ValueError, TypeError)):
        HistorianRecord.from_dict({"quality": ""})


@pytest.mark.asyncio
async def test_tampered_spool_payload_is_rejected_and_quarantined(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify tampered or corrupt spool files are quarantined and not executed."""
    import struct
    import zlib

    downstream = FlakyDownstreamSink()
    spool_path = tmp_path / "historian.spool"
    checkpoint_path = tmp_path / "historian.checkpoint"

    sink = SpoolingSink(
        downstream, spool_path=spool_path, checkpoint_path=checkpoint_path
    )
    downstream.should_fail = True

    rec_good = HistorianRecord(tags={"sensor": 42.0}, quality="live")
    sink.submit(rec_good)

    # Craft tampered record frame with valid magic/CRC but invalid payload
    tampered_payload = (
        b'{"tampered": true, "tags": {"bad": "injected_payload"}, "quality": ""}'
    )
    crc = zlib.crc32(tampered_payload) & 0xFFFFFFFF
    header = struct.pack(">2sII", b"\xaa\x55", len(tampered_payload), crc)
    with open(spool_path, "ab") as f:
        f.write(header + tampered_payload)

    downstream.should_fail = False
    with caplog.at_level(logging.WARNING):
        replayed = await sink.drain_once()

    # Good record replayed, tampered record NOT executed or submitted
    assert replayed == 1
    assert downstream.submitted == [rec_good]

    # Verify corrupt/tampered file was quarantined
    quarantine_files = list(tmp_path.glob("historian.spool.quarantine*"))
    assert len(quarantine_files) >= 1
    assert any("quarantin" in msg.lower() for msg in caplog.messages)

    sink.close()
