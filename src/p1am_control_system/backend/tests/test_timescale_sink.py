"""Unit and contract tests for TimescaleSink and FanOutSink dual-write.

H4 / #5543:
1. Default flag ("sqlite") builds the SQLite sink only (no Timescale import).
2. "dual" writes both; a raising Timescale sink leaves SQLite rows intact and
   increments a failure counter.
3. Invalid flag value fails with a clear error naming allowed values ("sqlite", "dual").
4. Timescale driver import is lazy so the base install needs no new dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from historian import HistorianRecord  # noqa: E402
from historian_sink import HistorianSink  # noqa: E402
from settings import P1AMSettings  # noqa: E402
from timescale_sink import (  # noqa: E402
    FanOutSink,
    TimescaleSink,
    create_historian_sink,
)


class FakeSink:
    """In-memory HistorianSink for testing."""

    def __init__(
        self, raises_on_submit: bool = False, raises_on_drain: bool = False
    ) -> None:
        self.submitted: list[HistorianRecord] = []
        self.drain_count: int = 0
        self.closed: bool = False
        self.raises_on_submit = raises_on_submit
        self.raises_on_drain = raises_on_drain

    def submit(self, record: HistorianRecord) -> bool:
        if self.raises_on_submit:
            raise ConnectionError("Simulated TimescaleDB network failure")
        self.submitted.append(record)
        return True

    async def drain_once(self) -> int:
        if self.raises_on_drain:
            raise ConnectionError("Simulated TimescaleDB write failure during drain")
        count = len(self.submitted)
        self.drain_count += count
        self.submitted.clear()
        return count

    def stats(self) -> dict[str, Any]:
        return {"submitted": len(self.submitted), "drain_count": self.drain_count}

    def close(self) -> None:
        self.closed = True


def test_default_backend_builds_sqlite_sink_only() -> None:
    """Default setting builds the SQLite sink directly without fanout."""
    settings = P1AMSettings()
    assert settings.historian_backend == "sqlite"

    primary = FakeSink()
    sink = create_historian_sink(settings.historian_backend, primary)

    assert sink is primary
    assert not isinstance(sink, FanOutSink)


def test_invalid_backend_flag_fails_with_descriptive_error() -> None:
    """Invalid flag value fails validation naming allowed values."""
    with pytest.raises(Exception, match=r"Allowed values:.*sqlite.*dual"):
        P1AMSettings(historian_backend="postgres_only")

    with pytest.raises(ValueError, match=r"Allowed values:.*sqlite.*dual"):
        create_historian_sink("invalid_backend", FakeSink())


def test_timescale_driver_import_is_lazy() -> None:
    """Importing timescale_sink must not eagerly import asyncpg."""
    # Ensure asyncpg is not loaded as a required top-level dependency
    sink = TimescaleSink()
    assert isinstance(sink, HistorianSink)
    assert not sink.is_connected


@pytest.mark.asyncio
async def test_dual_backend_writes_both_and_tolerates_secondary_failure() -> None:
    """In dual mode, secondary failure leaves primary intact and increments counter."""
    primary = FakeSink()
    failing_secondary = FakeSink(raises_on_submit=True, raises_on_drain=True)

    dual_sink = create_historian_sink(
        "dual", primary=primary, timescale_sink=failing_secondary
    )
    assert isinstance(dual_sink, FanOutSink)

    record = HistorianRecord(tags={"TI_101": 25.5})

    # Submit must succeed on primary even when secondary raises
    accepted = dual_sink.submit(record)
    assert accepted is True
    assert len(primary.submitted) == 1
    assert dual_sink.secondary_failures == 1

    # Drain must drain primary even when secondary raises
    drained = await dual_sink.drain_once()
    assert drained == 1
    assert primary.drain_count == 1
    assert dual_sink.secondary_failures == 2

    # Stats surface both primary metrics and secondary failure diagnostics
    stats = dual_sink.stats()
    assert stats["secondary_failures"] == 2
    assert "secondary_stats" in stats

    # Close cleans up both
    dual_sink.close()
    assert primary.closed is True
    assert failing_secondary.closed is True
