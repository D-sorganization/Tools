"""Tests for historian wiring, settings validation, and DSN redaction.

The recurring theme: a misconfigured plant historian must fail loudly at
startup, and an unconfigured one must cost nothing at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from historian_shipper import ShipperStats  # noqa: E402
from historian_sink import HistorianWriter, NullHistorianSink  # noqa: E402
from historian_wiring import build_historian_writer, shipper_stats  # noqa: E402
from settings import P1AMSettings  # noqa: E402
from timescale_writer import TimescaleWriter, redact_dsn  # noqa: E402

pytestmark = pytest.mark.unit


def _always_due() -> bool:
    return True


# ------------------------------------------------------------------ settings ---


def test_forwarding_is_off_by_default() -> None:
    """Merging this must change nothing for an existing deployment."""
    settings = P1AMSettings(_env_file=None)
    assert settings.timescale_enabled is False
    assert settings.timescale_dsn == ""


def test_enabled_without_a_dsn_is_rejected_at_startup() -> None:
    """A historian everyone believes is recording but isn't is the worst case."""
    with pytest.raises(ValueError, match="P1AM_TIMESCALE_DSN is empty"):
        P1AMSettings(_env_file=None, timescale_enabled=True, timescale_dsn="")


def test_enabled_with_whitespace_dsn_is_rejected() -> None:
    with pytest.raises(ValueError, match="P1AM_TIMESCALE_DSN is empty"):
        P1AMSettings(_env_file=None, timescale_enabled=True, timescale_dsn="   ")


def test_enabled_with_a_dsn_is_accepted() -> None:
    settings = P1AMSettings(
        _env_file=None,
        timescale_enabled=True,
        timescale_dsn="postgresql://u:p@host/db",
    )
    assert settings.timescale_enabled is True


def test_disabled_with_empty_dsn_is_fine() -> None:
    assert P1AMSettings(_env_file=None, timescale_enabled=False).timescale_dsn == ""


# ------------------------------------------------------------------- wiring ---


def test_disabled_builds_an_inert_writer_and_no_shipper() -> None:
    """Flag off must mean no thread, no queue, no driver import."""
    settings = P1AMSettings(_env_file=None, timescale_enabled=False)
    writer, shipper = build_historian_writer(_always_due, settings)

    assert isinstance(writer, HistorianWriter)
    assert isinstance(writer.sink, NullHistorianSink)
    assert shipper is None


def test_disabled_wiring_does_not_import_psycopg() -> None:
    """A bench Pi with no Postgres driver must still boot."""
    sys.modules.pop("psycopg", None)
    settings = P1AMSettings(_env_file=None, timescale_enabled=False)
    build_historian_writer(_always_due, settings)
    assert "psycopg" not in sys.modules


def test_wiring_rejects_a_non_callable_due() -> None:
    with pytest.raises(TypeError, match="due must be callable"):
        bad: Any = "nope"
        build_historian_writer(bad)


def test_stats_for_a_disabled_shipper_are_a_clean_disabled_snapshot() -> None:
    """The health surface answers the same shape whether or not forwarding is on."""
    stats = shipper_stats(None)
    assert isinstance(stats, ShipperStats)
    assert stats.enabled is False
    assert stats.connected is False
    assert stats.queue_depth == 0
    assert stats.as_dict()["enabled"] is False


# ------------------------------------------------------------ DSN redaction ---


@pytest.mark.parametrize(
    ("dsn", "must_not_contain"),
    [
        # These DSNs carry password-shaped values on purpose: stripping them is
        # the entire contract under test. The allowlist pragmas keep
        # detect-secrets from treating the fixtures as leaked credentials.
        # Kept short so line + pragma stays inside the 88-char limit; what is
        # under test is the URI/key-value shape, not the length.
        ("postgresql://u:s3cret@h:5432/db", "s3cret"),  # pragma: allowlist secret
        ("postgres://a:p%40ss@10.0.0.5/db", "p%40ss"),  # pragma: allowlist secret
        ("host=10.0.0.5 user=a password=hunter2 db=h", "hunter2"),
        ("host=10.0.0.5 PASSWORD=Hunter2 db=h", "Hunter2"),
    ],
)
def test_redaction_removes_the_password(dsn: str, must_not_contain: str) -> None:
    redacted = redact_dsn(dsn)
    assert must_not_contain not in redacted
    assert "***" in redacted


def test_redaction_preserves_the_diagnostic_parts() -> None:
    """Redaction must not destroy the host/db, or it stops being useful."""
    redacted = redact_dsn("postgresql://user:secret@plant-historian:5432/history")
    assert "plant-historian" in redacted
    assert "history" in redacted
    assert "user" in redacted
    assert "secret" not in redacted


def test_redaction_is_a_noop_without_a_password() -> None:
    dsn = "postgresql://plant-historian:5432/history"
    assert redact_dsn(dsn) == dsn


def test_redaction_rejects_non_strings() -> None:
    with pytest.raises(TypeError, match="dsn must be a str"):
        bad: Any = None
        redact_dsn(bad)


def test_writer_exposes_only_a_redacted_dsn() -> None:
    writer = TimescaleWriter("postgresql://u:topsecret@host/db")
    assert "topsecret" not in writer.safe_dsn


# --------------------------------------------------------- TimescaleWriter DbC ---


def test_writer_rejects_an_empty_dsn() -> None:
    with pytest.raises(ValueError, match="dsn must not be empty"):
        TimescaleWriter("")


def test_writer_rejects_a_non_string_dsn() -> None:
    with pytest.raises(TypeError, match="dsn must be a str"):
        bad: Any = None
        TimescaleWriter(bad)


@pytest.mark.parametrize("bad", [0, -1.0])
def test_writer_rejects_non_positive_timeout(bad: float) -> None:
    with pytest.raises(ValueError, match="connect_timeout_s must be positive"):
        TimescaleWriter("postgresql://host/db", connect_timeout_s=bad)


def test_write_batch_before_connect_is_an_error() -> None:
    writer = TimescaleWriter("postgresql://host/db")
    with pytest.raises(RuntimeError, match="before connect"):
        writer.write_batch([])


def test_close_without_connect_is_a_noop() -> None:
    TimescaleWriter("postgresql://host/db").close()
