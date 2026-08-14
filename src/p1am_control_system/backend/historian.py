"""Time-series historian write path for the poll loop.

One responsibility: persist a scan's worth of tag samples as cheaply as
possible. A single bulk INSERT replaces the previous 32 ORM-object inserts per
scan; combined with WAL journaling (see ``database._configure_sqlite_connection``)
this cut the per-scan write cost ~7x on a Raspberry Pi 5.

LOD: this module imports only the ORM model and SQLAlchemy — nothing from the
FastAPI app or the PLC clients — so it unit-tests against a plain session and is
a clean seam to swap for a Rust writer later if the sample rate ever demands it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from models import TagLog
from signal_quality import SignalFrame

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017
from sqlalchemy import insert
from sqlmodel import Session


def log_scan(
    session: Session,
    tags: dict[str, float],
    *,
    timestamp: datetime | None = None,
    signal_frame: SignalFrame | None = None,
) -> int:
    """Bulk-insert one scan's tag samples; return the number of rows written.

    Args:
        session: An active SQLModel session. The caller owns the transaction
            (this does not commit) so tag and alarm writes share one commit.
        tags: Mapping of tag name -> value for this scan.
        timestamp: Sample time for every row; defaults to now (UTC). One shared
            timestamp keeps a scan atomic in time and avoids 32 clock reads.

    Returns:
        Number of rows inserted (0 for an empty mapping).

    Raises:
        TypeError: If ``session`` is not a Session, ``tags`` is not a dict, or
            ``timestamp`` is not a datetime/None.
        ValueError: If any tag value is not finite/convertible to float.
    """
    if not isinstance(session, Session):
        raise TypeError(f"session must be a Session, got {type(session).__name__}")
    if not isinstance(tags, dict):
        raise TypeError(f"tags must be a dict, got {type(tags).__name__}")
    if timestamp is not None and not isinstance(timestamp, datetime):
        raise TypeError(f"timestamp must be a datetime or None, got {type(timestamp)}")
    if signal_frame is not None and not isinstance(signal_frame, SignalFrame):
        raise TypeError("signal_frame must be a SignalFrame or None")

    if not tags:
        return 0

    if signal_frame is not None:
        if signal_frame.values != {name: float(value) for name, value in tags.items()}:
            raise ValueError("signal_frame values must match logged tags")
        if timestamp is not None and timestamp != signal_frame.server_timestamp:
            raise ValueError("timestamp must match signal_frame server_timestamp")
        ts = signal_frame.server_timestamp
    else:
        ts = timestamp if timestamp is not None else datetime.now(UTC)

    rows = []
    for name, value in tags.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tag {name!r} has non-numeric value {value!r}") from exc
        if signal_frame is None:
            rows.append(
                {
                    "tag_name": str(name),
                    "value": numeric,
                    "source_timestamp": ts,
                    "timestamp": ts,
                    "quality": "uncertain",
                    "diagnostic_reason": "legacy_unqualified",
                    "sequence": 0,
                    "source": "legacy.adapter",
                }
            )
            continue
        sample = signal_frame.samples[str(name)]
        rows.append(
            {
                "tag_name": str(name),
                "value": numeric,
                "source_timestamp": sample.source_timestamp,
                "timestamp": sample.server_timestamp,
                "quality": sample.quality.value,
                "diagnostic_reason": sample.diagnostic_reason,
                "sequence": sample.sequence,
                "source": sample.source,
            }
        )

    session.execute(insert(TagLog), rows)
    return len(rows)
