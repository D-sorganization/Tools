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

from models import DataSource, TagLog

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
    quality: str = DataSource.LIVE.value,
) -> int:
    """Bulk-insert one scan's tag samples; return the number of rows written.

    Args:
        session: An active SQLModel session. The caller owns the transaction
            (this does not commit) so tag and alarm writes share one commit.
        tags: Mapping of tag name -> value for this scan.
        timestamp: Sample time for every row; defaults to now (UTC). One shared
            timestamp keeps a scan atomic in time and avoids 32 clock reads.
        quality: Provenance stamped on every row (see ``models.DataSource``).
            The caller must not persist held or faulted scans at all — a gap is
            the truthful record of an outage (issue #4004).

    Returns:
        Number of rows inserted (0 for an empty mapping).

    Raises:
        TypeError: If ``session`` is not a Session, ``tags`` is not a dict,
            ``timestamp`` is not a datetime/None, or ``quality`` is not a str.
        ValueError: If ``quality`` is blank or any tag value is not
            finite/convertible to float.
    """
    if not isinstance(session, Session):
        raise TypeError(f"session must be a Session, got {type(session).__name__}")
    if not isinstance(tags, dict):
        raise TypeError(f"tags must be a dict, got {type(tags).__name__}")
    if timestamp is not None and not isinstance(timestamp, datetime):
        raise TypeError(f"timestamp must be a datetime or None, got {type(timestamp)}")
    if not isinstance(quality, str):
        raise TypeError(f"quality must be a str, got {type(quality).__name__}")
    if not quality.strip():
        raise ValueError("quality must be a non-empty string")

    if not tags:
        return 0

    ts = timestamp if timestamp is not None else datetime.now(UTC)

    rows = []
    for name, value in tags.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tag {name!r} has non-numeric value {value!r}") from exc
        rows.append(
            {
                "tag_name": str(name),
                "value": numeric,
                "timestamp": ts,
                "quality": str(quality),
            }
        )

    session.execute(insert(TagLog), rows)
    return len(rows)
