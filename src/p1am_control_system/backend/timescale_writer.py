"""TimescaleDB implementation of :class:`~historian_shipper.RemoteHistorianWriter`.

One responsibility: turn a batch of ``(timestamp, tag_name, value)`` tuples into
rows in the ``tag_sample`` hypertable as cheaply as possible.

``psycopg`` is imported lazily inside :meth:`connect` so a bench Raspberry Pi
that has never installed a Postgres driver still boots the backend. Nothing in
this module is imported at application start unless remote forwarding is
actually enabled.

Only ever touched from the shipper's worker thread, so no internal locking.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from historian_shipper import Sample

__all__ = ["TimescaleWriter", "redact_dsn"]

logger = logging.getLogger("dcs_backend.timescale_writer")

# Matches the password field of a libpq URI or key/value DSN.
_DSN_PASSWORD_URI = re.compile(r"(?<=://)([^:/@]+):([^@]*)(?=@)")
_DSN_PASSWORD_KV = re.compile(r"(password\s*=\s*)(\S+)", re.IGNORECASE)


def redact_dsn(dsn: str) -> str:
    """Return ``dsn`` with any password replaced by ``***``.

    A DSN reaches logs through startup banners, error paths, and diagnostics.
    Redaction is applied at every one of those points, so it lives here rather
    than at each call site.

    Args:
        dsn: A libpq connection string, URI or key/value form.

    Returns:
        The same string with the password obscured.

    Raises:
        TypeError: If ``dsn`` is not a string.
    """
    if not isinstance(dsn, str):
        raise TypeError(f"dsn must be a str, got {type(dsn).__name__}")
    redacted = _DSN_PASSWORD_URI.sub(r"\1:***", dsn)
    return _DSN_PASSWORD_KV.sub(r"\1***", redacted)


class TimescaleWriter:
    """Writes scan samples into a TimescaleDB hypertable.

    Tag names are resolved to the integer ``tag_definition.id`` surrogate key so
    samples carry a 4-byte reference rather than a repeated string, and so the
    asset hierarchy (area -> unit -> equipment -> tag) can be joined onto a
    sample. Unknown tags are registered on first sight.
    """

    def __init__(
        self,
        dsn: str,
        *,
        connect_timeout_s: float = 5.0,
        application_name: str = "p1am-historian-shipper",
    ) -> None:
        """Build a writer. No connection is opened until :meth:`connect`.

        Args:
            dsn: libpq connection string for the historian database.
            connect_timeout_s: Fail-fast bound on connection establishment.
            application_name: Reported in ``pg_stat_activity``.

        Raises:
            TypeError: If ``dsn`` is not a string or the timeout is not numeric.
            ValueError: If ``dsn`` is empty or the timeout is not positive.
        """
        if not isinstance(dsn, str):
            raise TypeError(f"dsn must be a str, got {type(dsn).__name__}")
        if not dsn.strip():
            raise ValueError("dsn must not be empty")
        if not isinstance(connect_timeout_s, int | float) or isinstance(
            connect_timeout_s, bool
        ):
            raise TypeError(
                "connect_timeout_s must be numeric, "
                f"got {type(connect_timeout_s).__name__}"
            )
        if connect_timeout_s <= 0:
            raise ValueError(
                f"connect_timeout_s must be positive, got {connect_timeout_s}"
            )

        self._dsn = dsn
        self._connect_timeout_s = float(connect_timeout_s)
        self._application_name = application_name
        self._conn: Any | None = None
        self._tag_ids: dict[str, int] = {}

    @property
    def safe_dsn(self) -> str:
        """The DSN with its password redacted, for logging."""
        return redact_dsn(self._dsn)

    def connect(self) -> None:
        """Open the connection and prime the tag-id cache.

        Raises:
            RuntimeError: If ``psycopg`` is not installed.
            Exception: Any driver-level connection error, for the shipper to
                treat as a retryable failure.
        """
        try:
            import psycopg  # noqa: PLC0415 - deliberate lazy import
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "psycopg is required for TimescaleDB forwarding. "
                "Install it with: pip install 'psycopg[binary]'"
            ) from exc

        self.close()
        logger.info("Connecting to plant historian at %s", self.safe_dsn)
        self._conn = psycopg.connect(
            self._dsn,
            connect_timeout=int(self._connect_timeout_s),
            application_name=self._application_name,
            autocommit=True,
        )
        self._load_tag_ids()

    def _load_tag_ids(self) -> None:
        """Populate the name -> id cache from the remote tag definitions."""
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute("SELECT name, id FROM tag_definition")
            self._tag_ids = {name: tag_id for name, tag_id in cur.fetchall()}
        logger.debug("Loaded %d tag definitions", len(self._tag_ids))

    def _resolve_tag_id(self, name: str) -> int:
        """Return the surrogate id for ``name``, registering it if unseen."""
        cached = self._tag_ids.get(name)
        if cached is not None:
            return cached

        assert self._conn is not None
        # ON CONFLICT covers the race where another shipper (or a manual insert)
        # registered the same tag between our cache miss and this statement.
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tag_definition (name)
                VALUES (%s)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                (name,),
            )
            row = cur.fetchone()
        if row is None:  # pragma: no cover - RETURNING always yields a row here
            raise RuntimeError(f"could not resolve a tag id for {name!r}")
        tag_id = int(row[0])
        self._tag_ids[name] = tag_id
        return tag_id

    def write_batch(self, samples: Sequence[Sample]) -> int:
        """Insert a batch of samples using COPY.

        Args:
            samples: Sequence of ``(timestamp, tag_name, value)``.

        Returns:
            Number of rows written.

        Raises:
            RuntimeError: If called before :meth:`connect`.
            Exception: Any driver error, for the shipper to treat as a
                retryable failure.
        """
        if self._conn is None:
            raise RuntimeError("write_batch called before connect")
        if not samples:
            return 0

        rows = [(ts, self._resolve_tag_id(name), value) for ts, name, value in samples]

        # COPY is an order of magnitude cheaper than executemany for this shape
        # and keeps the worker's round-trip count at one per batch.
        with (
            self._conn.cursor() as cur,
            cur.copy("COPY tag_sample (ts, tag_id, value) FROM STDIN") as copy,
        ):
            for row in rows:
                copy.write_row(row)
        return len(rows)

    def close(self) -> None:
        """Close the connection. Idempotent; never raises."""
        conn = self._conn
        self._conn = None
        self._tag_ids = {}
        if conn is None:
            return
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - close must not fail the caller
            logger.debug("Timescale connection close failed", exc_info=True)
