"""Data quality provenance tracking for the P1AM poll loop.

Turns changes in PLC data provenance into auditable EventLog entries.
"""

from __future__ import annotations

from models import DATA_SOURCE_SEVERITY, DataSource, EventLog

__all__ = ["DataQualityTracker"]


class DataQualityTracker:
    """Emits an EventLog row whenever the poll loop's data provenance changes.

    Stateful across scans (the loop owns one instance) and deliberately silent
    while the source is unchanged, so a sustained outage costs one row, not ten
    per second.
    """

    def __init__(self) -> None:
        self._source: DataSource | None = None

    @property
    def source(self) -> DataSource | None:
        """The most recently observed source, or None before the first scan."""
        return self._source

    def observe(self, source: str) -> EventLog | None:
        """Record ``source``; return an EventLog row only on a transition.

        Raises:
            TypeError: if ``source`` is not a string.
            ValueError: if ``source`` is not a known DataSource value.
        """
        if not isinstance(source, str):
            raise TypeError(f"source must be a str, got {type(source).__name__}")
        try:
            resolved = DataSource(source)
        except ValueError as exc:
            raise ValueError(f"unknown data source {source!r}") from exc
        if resolved == self._source:
            return None
        previous = self._source
        self._source = resolved
        severity = DATA_SOURCE_SEVERITY.get(resolved.value, 0)
        description = (
            f"PLC data source changed from {previous.value if previous else 'unknown'}"
            f" to {resolved.value}."
        )
        if resolved == DataSource.FAULT:
            description += (
                " Control laws, alarm evaluation and historian sampling are"
                " suspended until a live reading returns."
            )
        elif resolved == DataSource.HELD:
            description += " Last good values are displayed but not controlled on."
        return EventLog(
            event_type="DATA_QUALITY",
            description=description,
            severity=severity,
        )
