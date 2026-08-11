"""Private ordered composition of plotting catalog rows."""

from __future__ import annotations

from rate_of_closure.plotting._catalog_entry_types import CatalogGroup
from rate_of_closure.plotting._catalog_scalar_entries import (
    IMPACT_ROWS,
    INPUT_ROWS,
    LAUNCH_ROWS,
    METRIC_ROWS,
)
from rate_of_closure.plotting._catalog_series_entries import (
    FLIGHT_ROWS,
    KINETICS_ROWS,
    SWING_ROWS,
)
from rate_of_closure.plotting.catalog_contract import VariableSpec

_GROUPS: tuple[CatalogGroup, ...] = (
    ("input", "Input", INPUT_ROWS),
    ("swing", "Swing Sample", SWING_ROWS),
    ("kinetics", "Kinetics", KINETICS_ROWS),
    ("impact", "Impact", IMPACT_ROWS),
    ("launch", "Launch", LAUNCH_ROWS),
    ("flight", "Flight", FLIGHT_ROWS),
    ("metric", "Metric", METRIC_ROWS),
)


def build_entries() -> tuple[VariableSpec, ...]:
    """Return the complete catalog in stable display order."""
    return tuple(
        VariableSpec(
            key=f"{prefix}.{name}",
            label=label,
            unit=unit,
            category=category,
            extractor=extractor,
        )
        for prefix, category, rows in _GROUPS
        for name, label, unit, extractor in rows
    )
