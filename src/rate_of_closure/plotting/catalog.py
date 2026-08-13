"""Stable public catalog for every plottable simulation variable.

The registry maps namespaced keys to :class:`VariableSpec` objects. Registry
data lives in the private ``_catalog_entries`` collaborator so this public API
stays focused on lookup, category filtering, and contract-checked extraction.
"""

from __future__ import annotations

import numpy as np

from rate_of_closure._contracts import ensure, require
from rate_of_closure.plotting._catalog_entries import build_entries
from rate_of_closure.plotting.catalog_contract import (
    CATEGORIES,
    DISTANCE_KEYS,
    VariableSpec,
)
from rate_of_closure.simulation.session import SimulationRun

__all__ = [
    "CATALOG",
    "CATEGORIES",
    "DISTANCE_KEYS",
    "VariableSpec",
    "catalog_keys",
    "extract",
    "variables_by_category",
]


_ENTRIES = build_entries()

#: The registry, keyed by namespaced variable key, in display order.
CATALOG: dict[str, VariableSpec] = {spec.key: spec for spec in _ENTRIES}
ensure(len(CATALOG) == len(_ENTRIES), "catalog keys must be unique")


def catalog_keys() -> tuple[str, ...]:
    """Return all catalog keys in display order."""
    return tuple(CATALOG)


def variables_by_category(category: str) -> tuple[VariableSpec, ...]:
    """Return entries for one declared category in display order."""
    require(category in CATEGORIES, "unknown category", category)
    return tuple(spec for spec in CATALOG.values() if spec.category == category)


def extract(run: SimulationRun, key: str) -> np.ndarray | float:
    """Extract one contract-checked scalar or one-dimensional series."""
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    require(key in CATALOG, f"unknown catalog key {key!r}", key)
    spec = CATALOG[key]
    value = spec.extractor(run)
    if spec.is_series:
        array: np.ndarray = np.asarray(value, dtype=float)
        ensure(array.ndim == 1, f"{key} extractor must yield a 1-D array")
        return array
    ensure(isinstance(value, float), f"{key} extractor must yield a float")
    return float(value)
