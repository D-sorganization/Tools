"""Bounded difference summary for large generated inventory payloads (#5432).

``assert big_dict == other_big_dict`` makes pytest build a quadratic diff of
every nested value, which stalled a CI shard for 90 minutes. This helper keeps
the comparison exactly as strict (any difference is reported) but names only
the shards and top-level keys that differ, capped at a small number.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from scripts.tools_module_inventory_storage import shard_package, shard_slug

DEFAULT_LIMIT = 20
_ENTRIES_KEY = "entries"


def _digest(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _shard_digests(payload: Mapping[str, Any]) -> dict[str, str]:
    groups: dict[str, list[Any]] = {}
    for entry in payload.get(_ENTRIES_KEY, []):
        groups.setdefault(shard_package(str(entry["path"])), []).append(entry)
    return {package: _digest(rows) for package, rows in groups.items()}


def _label(package: str) -> str:
    return f"{package} (entries-{shard_slug(package)}.json)"


def summarize_inventory_difference(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    limit: int = DEFAULT_LIMIT,
) -> str | None:
    """Return a bounded summary of how two inventory payloads differ.

    Preconditions: ``limit`` >= 1; each payload is a mapping whose optional
    ``entries`` value is a list of mappings carrying a ``path`` string.
    Postconditions: returns ``None`` if and only if the payloads are equal
    (``expected == actual``); otherwise a string listing at most ``limit``
    changed / added / removed shards (named with their ``entries-*.json``
    file) and top-level keys, plus a count of any omitted entries.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if expected == actual:
        return None

    lines: list[str] = []
    exp_shards, act_shards = _shard_digests(expected), _shard_digests(actual)
    for kind, names in (
        (
            "changed shard",
            sorted(
                p
                for p in exp_shards.keys() & act_shards.keys()
                if exp_shards[p] != act_shards[p]
            ),
        ),
        ("removed shard", sorted(exp_shards.keys() - act_shards.keys())),
        ("added shard", sorted(act_shards.keys() - exp_shards.keys())),
    ):
        lines.extend(f"{kind}: {_label(name)}" for name in names)

    keys = (expected.keys() | actual.keys()) - {_ENTRIES_KEY}
    for key in sorted(keys):
        if key not in actual:
            lines.append(f"removed top-level key: {key}")
        elif key not in expected:
            lines.append(f"added top-level key: {key}")
        elif _digest(expected[key]) != _digest(actual[key]):
            lines.append(f"changed top-level key: {key}")

    if not lines:  # equal digests but unequal values (e.g. entry order)
        lines.append("entries differ in order or in non-JSON-visible values")
    shown = lines[:limit]
    header = f"inventory is stale: {len(lines)} difference(s)"
    if len(lines) > limit:
        shown.append(f"... and {len(lines) - limit} more")
    return "\n".join([header, *shown])
