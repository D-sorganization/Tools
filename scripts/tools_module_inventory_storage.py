"""Package-sharded storage and verified reassembly for the Tools module inventory.

Layout (Tools #4818 / #4915)
----------------------------
``manuals/tools/manifests/module-inventory.json`` is a *thin index*: envelope
metadata plus one descriptor per shard. Shards live under
``manuals/tools/manifests/module-inventory/entries-<package>.json`` and are cut
**by top-level package** (``src/<pkg>``, ``src/shared/python/<pkg>``,
``rust_core/<crate>``, ``scripts``, ``config``, ...), never by entry count, so a
PR that touches one package rewrites exactly one shard and one descriptor line
group in the index. The all-pairs conflict of the previous layout came from the
index carrying whole-tree values (``source_tree_sha256``, ``summary``,
``families``) that every regeneration changed; those are now *derived at load
time* from the shards and are not stored, so two PRs touching different
packages produce disjoint index hunks and merge cleanly.

The assembled consumer payload returned by :func:`read_inventory` is unchanged
(``tools-module-inventory/1.0.0``: it still carries ``entries``, ``families``,
``summary`` and ``source_tree_sha256``); only the on-disk projection changed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import cast

from scripts.tools_module_inventory_contract import (
    AUTHORITY,
    ENTRY_FIELDS,
    ENVELOPE_FIELDS,
    MODULE_INVENTORY_SCHEMA_VERSION,
    RELEASE_STATUS,
    ToolsModuleInventoryError,
    _array,
    _object,
    _safe_path,
    _sha256,
    load_inventory,
)

DERIVED_FIELDS = {"families", "source_tree_sha256", "summary"}
INDEX_FIELDS = (ENVELOPE_FIELDS - {"entries"} - DERIVED_FIELDS) | {"shards"}
SHARD_SCHEMA_VERSION = "tools-module-inventory-shard/1.1.0"
SHARD_FIELDS = {"authority", "entries", "package", "schema_version"}
SHARD_DESCRIPTOR_FIELDS = {"content_sha256_lf", "entry_count", "package", "path"}
SHARD_RELATIVE_ROOT = Path("manuals/tools/manifests/module-inventory")
ROOT_PACKAGE = "root"
SHARD_BUDGET_BYTES = 700_000
FAMILY_RATIONALE = (
    "Deterministic repository-domain and conservative calculation-signal "
    "classification."
)
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _serialized(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _normalized_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def shard_package(path: str) -> str:
    """Return the top-level package key that owns ``path``.

    ``src/shared/python/<pkg>/...`` -> ``src/shared/python/<pkg>``;
    ``src/<pkg>/...`` -> ``src/<pkg>``; ``rust_core/<crate>/...`` ->
    ``rust_core/<crate>``; anything else -> its first path component. A file
    that sits directly in one of those roots (``src/shared/python/contracts.py``)
    keys on the root itself.
    """
    parts = path.split("/")
    if len(parts) == 1:
        return ROOT_PACKAGE
    if parts[:3] == ["src", "shared", "python"]:
        depth = 4 if len(parts) > 4 else 3
        return "/".join(parts[:depth])
    if parts[0] in {"src", "rust_core"}:
        depth = 2 if len(parts) > 2 else 1
        return "/".join(parts[:depth])
    return parts[0]


def _subpackage(path: str, package: str) -> str:
    """Return the next-deeper key below ``package`` for ``path`` (or ``package``)."""
    if package == ROOT_PACKAGE:
        return package
    remainder = path[len(package) + 1 :].split("/")
    if len(remainder) <= 1:
        return package
    return f"{package}/{remainder[0]}"


def partition_entries(
    entries: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Group entries by package, splitting any group over the size budget.

    A package whose serialized shard would exceed ``SHARD_BUDGET_BYTES`` is
    cut one path component deeper (``src/rate_of_closure`` ->
    ``src/rate_of_closure/web``, ...) until every shard fits or no deeper
    component exists. The cut is a pure function of the entries, so the
    layout is deterministic and a PR touching one sub-package rewrites one
    shard.
    """
    grouped: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        grouped.setdefault(shard_package(str(entry["path"])), []).append(entry)
    result: dict[str, list[dict[str, object]]] = {}
    pending = sorted(grouped.items())
    while pending:
        package, members = pending.pop()
        if len(_serialized({"entries": members}).encode("utf-8")) <= SHARD_BUDGET_BYTES:
            result[package] = members
            continue
        deeper: dict[str, list[dict[str, object]]] = {}
        for entry in members:
            deeper.setdefault(_subpackage(str(entry["path"]), package), []).append(
                entry
            )
        if list(deeper) == [package]:
            result[package] = members
            continue
        pending.extend(sorted(deeper.items()))
    return dict(sorted(result.items()))


def shard_slug(package: str) -> str:
    """Return the stable file slug for a package key."""
    return _SLUG_PATTERN.sub("-", package.lower()).strip("-")


def derive_families(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Recompute the ``families`` block from entries (first entry per family)."""
    families: list[dict[str, object]] = []
    for family in sorted({str(entry["family"]) for entry in entries}):
        exemplar = next(entry for entry in entries if entry["family"] == family)
        families.append(
            {
                "classification": exemplar["classification"],
                "id": family,
                "maintainer": exemplar["maintainer"],
                "rationale": FAMILY_RATIONALE,
            }
        )
    return families


def derive_source_tree_sha256(entries: list[dict[str, object]]) -> str:
    """Recompute the tree authority digest (path:content_sha256_lf per line)."""
    tree_authority = "".join(
        f"{entry['path']}:{entry['content_sha256_lf']}\n" for entry in entries
    )
    return hashlib.sha256(tree_authority.encode("utf-8")).hexdigest()


def derive_summary(entries: list[dict[str, object]]) -> dict[str, object]:
    """Recompute the ``summary`` block from entries."""
    classifications = Counter(str(entry["classification"]) for entry in entries)
    authorities = Counter(str(entry["authority_status"]) for entry in entries)
    families = Counter(str(entry["family"]) for entry in entries)
    reviews = Counter(str(entry["review_status"]) for entry in entries)
    return {
        "authority_status_counts": dict(sorted(authorities.items())),
        "blocked_count": authorities["blocked"],
        "calculation_count": classifications["calculation"],
        "classification_counts": dict(sorted(classifications.items())),
        "family_counts": dict(sorted(families.items())),
        "module_count": len(entries),
        "non_calculation_count": classifications["non-calculation"],
        "provisional_count": authorities["provisional"],
        "review_status_counts": dict(sorted(reviews.items())),
    }


def project_shards(
    payload: dict[str, object],
) -> tuple[dict[str, object], dict[Path, str]]:
    """Project one logical payload into a thin index and per-package shards."""
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("inventory entries must be a list")
    typed: list[dict[str, object]] = []
    for value in raw_entries:
        if not isinstance(value, dict):
            raise TypeError("inventory entries must be objects")
        typed.append(value)
    grouped = partition_entries(typed)
    shard_texts: dict[Path, str] = {}
    descriptors: list[dict[str, object]] = []
    used_slugs: set[str] = set()
    for package in sorted(grouped):
        slug = shard_slug(package)
        if slug in used_slugs:
            raise ToolsModuleInventoryError(f"shard slug collision: {slug}")
        used_slugs.add(slug)
        entries = grouped[package]
        shard = {
            "authority": AUTHORITY,
            "entries": entries,
            "package": package,
            "schema_version": SHARD_SCHEMA_VERSION,
        }
        text = _serialized(shard)
        relative = SHARD_RELATIVE_ROOT / f"entries-{slug}.json"
        shard_texts[relative] = text
        descriptors.append(
            {
                "content_sha256_lf": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "entry_count": len(entries),
                "package": package,
                "path": relative.as_posix(),
            }
        )
    index = {
        key: value
        for key, value in payload.items()
        if key != "entries" and key not in DERIVED_FIELDS
    }
    index["shards"] = descriptors
    return index, shard_texts


def write_projection(
    root: Path,
    output_path: Path,
    index: dict[str, object],
    shards: dict[Path, str],
) -> None:
    """Write the exact generated index and shard set, removing only obsolete shards."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_serialized(index), encoding="utf-8", newline="\n")
    shard_root = root / SHARD_RELATIVE_ROOT
    shard_root.mkdir(parents=True, exist_ok=True)
    for relative, shard_text in shards.items():
        (root / relative).write_text(shard_text, encoding="utf-8", newline="\n")
    for obsolete in shard_root.glob("*.json"):
        if obsolete.relative_to(root) not in shards:
            obsolete.unlink()


def check_projection(
    root: Path,
    output_path: Path,
    index: dict[str, object],
    shards: dict[Path, str],
) -> str | None:
    """Return a deterministic stale diagnostic, or None when bytes match."""
    if not output_path.is_file() or output_path.read_text(
        encoding="utf-8"
    ) != _serialized(index):
        return f"stale or missing module inventory: {output_path.as_posix()}"
    shard_root = root / SHARD_RELATIVE_ROOT
    actual_paths = {path.relative_to(root) for path in shard_root.glob("*.json")}
    if actual_paths != set(shards):
        return "module inventory shard set is stale"
    for relative, shard_text in shards.items():
        if (root / relative).read_text(encoding="utf-8") != shard_text:
            return f"stale module inventory shard: {relative.as_posix()}"
    return None


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_inventory(root: Path, index_path: Path) -> dict[str, object]:
    """Load, verify, and assemble a sharded inventory as one consumer payload."""
    index = _object(_read_json(index_path), "module inventory index", INDEX_FIELDS)
    if index["schema_version"] != MODULE_INVENTORY_SCHEMA_VERSION:
        raise ToolsModuleInventoryError(
            "module inventory schema version is unsupported"
        )
    if index["authority"] != AUTHORITY or index["release_status"] != RELEASE_STATUS:
        raise ToolsModuleInventoryError(
            "module inventory index authority is unsupported"
        )
    descriptors = _array(index["shards"], "shards")
    if not descriptors:
        raise ToolsModuleInventoryError("module inventory index requires shards")
    entries: list[dict[str, object]] = []
    declared_paths: list[str] = []
    declared_packages: list[str] = []
    for descriptor_value in descriptors:
        descriptor = _object(
            descriptor_value, "shard descriptor", SHARD_DESCRIPTOR_FIELDS
        )
        relative = _safe_path(descriptor["path"], "shard path")
        if relative.parts[:4] != (
            "manuals",
            "tools",
            "manifests",
            "module-inventory",
        ):
            raise ToolsModuleInventoryError(
                "shard path is outside the governed directory"
            )
        package = descriptor["package"]
        if not isinstance(package, str) or not package:
            raise ToolsModuleInventoryError("shard package must be a non-empty string")
        if relative.name != f"entries-{shard_slug(package)}.json":
            raise ToolsModuleInventoryError("shard file name differs from its package")
        declared_paths.append(relative.as_posix())
        declared_packages.append(package)
        absolute = root.joinpath(*relative.parts)
        digest = hashlib.sha256(_normalized_bytes(absolute)).hexdigest()
        if digest != _sha256(descriptor["content_sha256_lf"], "shard SHA-256"):
            raise ToolsModuleInventoryError(f"shard digest differs: {relative}")
        shard = _object(_read_json(absolute), "module inventory shard", SHARD_FIELDS)
        if shard["schema_version"] != SHARD_SCHEMA_VERSION:
            raise ToolsModuleInventoryError(
                "module inventory shard version is unsupported"
            )
        if shard["authority"] != AUTHORITY or shard["package"] != package:
            raise ToolsModuleInventoryError("module inventory shard identity differs")
        shard_entries = _array(shard["entries"], "shard entries")
        if descriptor["entry_count"] != len(shard_entries) or not shard_entries:
            raise ToolsModuleInventoryError("shard entry count differs")
        for entry_value in shard_entries:
            entry = _object(entry_value, "shard entry", ENTRY_FIELDS)
            entry_path = str(entry["path"])
            if not (
                package == ROOT_PACKAGE
                and shard_package(entry_path) == ROOT_PACKAGE
                or entry_path.startswith(package + "/")
            ):
                raise ToolsModuleInventoryError(
                    f"entry {entry_path} is filed under the wrong package shard"
                )
            entries.append(entry)
    if declared_packages != sorted(set(declared_packages)):
        raise ToolsModuleInventoryError("shard packages must be sorted and unique")
    if len(declared_paths) != len(set(declared_paths)):
        raise ToolsModuleInventoryError("shard paths must be unique")
    shard_root = root / SHARD_RELATIVE_ROOT
    actual = sorted(
        path.relative_to(root).as_posix() for path in shard_root.glob("*.json")
    )
    if set(actual) != set(declared_paths):
        raise ToolsModuleInventoryError("shard file set differs from the index")
    entries.sort(key=lambda entry: str(entry["path"]))
    if sorted(partition_entries(entries)) != declared_packages:
        raise ToolsModuleInventoryError("shard partition differs from the entries")
    assembled = {key: value for key, value in index.items() if key != "shards"}
    assembled["entries"] = cast(list[object], entries)
    assembled["families"] = derive_families(entries)
    assembled["source_tree_sha256"] = derive_source_tree_sha256(entries)
    assembled["summary"] = derive_summary(entries)
    load_inventory(assembled)
    return assembled
