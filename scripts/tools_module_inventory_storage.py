"""Bounded sharding and verified reassembly for the Tools module inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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

INDEX_FIELDS = (ENVELOPE_FIELDS - {"entries"}) | {"shards"}
SHARD_FIELDS = {"authority", "entries", "schema_version", "shard_index"}
SHARD_DESCRIPTOR_FIELDS = {
    "content_sha256_lf",
    "entry_count",
    "first_path",
    "last_path",
    "path",
    "shard_index",
}
SHARD_SIZE = 200
SHARD_RELATIVE_ROOT = Path("manuals/tools/manifests/module-inventory")


def _serialized(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _normalized_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def project_shards(
    payload: dict[str, object],
) -> tuple[dict[str, object], dict[Path, str]]:
    """Project one logical payload into a bounded index and deterministic shards."""
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("inventory entries must be a list")
    shard_texts: dict[Path, str] = {}
    descriptors: list[dict[str, object]] = []
    for shard_index, offset in enumerate(range(0, len(raw_entries), SHARD_SIZE)):
        entries = raw_entries[offset : offset + SHARD_SIZE]
        shard = {
            "authority": AUTHORITY,
            "entries": entries,
            "schema_version": "tools-module-inventory-shard/1.0.0",
            "shard_index": shard_index,
        }
        text = _serialized(shard)
        relative = SHARD_RELATIVE_ROOT / f"entries-{shard_index:03d}.json"
        shard_texts[relative] = text
        first = entries[0]
        last = entries[-1]
        if not isinstance(first, dict) or not isinstance(last, dict):
            raise TypeError("inventory entries must be objects")
        descriptors.append(
            {
                "content_sha256_lf": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "entry_count": len(entries),
                "first_path": first["path"],
                "last_path": last["path"],
                "path": relative.as_posix(),
                "shard_index": shard_index,
            }
        )
    index = {key: value for key, value in payload.items() if key != "entries"}
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
    entries: list[object] = []
    declared_paths: list[str] = []
    for expected_index, descriptor_value in enumerate(descriptors):
        descriptor = _object(
            descriptor_value, "shard descriptor", SHARD_DESCRIPTOR_FIELDS
        )
        if descriptor["shard_index"] != expected_index:
            raise ToolsModuleInventoryError("shard indexes must be contiguous")
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
        declared_paths.append(relative.as_posix())
        absolute = root.joinpath(*relative.parts)
        digest = hashlib.sha256(_normalized_bytes(absolute)).hexdigest()
        if digest != _sha256(descriptor["content_sha256_lf"], "shard SHA-256"):
            raise ToolsModuleInventoryError(f"shard digest differs: {relative}")
        shard = _object(_read_json(absolute), "module inventory shard", SHARD_FIELDS)
        if shard["schema_version"] != "tools-module-inventory-shard/1.0.0":
            raise ToolsModuleInventoryError(
                "module inventory shard version is unsupported"
            )
        if shard["authority"] != AUTHORITY or shard["shard_index"] != expected_index:
            raise ToolsModuleInventoryError("module inventory shard identity differs")
        shard_entries = _array(shard["entries"], "shard entries")
        if descriptor["entry_count"] != len(shard_entries) or not shard_entries:
            raise ToolsModuleInventoryError("shard entry count differs")
        first = _object(shard_entries[0], "first shard entry", ENTRY_FIELDS)
        last = _object(shard_entries[-1], "last shard entry", ENTRY_FIELDS)
        if (
            descriptor["first_path"] != first["path"]
            or descriptor["last_path"] != last["path"]
        ):
            raise ToolsModuleInventoryError("shard path range differs")
        entries.extend(shard_entries)
    if declared_paths != sorted(set(declared_paths)):
        raise ToolsModuleInventoryError("shard paths must be sorted and unique")
    shard_root = root / SHARD_RELATIVE_ROOT
    actual = sorted(
        path.relative_to(root).as_posix() for path in shard_root.glob("*.json")
    )
    if actual != declared_paths:
        raise ToolsModuleInventoryError("shard file set differs from the index")
    assembled = {key: value for key, value in index.items() if key != "shards"}
    assembled["entries"] = entries
    load_inventory(assembled)
    return assembled
