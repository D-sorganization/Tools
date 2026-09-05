"""Consumer and freshness contracts for the Tools module inventory."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from scripts.build_tools_module_inventory import (
    build_inventory,
    discover_governed_paths,
    main,
)
from scripts.tools_module_inventory_contract import (
    MODULE_INVENTORY_SCHEMA_VERSION,
    ToolsModuleInventoryError,
    load_inventory,
)
from scripts.tools_module_inventory_storage import (
    derive_index_from_shards,
    read_inventory,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manuals" / "tools" / "manifests" / "module-inventory.json"
SCHEMA = ROOT / "manuals" / "tools" / "schemas" / "module-inventory.schema.json"
SHARD_SCHEMA = (
    ROOT / "manuals" / "tools" / "schemas" / "module-inventory-shard.schema.json"
)


@lru_cache(maxsize=1)
def _index() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@lru_cache(maxsize=1)
def _payload() -> dict[str, Any]:
    payload = read_inventory(ROOT, MANIFEST)
    assert isinstance(payload, dict)
    return payload


def _entry(path: str) -> dict[str, Any]:
    return next(item for item in _payload()["entries"] if item["path"] == path)


def _with_first_entry(payload: dict[str, Any], **updates: object) -> dict[str, Any]:
    first = {**payload["entries"][0], **updates}
    return {**payload, "entries": [first, *payload["entries"][1:]]}


def test_inventory_is_deterministic_and_fresh() -> None:
    """The checked-in registry must be exactly reproducible from tracked files."""
    assert main(["--check"]) == 0
    assert _payload() == build_inventory(ROOT)


def test_inventory_conforms_to_owned_strict_schema() -> None:
    """Consumers can validate the Tools extension without producer imports."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    shard_schema = json.loads(SHARD_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator.check_schema(shard_schema)
    registry = Registry().with_resources(
        [
            (schema["$id"], Resource.from_contents(schema)),
            (shard_schema["$id"], Resource.from_contents(shard_schema)),
        ]
    )
    Draft202012Validator(schema, registry=registry).validate(_index())
    shard_validator = Draft202012Validator(shard_schema, registry=registry)
    for descriptor in _index()["shards"]:
        shard = json.loads((ROOT / descriptor["path"]).read_text(encoding="utf-8"))
        shard_validator.validate(shard)


def test_index_and_shards_stay_below_repository_file_budget() -> None:
    paths = [MANIFEST, *[ROOT / item["path"] for item in _index()["shards"]]]
    assert all(path.stat().st_size < 1_000_000 for path in paths)
    assert sum(item["entry_count"] for item in _index()["shards"]) == len(
        _payload()["entries"]
    )


def test_inventory_denominator_covers_every_governed_module() -> None:
    """No tracked implementation module can silently evade classification."""
    paths = [item["path"] for item in _payload()["entries"]]
    expected = [path.as_posix() for path in discover_governed_paths(ROOT)]

    assert paths == expected
    assert len(paths) > 3_000
    assert len(paths) == len(set(paths))
    assert any(path.startswith("src/") for path in paths)
    assert any(path.startswith("rust_core/") for path in paths)
    assert "config/design_manual_governance.json" in paths
    assert not any(path.startswith("tests/") for path in paths)
    assert not any("/dist/" in path for path in paths)


@pytest.mark.parametrize(
    ("path", "classification", "authority_status", "owner"),
    [
        (
            "src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/friction_factors.py",
            "calculation",
            "provisional",
            "Process calculator maintainers",
        ),
        (
            "src/rate_of_closure/application/camera_preferences.py",
            "non-calculation",
            "not-applicable",
            "Rate of Closure maintainers",
        ),
        (
            "src/shared/python/signal_toolkit/calculus.py",
            "calculation",
            "provisional",
            "Signal processing maintainers",
        ),
        (
            "rust_core/data-processor-core/src/engine/bulk_io.rs",
            "non-calculation",
            "not-applicable",
            "Native kernel maintainers",
        ),
        (
            "config/design_manual_governance.json",
            "non-calculation",
            "not-applicable",
            "Repository governance maintainers",
        ),
    ],
)
def test_representative_modules_have_explicit_owned_classification(
    path: str, classification: str, authority_status: str, owner: str
) -> None:
    entry = _entry(path)
    assert entry["classification"] == classification
    assert entry["authority_status"] == authority_status
    assert entry["maintainer"] == owner


def test_every_entry_exposes_required_traceability_and_lf_integrity() -> None:
    """Inventory rows are useful even when later pathway work is unavailable."""
    for entry in _payload()["entries"]:
        assert entry["classification"] in {"calculation", "non-calculation"}
        assert entry["authority_status"] in {
            "blocked",
            "not-applicable",
            "provisional",
        }
        assert entry["review_status"] in {
            "blocked",
            "inventory-baseline",
            "review-required",
        }
        assert entry["maintainer"]
        assert len(entry["content_sha256_lf"]) == 64
        assert entry["bytes_lf"] >= 0
        assert sorted(entry["risk_tags"]) == entry["risk_tags"]
        assert len(entry["risk_tags"]) == len(set(entry["risk_tags"]))
        assert set(entry["traceability"]) == {
            "adr_paths",
            "artifact_sha256",
            "chapter_paths",
            "citation_refs",
            "equation_refs",
            "public_surfaces",
            "public_routes",
            "test_paths",
            "unit_mentions",
            "validation_paths",
        }
        for path in entry["traceability"]["test_paths"]:
            assert (ROOT / path).is_file()


def test_calculation_candidate_stays_provisional_and_unapproved() -> None:
    """Detection is an inventory signal, never equation or use approval."""
    entry = _entry(
        "src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/friction_factors.py"
    )
    assert entry["classification"] == "calculation"
    assert entry["authority_status"] == "provisional"
    assert entry["review_status"] == "review-required"
    assert entry["states"]["equation_pathway"] == "unmapped-pending-TOOLS-D4"
    assert entry["states"]["publication"] == "blocked"
    assert entry["traceability"]["public_surfaces"]
    assert entry["traceability"]["test_paths"]


def test_non_calculation_is_not_promoted_to_scientific_authority() -> None:
    entry = _entry("src/rate_of_closure/application/camera_preferences.py")
    assert entry["classification"] == "non-calculation"
    assert entry["authority_status"] == "not-applicable"
    assert entry["states"]["equation_pathway"] == "not-applicable"
    assert entry["states"]["publication"] == "blocked"


def test_consumer_loader_rejects_unknown_version_fields_and_duplicates() -> None:
    """The Python consumer boundary fails closed independently of JSON Schema."""
    payload = _payload()
    view = load_inventory(payload)
    assert view.schema_version == MODULE_INVENTORY_SCHEMA_VERSION
    assert view.module_count == len(payload["entries"])
    assert view.calculation_count > 0
    assert view.non_calculation_count > 0

    wrong_version = {**payload, "schema_version": "tools-module-inventory/2.0.0"}
    with pytest.raises(ToolsModuleInventoryError, match="schema version"):
        load_inventory(wrong_version)

    extra_field = {**payload, "alternate_authority": "private-registry"}
    with pytest.raises(ToolsModuleInventoryError, match="fields differ"):
        load_inventory(extra_field)

    duplicate = {**payload, "entries": [*payload["entries"], payload["entries"][0]]}
    with pytest.raises(ToolsModuleInventoryError, match="duplicate module"):
        load_inventory(duplicate)


def test_consumer_loader_rejects_unsafe_paths_and_invalid_hashes() -> None:
    payload = _payload()
    unsafe = _with_first_entry(payload, path="../outside.py")
    with pytest.raises(ToolsModuleInventoryError, match="normalized relative path"):
        load_inventory(unsafe)

    invalid_hash = _with_first_entry(payload, content_sha256_lf="A" * 64)
    with pytest.raises(ToolsModuleInventoryError, match="SHA-256"):
        load_inventory(invalid_hash)


def test_summary_retains_denominator_and_authority_boundaries() -> None:
    summary = _payload()["summary"]
    assert summary["module_count"] == sum(summary["classification_counts"].values())
    assert summary["module_count"] == sum(summary["authority_status_counts"].values())
    assert summary["calculation_count"] > 0
    assert summary["non_calculation_count"] > 0
    assert summary["provisional_count"] == summary["calculation_count"]
    assert summary["blocked_count"] >= 0
    assert _payload()["release_status"] == "blocked-pathways-required"
    assert _payload()["blockers"]


def test_check_mode_derives_index_when_missing_or_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import build_tools_module_inventory as inventory_module
    from scripts.tools_module_inventory_storage import check_projection

    monkeypatch.setattr(inventory_module, "build_inventory", lambda _root: _payload())
    missing = tmp_path / "module-inventory.json"
    monkeypatch.setattr(inventory_module, "OUTPUT_PATH", missing)
    # When index is missing, check mode derives and writes it from fresh shards
    assert main(["--check"]) == 0
    assert missing.is_file()

    # Fast unit checks on check_projection logic
    index: dict[str, object] = {"shards": []}
    shards: dict[Path, str] = {}
    stale_file = tmp_path / "stale.json"
    stale_file.write_text("{}\n", encoding="utf-8")

    # With allow_derivable_index=True, stale index is refreshed and check returns None
    assert (
        check_projection(
            tmp_path, stale_file, index, shards, allow_derivable_index=True
        )
        is None
    )
    assert stale_file.read_text(encoding="utf-8") != "{}\n"

    # In strict mode (allow_derivable_index=False), missing or stale index fails closed
    absent = tmp_path / "absent.json"
    assert (
        check_projection(tmp_path, absent, index, shards, allow_derivable_index=False)
        is not None
    )
    stale_payload: dict[str, object] = {"shards": [1]}
    assert (
        check_projection(
            tmp_path, stale_file, stale_payload, shards, allow_derivable_index=False
        )
        is not None
    )


def test_check_mode_fails_closed_when_shard_is_stale(tmp_path: Path) -> None:
    from scripts.tools_module_inventory_storage import check_projection

    # Shards are authoritative: if any shard is stale, check_projection must fail
    index: dict[str, object] = {"shards": []}
    shard_rel = Path("manuals/tools/manifests/module-inventory/entries-fake.json")
    shard_file = tmp_path / shard_rel
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    shard_file.write_text('{"actual": true}\n', encoding="utf-8")

    diagnostic = check_projection(
        tmp_path,
        tmp_path / "module-inventory.json",
        index,
        {shard_rel: '{"expected": true}\n'},
        allow_derivable_index=True,
    )
    assert diagnostic is not None
    assert "stale module inventory shard" in diagnostic


def test_read_inventory_derives_index_when_omitted_or_missing(tmp_path: Path) -> None:
    # 1. Calling read_inventory without index_path derives directly from shards
    payload_omitted = read_inventory(ROOT)
    assert payload_omitted["entries"] == _payload()["entries"]
    assert payload_omitted["summary"] == _payload()["summary"]

    # 2. Calling read_inventory with non-existent index_path derives from shards
    payload_missing = read_inventory(ROOT, tmp_path / "nonexistent-index.json")
    assert payload_missing["entries"] == _payload()["entries"]
    assert payload_missing["source_tree_sha256"] == _payload()["source_tree_sha256"]


def test_derive_index_from_shards_matches_checked_in_index() -> None:
    derived = derive_index_from_shards(ROOT)
    assert derived["shards"] == _index()["shards"]
    assert derived == _index()


# ---------------------------------------------------------------------------
# Package sharding (Tools #4818 / #4915): concurrent PRs must not conflict.
# ---------------------------------------------------------------------------


def _mini_entry(path: str, digest: str) -> dict[str, Any]:
    entry = dict(_payload()["entries"][0])
    entry["path"] = path
    entry["id"] = f"TOOLS-MODULE-{path.upper().replace('/', '-').replace('.', '-')}"
    entry["content_sha256_lf"] = digest
    return entry


def _mini_payload(entries: list[dict[str, Any]]) -> dict[str, Any]:
    from scripts.tools_module_inventory_storage import (
        derive_families,
        derive_source_tree_sha256,
        derive_summary,
    )

    payload = {k: v for k, v in _payload().items() if k != "entries"}
    payload["entries"] = entries
    payload["families"] = derive_families(entries)
    payload["summary"] = derive_summary(entries)
    payload["source_tree_sha256"] = derive_source_tree_sha256(entries)
    return payload


def test_shards_are_cut_by_top_level_package() -> None:
    from scripts.tools_module_inventory_storage import shard_package

    assert shard_package("src/shared/python/sidekick/ui/a.py") == (
        "src/shared/python/sidekick"
    )
    assert shard_package("src/shared/python/contracts.py") == "src/shared/python"
    assert shard_package("src/rate_of_closure/x/y.py") == "src/rate_of_closure"
    assert shard_package("rust_core/tools-core/src/lib.rs") == "rust_core/tools-core"
    assert shard_package("scripts/a.py") == "scripts"
    assert shard_package("config/x.json") == "config"
    for descriptor in _index()["shards"]:
        shard = json.loads((ROOT / descriptor["path"]).read_text(encoding="utf-8"))
        assert shard["package"] == descriptor["package"]
        package = descriptor["package"]
        assert all(
            entry["path"].startswith(package + "/")
            or (package == "root" and "/" not in entry["path"])
            for entry in shard["entries"]
        )


def test_thin_index_carries_no_whole_tree_values() -> None:
    """The keys that made every regeneration conflict are derived, not stored."""
    assert not {"summary", "source_tree_sha256", "families"} & set(_index())
    assert {"summary", "source_tree_sha256", "families"} <= set(_payload())


def test_changes_in_different_packages_touch_disjoint_shards_and_index_lines() -> None:
    """Two branches editing different packages must merge without conflict."""
    import difflib

    from scripts.tools_module_inventory_storage import _serialized, project_shards

    a = _mini_entry("src/alpha/a.py", "a" * 64)
    b = _mini_entry("src/beta/b.py", "b" * 64)
    c = _mini_entry("src/gamma/c.py", "c" * 64)
    base_index, base_shards = project_shards(_mini_payload([a, b, c]))
    left_index, left_shards = project_shards(
        _mini_payload([_mini_entry("src/alpha/a.py", "d" * 64), b, c])
    )
    right_index, right_shards = project_shards(
        _mini_payload([a, b, _mini_entry("src/gamma/c.py", "e" * 64)])
    )
    changed_left = {p for p in base_shards if left_shards[p] != base_shards[p]}
    changed_right = {p for p in base_shards if right_shards[p] != base_shards[p]}
    assert changed_left.isdisjoint(changed_right)

    def changed_lines(before: dict[str, Any], after: dict[str, Any]) -> set[int]:
        before_lines = _serialized(before).splitlines()
        after_lines = _serialized(after).splitlines()
        matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
        touched: set[int] = set()
        for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
            if tag != "equal":
                touched.update(range(i1, i2))
        return touched

    left_lines = changed_lines(base_index, left_index)
    right_lines = changed_lines(base_index, right_index)
    assert left_lines and right_lines
    # Non-adjacent hunks: git's three-way merge resolves these cleanly.
    assert min(abs(x - y) for x in left_lines for y in right_lines) > 1
