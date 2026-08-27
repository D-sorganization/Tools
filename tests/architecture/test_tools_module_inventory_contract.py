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
from scripts.tools_module_inventory_storage import read_inventory

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


def test_check_mode_rejects_missing_or_stale_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import build_tools_module_inventory as inventory_module

    missing = tmp_path / "module-inventory.json"
    monkeypatch.setattr(inventory_module, "OUTPUT_PATH", missing)
    assert main(["--check"]) == 1

    missing.write_text("{}\n", encoding="utf-8")
    assert main(["--check"]) == 1
