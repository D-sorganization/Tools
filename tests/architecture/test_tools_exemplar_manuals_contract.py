"""Consumer contracts for the TOOLS-D4 governed exemplar manuals."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from scripts.tools_exemplar_contract import (
    ExemplarContractError,
    load_exemplar_coverage,
    verify_exemplar_repository,
    verify_exemplar_traceability,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COVERAGE_PATH = REPO_ROOT / "manuals" / "tools" / "exemplar-coverage.json"
SCHEMA_PATH = (
    REPO_ROOT / "manuals" / "tools" / "schemas" / "exemplar-coverage.schema.json"
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_exemplar_schema_is_strict_versioned_and_current_document_conforms() -> None:
    schema = _json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_json(COVERAGE_PATH))

    assert schema["$id"].endswith("/tools/exemplar-coverage/1.0.0.json")
    assert schema["additionalProperties"] is False


def test_coverage_registers_one_verified_exemplar_and_one_blocked_dependency() -> None:
    coverage = load_exemplar_coverage(_json(COVERAGE_PATH))

    assert coverage.schema_version == "tools-exemplar-coverage/1.0.0"
    assert coverage.release_status == "provisional"
    assert [item.exemplar_id for item in coverage.entries] == [
        "tools.exemplar.markerless-mocap",
        "tools.exemplar.swing-rate-of-closure-dplane",
    ]
    markerless, swing = coverage.entries
    assert markerless.status == "blocked"
    assert markerless.chapter_id is None
    assert markerless.calculation_ids == ()
    assert markerless.blockers == (
        "TOOLS-M0 issue #4708 and PR #4734 are open; no markerless module is "
        "present on the exact D3 base.",
    )
    assert swing.status == "verified-unapproved"
    assert swing.chapter_id == "tools.textbook.swing-rate-of-closure-dplane"
    assert swing.calculation_ids == ("TOOLS-DPLANE-GEOMETRY",)
    assert swing.blockers == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(extra=True), "fields differ"),
        (
            lambda value: value.update(schema_version="tools-exemplar-coverage/9.0.0"),
            "schema version",
        ),
        (
            lambda value: value["entries"][0].update(status="verified-unapproved"),
            "verified exemplar",
        ),
        (
            lambda value: value["entries"][1].update(status="blocked"),
            "blocked exemplar",
        ),
        (
            lambda value: value["entries"].reverse(),
            "sorted",
        ),
    ],
)
def test_loader_rejects_drift_and_status_claims(
    mutation: Any,
    message: str,
) -> None:
    payload = copy.deepcopy(_json(COVERAGE_PATH))
    mutation(payload)

    with pytest.raises(ExemplarContractError, match=message):
        load_exemplar_coverage(payload)


def test_repository_contract_links_registry_chapter_source_tests_and_example() -> None:
    summary = verify_exemplar_repository(REPO_ROOT)

    assert summary.verified_exemplar_count == 1
    assert summary.blocked_exemplar_count == 1
    assert summary.calculation_ids == ("TOOLS-DPLANE-GEOMETRY",)
    assert summary.chapter_ids == ("tools.textbook.swing-rate-of-closure-dplane",)
    assert summary.worked_example_ids == ("square-descending",)


def test_module_inventory_projects_exemplar_chapter_equations_units_and_sources() -> (
    None
):
    manifest = _json(
        REPO_ROOT / "manuals" / "tools" / "manifests" / "module-inventory.json"
    )
    wanted = {
        "TOOLS-MODULE-SRC-RATE-OF-CLOSURE-SIMULATION-IMPACT-KINEMATICS-PY-7B1E029F3793",
        "TOOLS-MODULE-SRC-SHARED-PYTHON-SWING-SIM-IMPACT-DPLANE-PY-2B75D87AF828",
    }
    found: dict[str, dict[str, Any]] = {}
    for shard in manifest["shards"]:
        document = _json(REPO_ROOT / shard["path"])
        for entry in document["entries"]:
            if entry["id"] in wanted:
                found[entry["id"]] = entry

    assert set(found) == wanted
    for entry in found.values():
        assert entry["states"]["chapters"] == "mapped"
        assert entry["states"]["equation_pathway"] == "mapped"
        assert entry["states"]["citations"] == "mapped"
        assert entry["states"]["units"] == "mapped"
        assert entry["traceability"]["chapter_paths"] == [
            "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd"
        ]
        assert entry["traceability"]["equation_refs"] == [
            "TOOLS-DPLANE-FRAME",
            "TOOLS-DPLANE-NORMAL",
            "TOOLS-DPLANE-SPIN-LOFT",
        ]


def test_repository_contract_fails_when_a_traceability_target_is_missing() -> None:
    payload = _json(COVERAGE_PATH)
    payload["entries"][1]["source_paths"][0] = "src/missing.py"
    coverage = load_exemplar_coverage(payload)

    with pytest.raises(ExemplarContractError, match="traceability target is missing"):
        verify_exemplar_traceability(REPO_ROOT, coverage)
