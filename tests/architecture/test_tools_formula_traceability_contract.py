"""Contract tests for full derivation-family traceability in Tools manuals."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.tools_formula_traceability_contract import (
    FormulaTraceabilityError,
    TraceabilityDocuments,
    load_traceability_documents,
    verify_formula_traceability,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FORMULA_IDS = (
    "TOOLS-DPLANE-FRAME",
    "TOOLS-DPLANE-NORMAL",
    "TOOLS-DPLANE-SPIN-LOFT",
)


def _mutated_documents(
    document_name: str,
    mutation: Callable[[dict[str, Any]], None],
) -> TraceabilityDocuments:
    documents = load_traceability_documents(REPO_ROOT)
    payload = copy.deepcopy(getattr(documents, document_name))
    mutation(payload)
    return replace(documents, **{document_name: payload})


def test_full_derivation_family_resolves_every_required_edge_bidirectionally() -> None:
    summary = verify_formula_traceability(REPO_ROOT)

    assert summary.chapter_count == 1
    assert summary.family_count == 1
    assert summary.formula_ids == EXPECTED_FORMULA_IDS
    assert summary.claim_ids == (
        "TOOLS-DPLANE-CLAIM-FRAME",
        "TOOLS-DPLANE-CLAIM-NORMAL",
        "TOOLS-DPLANE-CLAIM-SPIN-LOFT",
    )


def test_every_formula_requires_all_symbolic_and_evidence_edges() -> None:
    documents = _mutated_documents(
        "chapter_registry",
        lambda payload: payload["chapters"][0]["derivation_families"][0][
            "formulas"
        ][0].update(verification_tests=[]),
    )

    with pytest.raises(FormulaTraceabilityError, match="verification tests"):
        verify_formula_traceability(REPO_ROOT, documents)


def test_orphaned_or_renamed_formula_mapping_fails_closed() -> None:
    documents = _mutated_documents(
        "chapter_registry",
        lambda payload: payload["chapters"][0]["derivation_families"][0][
            "formulas"
        ][0].update(formula_id="TOOLS-DPLANE-RENAMED"),
    )

    with pytest.raises(FormulaTraceabilityError, match="formula IDs differ"):
        verify_formula_traceability(REPO_ROOT, documents)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "implementation_symbols",
            [
                {
                    "path": "src/shared/python/swing_sim/impact/dplane.py",
                    "symbol": "renamed_public_symbol",
                }
            ],
            "public symbol is missing",
        ),
        (
            "verification_tests",
            [
                "src/shared/python/swing_sim/impact/tests/test_dplane.py::"
                "TestDPlaneAnalyticCases::test_renamed"
            ],
            "test target is missing",
        ),
        ("citation_ids", ["TOOLS-DPLANE-MISSING-SOURCE"], "citation is missing"),
        ("worked_example_ids", ["missing-example"], "worked example is missing"),
        ("claim_ids", ["TOOLS-DPLANE-CLAIM-MISSING"], "claim is missing"),
        (
            "rendered_artifact_ids",
            ["tools-engineering-design-manual.missing"],
            "rendered artifact is missing",
        ),
    ],
)
def test_renamed_equation_edges_fail_at_the_owning_authority(
    field: str,
    replacement: list[object],
    message: str,
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        formula = payload["chapters"][0]["derivation_families"][0]["formulas"][0]
        formula[field] = replacement

    documents = _mutated_documents("chapter_registry", mutate)

    with pytest.raises(FormulaTraceabilityError, match=message):
        verify_formula_traceability(REPO_ROOT, documents)


def test_placeholder_derivation_metadata_cannot_satisfy_completeness() -> None:
    def mutate(payload: dict[str, Any]) -> None:
        family = payload["chapters"][0]["derivation_families"][0]
        family["assumptions"] = ["TBD"]

    documents = _mutated_documents("chapter_registry", mutate)

    with pytest.raises(FormulaTraceabilityError, match="placeholder"):
        verify_formula_traceability(REPO_ROOT, documents)


def test_registry_schema_requires_derivation_families() -> None:
    schema_path = (
        REPO_ROOT
        / "manuals"
        / "tools"
        / "schemas"
        / "textbook-chapter-registry.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    chapter = schema["$defs"]["chapter"]

    assert schema["$id"].endswith("textbook-chapter-registry/1.1.0.json")
    assert "derivation_families" in chapter["required"]
