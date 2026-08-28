"""Fail-closed contract for R14.3 PyQt and React workflow parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[2]
MATRIX = ROOT / "docs/audits/rate_of_closure_r14_3_surface_parity.v1.json"

REQUIRED_CAPABILITIES = {
    "plan.pipeline-and-noise-authoring",
    "plan.localized-torque-locus-authoring",
    "plan.grouped-plan-preservation",
    "execution.analysis-policy",
    "execution.background-authority",
    "execution.progress-and-cancellation",
    "execution.stale-result-rejection",
    "analysis.scalar-summary-and-rank-correlation",
    "analysis.one-at-a-time-sensitivity",
    "analysis.linked-trial-visuals",
    "persistence.canonical-plan-round-trip",
    "persistence.durable-archive-lifecycle",
    "export.dataset-csv-and-json",
    "export.swing-traces-and-ensemble",
    "state.invalid-plan-fails-before-mutation",
    "state.unavailable-authority-is-explicit",
    "state.error-is-bounded-and-retains-valid-evidence",
}


def _document() -> dict[str, object]:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_matrix_covers_every_required_interaction_once() -> None:
    document = _document()
    assert document["schema_version"] == 1
    assert document["requirement_id"] == "R14.3"
    assert document["status"] == "verified"
    rows = document["capabilities"]
    assert isinstance(rows, list)
    ids = [row["capability_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert set(ids) == REQUIRED_CAPABILITIES


def test_required_rows_prove_both_surfaces_with_local_evidence() -> None:
    document = _document()
    for row in document["capabilities"]:
        assert row["parity_status"] in {"equivalent", "environment_bounded"}
        assert row["requirement"]
        for surface in ("pyqt", "react"):
            evidence = row[surface]
            assert evidence["behavior"]
            assert evidence["implementation_files"]
            assert evidence["test_files"]
            for relative_path in (
                *evidence["implementation_files"],
                *evidence["test_files"],
            ):
                assert (ROOT / relative_path).is_file(), (
                    row["capability_id"],
                    surface,
                    relative_path,
                )
        assert row["validation_commands"]


def test_matrix_pins_analysis_modes_and_durable_chunk_bounds() -> None:
    document = _document()
    rows = {row["capability_id"]: row for row in document["capabilities"]}
    policy = rows["execution.analysis-policy"]
    assert policy["shared_contract"]["modes"] == [
        "all_together",
        "individual",
        "both",
    ]
    durable = rows["persistence.durable-archive-lifecycle"]
    assert durable["shared_contract"]["chunk_size"] == {"minimum": 1, "maximum": 4096}
    assert "human evidence" in document["scientific_boundary"].lower()
