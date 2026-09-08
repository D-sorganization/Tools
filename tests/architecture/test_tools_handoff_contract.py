"""Consumer contracts for TOOLS-D9 governed completion-audit handoff and maintenance."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from scripts.tools_handoff_contract import (
    HANDOFF_SCHEMA_VERSION,
    MAX_HANDOFF_LINE_BUDGET,
    REQUIRED_HANDOFF_SECTIONS,
    TRACKED_HANDOFF_PATHS,
    HandoffMaintenanceError,
    check_diff_aware_freshness,
    load_handoff_manifest,
    verify_handoff_maintenance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUAL_ROOT = REPO_ROOT / "manuals" / "tools"
MANIFEST_PATH = MANUAL_ROOT / "handoff-manifest.json"
SCHEMA_PATH = MANUAL_ROOT / "schemas" / "handoff-maintenance.schema.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_handoff_maintenance_schema_is_strict_and_manifest_conforms() -> None:
    schema = _json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    manifest_doc = _json(MANIFEST_PATH)
    Draft202012Validator(schema).validate(manifest_doc)

    assert schema["$id"].endswith("/tools/handoff-maintenance/1.0.0.json")
    assert schema["additionalProperties"] is False


def test_handoff_manifest_loader_validates_fields() -> None:
    manifest = load_handoff_manifest(_json(MANIFEST_PATH))

    assert manifest.schema_version == HANDOFF_SCHEMA_VERSION
    assert manifest.manual_id == "tools"
    assert manifest.release_status == "unapproved-handoff-verified"
    assert manifest.owner == "Tools maintainers"
    assert manifest.program["epic"] == 4707
    assert manifest.program["subepic"] == 4730
    assert manifest.repository == "D-sorganization/Tools"
    assert "github.com" in manifest.issue_url
    assert "github.com" in manifest.pr_url

    # Commit evidence
    assert len(manifest.commit_evidence.local_head_sha) == 40
    assert (
        manifest.commit_evidence.remote_head_sha is None
        or len(manifest.commit_evidence.remote_head_sha) == 40
    )
    assert len(manifest.commit_evidence.reviewed_tree_sha) == 40

    # Branch and worktree
    assert isinstance(manifest.branch_and_worktree.is_clean, bool)

    # Check evidence
    for check_name in (
        "check_design_manual_governance",
        "build_tools_module_inventory",
        "lint_tools_textbook_chapters",
        "check_tools_exemplars",
        "check_tools_calculation_freshness",
        "check_tools_manual_qa",
        "check_tools_publication_projection",
        "render_tools_design_manual",
    ):
        assert check_name in manifest.check_evidence
        rec = manifest.check_evidence[check_name]
        assert rec.status == "passed"
        assert rec.command != ""

    # Governed handoff files
    for rel_path in TRACKED_HANDOFF_PATHS:
        assert rel_path in manifest.governed_handoff_files
        record = manifest.governed_handoff_files[rel_path]
        assert len(record.sha256_lf) == 64
        assert 0 < record.line_count <= MAX_HANDOFF_LINE_BUDGET
        assert record.max_line_budget == MAX_HANDOFF_LINE_BUDGET
        assert len(record.required_sections) > 0


def test_tracked_handoff_files_line_budget_and_sections() -> None:
    for rel_path in TRACKED_HANDOFF_PATHS:
        file_path = REPO_ROOT / rel_path
        assert file_path.is_file(), f"Missing tracked handoff: {rel_path}"
        lines = file_path.read_text(encoding="utf-8").splitlines()
        line_count = len(lines)
        assert line_count <= MAX_HANDOFF_LINE_BUDGET, (
            f"{rel_path} has {line_count} lines, "
            f"exceeding limit of {MAX_HANDOFF_LINE_BUDGET}"
        )

        expected_sections = REQUIRED_HANDOFF_SECTIONS.get(rel_path, [])
        live_sections = {ln.strip() for ln in lines if ln.startswith("## ")}
        for s in expected_sections:
            assert s in live_sections, f"{rel_path} missing expected section {s}"


def test_verify_handoff_maintenance_succeeds_on_current_tree() -> None:
    manifest = verify_handoff_maintenance(REPO_ROOT)
    assert manifest.release_status == "unapproved-handoff-verified"


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (
            lambda doc: doc.update(repository="D-sorganization/Other"),
            ValidationError,
            "was expected",
        ),
        (
            lambda doc: doc["program"].update(subepic=9999),
            ValidationError,
            "was expected",
        ),
        (
            lambda doc: doc["governed_handoff_files"]["AGENT_HANDOFF.md"].update(
                sha256_lf="0" * 64
            ),
            HandoffMaintenanceError,
            "sha256 mismatch",
        ),
        (
            lambda doc: doc["governed_handoff_files"]["AGENT_HANDOFF.md"].update(
                line_count=999
            ),
            HandoffMaintenanceError,
            "line count mismatch",
        ),
    ],
)
def test_verify_handoff_maintenance_fails_closed_on_tampering(
    mutation: Any, error_type: type[Exception], message: str, tmp_path: Path
) -> None:
    manifest_doc = copy.deepcopy(_json(MANIFEST_PATH))
    mutation(manifest_doc)

    temp_manual = tmp_path / "manuals" / "tools"
    temp_manual.mkdir(parents=True, exist_ok=True)
    (temp_manual / "schemas").mkdir(parents=True, exist_ok=True)

    (temp_manual / "handoff-manifest.json").write_text(
        json.dumps(manifest_doc, indent=2), encoding="utf-8", newline="\n"
    )
    (temp_manual / "schemas" / "handoff-maintenance.schema.json").write_bytes(
        SCHEMA_PATH.read_bytes()
    )

    # Copy tracked handoffs to temp dir
    for rel_path in TRACKED_HANDOFF_PATHS:
        dest = tmp_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO_ROOT / rel_path).read_bytes())

    with pytest.raises(error_type, match=message):
        verify_handoff_maintenance(tmp_path)


def test_diff_aware_freshness_success_when_handoffs_updated() -> None:
    diff_output = (
        "src/pendulum_simulator/core.py\n"
        "manuals/tools/handoff-manifest.json\n"
        "AGENT_HANDOFF.md\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = diff_output
        mock_run.return_value = mock_proc

        assert check_diff_aware_freshness(REPO_ROOT) is True


def test_diff_aware_freshness_fails_when_governed_files_lack_handoff() -> None:
    diff_output = "src/pendulum_simulator/core.py\n"
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = diff_output
        mock_run.return_value = mock_proc

        with pytest.raises(HandoffMaintenanceError, match="Governed files changed"):
            check_diff_aware_freshness(REPO_ROOT)
