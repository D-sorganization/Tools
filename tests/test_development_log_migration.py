"""Provenance-bound repair of inherited fleet development logs (#2081)."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from shared_scripts.development_log import validate_devlog_content
from shared_scripts.development_log_migration import MigrationError, migrate

SOURCE = """# Development Log

## Active

## Shipped

### DL-0054 · First Historical Feature

- **State:** shipped
- **PR:** #7
- **Paths:** src/first.py
- **Started:** 2026-09-01
- **Last verified:** 2026-09-02 (`abcdef12`)
- **Summary:** First historical feature.

### DL-0054 · Second Historical Feature

- **State:** shipped
- **Owner:** codex
- **Issue:** https://github.com/example/repo/issues/42
- **PR:** #8
- **Paths:** src/second.py
- **Started:** 2026-09-03
- **Last verified:** 2026-09-04 (`abcdef12`)
- **Summary:** Second historical feature.

"""

ACTIVE = """## Active

### DL-#41 · Active Feature

- **State:** in_review
- **Owner:** codex
- **Issue:** #41
- **Branch:** fix/41
- **PR:** #7
- **Paths:** src/active.py
- **Started:** 2026-09-01
- **Last verified:** 2026-09-02 (SELF)
- **Summary:** Active feature.
- **Next step:** Review metadata.

"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "evidence",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return tmp_path, commit


def _plan() -> dict[str, Any]:
    blocks = SOURCE.split("### DL-0054")
    first = "### DL-0054" + blocks[1]
    second = "### DL-0054" + blocks[2]
    return {
        "schema_version": 1,
        "source_sha256": _sha(SOURCE.encode()),
        "decisions": [
            {
                "source_line": 7,
                "source_sha256": _sha(first.encode()),
                "action": "amend",
                "entry_id": "DL-#41",
                "fields": {
                    "Owner": {
                        "value": "historical-owner",
                        "evidence": "PR #7 author record",
                    },
                    "Issue": {"value": "#41", "evidence": "governing issue record"},
                },
                "rationale": "Resolve the first legacy serial from its issue.",
            },
            {
                "source_line": 16,
                "source_sha256": _sha(second.encode()),
                "action": "amend",
                "entry_id": "DL-#42",
                "fields": {
                    "Issue": {
                        "value": "#42 (https://github.com/example/repo/issues/42)",
                        "evidence": "existing issue URL",
                    }
                },
                "rationale": "Resolve the second legacy serial without losing its URL.",
            },
        ],
    }


def _active_plan(root: Path, commit: str) -> dict[str, Any]:
    evidence = root / "reviewed-metadata-receipt.txt"
    evidence.write_text(
        f"Log metadata only; feature acceptance unverified at commit {commit}\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "source_sha256": _sha(ACTIVE.encode()),
        "decisions": [
            {
                "source_line": 3,
                "source_sha256": _sha(
                    ("### DL-#41" + ACTIVE.split("### DL-#41", 1)[1]).encode()
                ),
                "action": "amend",
                "fields": {},
                "verification": {
                    "date": "2026-10-12",
                    "commit": commit,
                    "scope": "log_metadata_only",
                    "receipt_path": evidence.name,
                    "receipt_sha256": _sha(evidence.read_bytes()),
                },
                "rationale": "Review active log metadata only.",
            }
        ],
    }


def test_migration_repairs_legacy_duplicates_with_complete_raw_receipt(
    tmp_path: Path,
) -> None:
    root, commit = _repo(tmp_path)
    output, receipt = migrate(SOURCE.encode(), _plan(), root)
    assert validate_devlog_content(output.decode(), Path("DEVELOPMENT_LOG.md")) == []
    assert "### DL-#41 · First Historical Feature" in output.decode()
    assert "### DL-#42 · Second Historical Feature" in output.decode()
    assert "https://github.com/example/repo/issues/42" in output.decode()
    assert base64.b64decode(receipt["source_bytes_b64"]) == SOURCE.encode()
    assert receipt["source_sha256"] == _sha(SOURCE.encode())
    assert receipt["output_sha256"] == _sha(output)
    assert [
        base64.b64decode(item["source_bytes_b64"]) for item in receipt["decisions"]
    ] == [("### DL-0054" + block).encode() for block in SOURCE.split("### DL-0054")[1:]]


def test_migration_rejects_stale_source_and_record_hash(tmp_path: Path) -> None:
    root, commit = _repo(tmp_path)
    plan = _plan()
    with pytest.raises(MigrationError, match="source_sha256"):
        migrate(SOURCE.replace("First", "Changed", 1).encode(), plan, root)
    plan["decisions"][0]["source_sha256"] = "0" * 64
    with pytest.raises(MigrationError, match="record hash"):
        migrate(SOURCE.encode(), plan, root)


def test_migration_requires_real_reviewed_verification_receipt(tmp_path: Path) -> None:
    root, commit = _repo(tmp_path)
    plan = _active_plan(root, commit)
    first = plan["decisions"][0]
    first["verification"]["receipt_sha256"] = "0" * 64
    with pytest.raises(MigrationError, match="receipt hash"):
        migrate(ACTIVE.encode(), plan, root)
    first["verification"]["receipt_sha256"] = _sha(
        (root / "reviewed-metadata-receipt.txt").read_bytes()
    )
    first["verification"]["commit"] = "1" * 40
    with pytest.raises(MigrationError, match="commit"):
        migrate(ACTIVE.encode(), plan, root)


def test_active_metadata_review_is_visibly_scoped(tmp_path: Path) -> None:
    root, commit = _repo(tmp_path)
    output, receipt = migrate(ACTIVE.encode(), _active_plan(root, commit), root)
    assert validate_devlog_content(output.decode(), Path("DEVELOPMENT_LOG.md")) == []
    assert "log metadata only; feature acceptance unverified" in output.decode()
    assert receipt["source_sha256"] == _sha(ACTIVE.encode())


def test_terminal_entry_cannot_receive_new_metadata_verification(
    tmp_path: Path,
) -> None:
    root, commit = _repo(tmp_path)
    plan = _plan()
    proof = _active_plan(root, commit)["decisions"][0]["verification"]
    plan["decisions"][0]["verification"] = proof
    with pytest.raises(MigrationError, match="terminal"):
        migrate(SOURCE.encode(), plan, root)


def test_issue_match_requires_exact_number(tmp_path: Path) -> None:
    root, _ = _repo(tmp_path)
    plan = _plan()
    plan["decisions"][0]["fields"]["Issue"]["value"] = "#410"
    with pytest.raises(MigrationError, match="heading issue"):
        migrate(SOURCE.encode(), plan, root)


def test_migration_rejects_unscoped_and_escaped_receipts(tmp_path: Path) -> None:
    root, commit = _repo(tmp_path)
    plan = _active_plan(root, commit)
    first = plan["decisions"][0]
    first["verification"]["scope"] = "feature_validation"
    with pytest.raises(MigrationError, match="log_metadata_only"):
        migrate(ACTIVE.encode(), plan, root)
    first["verification"]["scope"] = "log_metadata_only"
    outside = tmp_path.parent / "outside-review.txt"
    outside.write_text(
        f"Log metadata only; feature acceptance unverified at {commit}\n",
        encoding="utf-8",
    )
    link = root / "escape-review.txt"
    link.symlink_to(outside)
    first["verification"]["receipt_path"] = link.name
    first["verification"]["receipt_sha256"] = _sha(outside.read_bytes())
    with pytest.raises(MigrationError, match="escapes the repository"):
        migrate(ACTIVE.encode(), plan, root)


def test_migration_rejects_incomplete_decisions_and_state_relabelling(
    tmp_path: Path,
) -> None:
    root, commit = _repo(tmp_path)
    plan = _plan()
    plan["decisions"].pop()
    with pytest.raises(MigrationError, match="lack explicit decisions"):
        migrate(SOURCE.encode(), plan, root)
    plan = _plan()
    plan["decisions"][0]["fields"]["State"] = {"value": "shipped", "evidence": "none"}
    with pytest.raises(MigrationError, match="State"):
        migrate(SOURCE.encode(), plan, root)


def test_quarantine_requires_terminal_row_and_preserves_raw_bytes(
    tmp_path: Path,
) -> None:
    root, commit = _repo(tmp_path)
    plan = _plan()
    plan["quarantine_receipt"] = "docs/development/migration-receipt.json"
    plan["decisions"][1] = {
        "source_line": 16,
        "source_sha256": plan["decisions"][1]["source_sha256"],
        "action": "quarantine",
        "rationale": "No historical verification receipt exists for this shipped row.",
    }
    output, receipt = migrate(SOURCE.encode(), plan, root)
    assert "Second Historical Feature" not in output.decode()
    assert validate_devlog_content(output.decode(), Path("DEVELOPMENT_LOG.md")) == []
    assert receipt["decisions"][1]["action"] == "quarantine"
    assert "## Quarantined Historical Records" in output.decode()
    assert "DL-0054 (original line 16)" in output.decode()
    assert "docs/development/migration-receipt.json" in output.decode()
    active = SOURCE.replace("- **State:** shipped", "- **State:** in_review", 1)
    plan["source_sha256"] = _sha(active.encode())
    with pytest.raises(MigrationError, match="record hash"):
        migrate(active.encode(), plan, root)


def test_quarantine_cannot_hide_live_work_or_omit_receipt_pointer(
    tmp_path: Path,
) -> None:
    root, commit = _repo(tmp_path)
    plan = _plan()
    second = plan["decisions"][1]
    plan["decisions"][1] = {
        "source_line": 16,
        "source_sha256": second["source_sha256"],
        "action": "quarantine",
        "rationale": "No reviewed verification receipt exists.",
    }
    with pytest.raises(MigrationError, match="quarantine_receipt"):
        migrate(SOURCE.encode(), plan, root)
    plan["quarantine_receipt"] = "docs/development/migration-receipt.json"
    active = SOURCE.replace("- **State:** shipped", "- **State:** in_review", 1)
    first = plan["decisions"][0]
    block = "### DL-0054" + active.split("### DL-0054")[1]
    first["source_sha256"] = _sha(block.encode())
    first["action"] = "quarantine"
    for key in ("entry_id", "fields", "verification"):
        first.pop(key, None)
    plan["source_sha256"] = _sha(active.encode())
    with pytest.raises(MigrationError, match="only terminal"):
        migrate(active.encode(), plan, root)


def test_migration_receipt_json_is_stable(tmp_path: Path) -> None:
    root, commit = _repo(tmp_path)
    first = migrate(SOURCE.encode(), _plan(), root)
    second = migrate(SOURCE.encode(), _plan(), root)
    assert first == second
    assert json.dumps(first[1], sort_keys=True)
