"""Tests for Slice 3 (Vocabulary Ratchet Test — Oracle 7) and Slices 6-7.

Verifies that all status enums ending in JobStatus or RunStatus conform to
CANONICAL_LIFECYCLE_WORDS, that the allowlist is tight, and that
AuthorityJobStatus and video_processor.JobStatus adopt the canonical lifecycle
contract without breaking wire compatibility.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from video_processor_src.api import JobStatus

from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobSnapshot,
    AuthorityJobStatus,
    transition_authority_job,
)
from shared.python.lifecycle import (
    CANONICAL_LIFECYCLE_WORDS,
    IllegalTransitionError,
    LifecycleStatus,
    is_terminal,
)

# Ratchet for legacy unmigrated status enums.
# Today both AuthorityJobStatus and video_processor.JobStatus conform,
# so this allowlist is completely empty.
LEGACY_STATUS_ENUM_ALLOWLIST: frozenset[str] = frozenset()


def _is_private_or_dunder(name: str) -> bool:
    return name.startswith("_")


def _is_member_alias(value_node: ast.AST, existing_members: set[str]) -> bool:
    return isinstance(value_node, ast.Name) and value_node.id in existing_members


def _extract_status_enums_from_ast(
    file_path: Path, content: str
) -> list[tuple[str, set[str], bool]]:
    """Extract (class_name, canonical_members, is_enum) from AST."""
    tree = ast.parse(content, filename=str(file_path))
    found: list[tuple[str, set[str], bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not (node.name.endswith("JobStatus") or node.name.endswith("RunStatus")):
            continue
        is_enum = any("Enum" in ast.unparse(b) for b in node.bases)
        canonical_members: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if not _is_private_or_dunder(target.id):
                            if not _is_member_alias(stmt.value, canonical_members):
                                canonical_members.add(target.id)
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name):
                    if not _is_private_or_dunder(stmt.target.id):
                        val = stmt.value
                        if val is None or not _is_member_alias(val, canonical_members):
                            canonical_members.add(stmt.target.id)
        found.append((node.name, canonical_members, is_enum))
    return found


def _scan_all_src_status_enums(
    src_dir: Path,
) -> dict[str, tuple[Path, set[str], bool]]:
    """Scan src/ for all classes ending in JobStatus or RunStatus."""
    discovered: dict[str, tuple[Path, set[str], bool]] = {}
    for py_file in sorted(src_dir.rglob("*.py")):
        content = py_file.read_text(encoding="utf-8")
        if "JobStatus" not in content and "RunStatus" not in content:
            continue
        for class_name, members, is_enum in _extract_status_enums_from_ast(
            py_file, content
        ):
            discovered[class_name] = (py_file, members, is_enum)
    return discovered


def test_oracle7_status_enums_use_canonical_lifecycle_words() -> None:
    """Every *JobStatus/*RunStatus enum in src/ must use canonical words."""
    src_dir = Path(__file__).resolve().parents[2] / "src"
    discovered = _scan_all_src_status_enums(src_dir)

    assert "AuthorityJobStatus" in discovered, "AuthorityJobStatus not found in src/"
    assert "JobStatus" in discovered, "video_processor.JobStatus not found in src/"

    for class_name, (file_path, members, is_enum) in discovered.items():
        if not is_enum:
            continue
        rel_path = file_path.relative_to(src_dir)
        qualified_name = f"{rel_path}:{class_name}"
        if (
            class_name in LEGACY_STATUS_ENUM_ALLOWLIST
            or qualified_name in LEGACY_STATUS_ENUM_ALLOWLIST
        ):
            continue

        non_canonical = members - CANONICAL_LIFECYCLE_WORDS
        assert not non_canonical, (
            f"Enum {class_name} in {rel_path} contains non-canonical member names: "
            f"{sorted(non_canonical)}. Canonical words allowed: "
            f"{sorted(CANONICAL_LIFECYCLE_WORDS)}"
        )


def test_oracle7_allowlist_is_tight() -> None:
    """Allowlist entries must not be compliant; resolved entries must be removed."""
    src_dir = Path(__file__).resolve().parents[2] / "src"
    discovered = _scan_all_src_status_enums(src_dir)

    for item in LEGACY_STATUS_ENUM_ALLOWLIST:
        target_name = item.split(":")[-1] if ":" in item else item
        assert target_name in discovered, (
            f"Allowlisted enum {item!r} does not exist in src/"
        )
        _, members, is_enum = discovered[target_name]
        if is_enum:
            violates = bool(members - CANONICAL_LIFECYCLE_WORDS)
            assert violates, (
                f"Allowlisted enum {item!r} has conformed to canonical words. "
                "Remove it from LEGACY_STATUS_ENUM_ALLOWLIST."
            )


def test_authority_job_status_lifecycle_adoption() -> None:
    """AuthorityJobStatus conforms to canonical words and adopts lifecycle kernel."""
    members = {m.name for m in AuthorityJobStatus}
    assert members == {
        "QUEUED",
        "RUNNING",
        "CANCEL_REQUESTED",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }
    assert members.issubset(CANONICAL_LIFECYCLE_WORDS)

    for status in AuthorityJobStatus:
        canonical_state = status.to_lifecycle()
        assert isinstance(canonical_state, LifecycleStatus)
        assert canonical_state.value == status.value
        assert status.is_terminal() == is_terminal(status)

    # Valid transitions
    assert (
        transition_authority_job(AuthorityJobStatus.QUEUED, AuthorityJobStatus.RUNNING)
        is AuthorityJobStatus.RUNNING
    )
    assert (
        transition_authority_job(
            AuthorityJobStatus.RUNNING, AuthorityJobStatus.CANCEL_REQUESTED
        )
        is AuthorityJobStatus.CANCEL_REQUESTED
    )
    assert (
        transition_authority_job(
            AuthorityJobStatus.CANCEL_REQUESTED, AuthorityJobStatus.SUCCEEDED
        )
        is AuthorityJobStatus.SUCCEEDED
    )

    # Illegal transitions raise IllegalTransitionError
    with pytest.raises(IllegalTransitionError):
        transition_authority_job(
            AuthorityJobStatus.SUCCEEDED, AuthorityJobStatus.RUNNING
        )
    with pytest.raises(IllegalTransitionError):
        transition_authority_job(
            AuthorityJobStatus.QUEUED, AuthorityJobStatus.SUCCEEDED
        )
    with pytest.raises(IllegalTransitionError):
        transition_authority_job(
            AuthorityJobStatus.CANCELLED, AuthorityJobStatus.QUEUED
        )

    # AuthorityJobSnapshot derived booleans agreement
    snapshot_ok = AuthorityJobSnapshot(
        job_id="job-01",
        job_sha256="a" * 64,
        status=AuthorityJobStatus.SUCCEEDED,
        completed=1,
        total=1,
        result_available=True,
    )
    assert snapshot_ok.done is True
    assert snapshot_ok.ok is True
    assert snapshot_ok.success is True
    assert snapshot_ok.cancelled is False

    snapshot_running = AuthorityJobSnapshot(
        job_id="job-02",
        job_sha256="a" * 64,
        status=AuthorityJobStatus.RUNNING,
        completed=0,
        total=1,
        result_available=False,
    )
    assert snapshot_running.done is False
    assert snapshot_running.ok is False
    assert snapshot_running.success is False
    assert snapshot_running.cancelled is False


def test_video_processor_job_status_lifecycle_adoption() -> None:
    """video_processor.JobStatus conforms with backward-compatible aliases."""
    canonical_members = [m.name for m in JobStatus]
    expected = ["QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]
    assert canonical_members == expected
    assert set(canonical_members).issubset(CANONICAL_LIFECYCLE_WORDS)

    # Backward-compatible aliases
    assert JobStatus.PROCESSING is JobStatus.RUNNING
    assert JobStatus.COMPLETED is JobStatus.SUCCEEDED

    # Wire values preserved
    assert JobStatus.RUNNING.value == "processing"
    assert JobStatus.PROCESSING.value == "processing"
    assert JobStatus.SUCCEEDED.value == "completed"
    assert JobStatus.COMPLETED.value == "completed"

    # Reverse lookup
    assert JobStatus("processing") is JobStatus.RUNNING
    assert JobStatus("completed") is JobStatus.SUCCEEDED

    # Lifecycle mapping and is_terminal
    assert JobStatus.RUNNING.to_lifecycle() is LifecycleStatus.RUNNING
    assert JobStatus.SUCCEEDED.to_lifecycle() is LifecycleStatus.SUCCEEDED
    assert JobStatus.RUNNING.is_terminal() is False
    assert JobStatus.SUCCEEDED.is_terminal() is True
    assert JobStatus.FAILED.is_terminal() is True
    assert JobStatus.CANCELLED.is_terminal() is True
    assert JobStatus.QUEUED.is_terminal() is False


def test_oracle7_scanner_detects_legacy_violations() -> None:
    """Scanner detects non-canonical enum member names on synthetic input."""
    bad_code = """
class LegacyWorkerRunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
"""
    results = _extract_status_enums_from_ast(Path("dummy.py"), bad_code)
    assert len(results) == 1
    class_name, members, is_enum = results[0]
    assert is_enum is True
    assert members == {"PENDING", "RUNNING", "COMPLETE"}
    assert not members.issubset(CANONICAL_LIFECYCLE_WORDS)
    assert members - CANONICAL_LIFECYCLE_WORDS == {"PENDING", "COMPLETE"}
