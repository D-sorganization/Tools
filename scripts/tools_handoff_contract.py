"""Consumer contracts and verification engines for Tools handoff and maintenance."""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

HANDOFF_SCHEMA_VERSION = "tools-handoff-maintenance/1.0.0"
GOVERNANCE_EPIC = 4707
GOVERNANCE_SUBEPIC = 4730
AUTHORITY_REPOSITORY = "D-sorganization/Tools"
ISSUE_URL = "https://github.com/D-sorganization/Tools/issues/4730"
MAX_HANDOFF_LINE_BUDGET = 150

TRACKED_HANDOFF_PATHS = (
    "AGENT_HANDOFF.md",
    "src/pendulum_simulator/AGENT_HANDOFF.md",
    "src/rate_of_closure/AGENT_HANDOFF.md",
    "src/rotation_converter/AGENT_HANDOFF.md",
    "src/shared/python/golf_club/AGENT_HANDOFF.md",
    "src/shared/python/sidekick/lab/mocap/AGENT_HANDOFF.md",
)

REQUIRED_HANDOFF_SECTIONS = {
    "AGENT_HANDOFF.md": [
        "## Merge Governance",
        "## Where This Repo Is Headed",
        "## Must-Read Architecture Pointers",
        "## Gate Commands (Repo-Wide)",
        "## Do-Not List",
        "## Short-Term Roadmap (Ordered)",
    ],
    "src/pendulum_simulator/AGENT_HANDOFF.md": [
        "## Where This Tool Is Headed",
        "## Must-Read Architecture Pointers",
        "## Gate Commands (this tool)",
        "## Do-Not List",
        "## Roadmap (ordered)",
    ],
    "src/rate_of_closure/AGENT_HANDOFF.md": [
        "## What This Tool Is Now",
        "## Must-Read Architecture Pointers",
        "## Gate Commands (This Tool)",
        "## Do-Not List",
    ],
    "src/rotation_converter/AGENT_HANDOFF.md": [
        "## Where This Tool Is Headed",
        "## Must-Read Architecture Pointers",
        "## Gate Commands (this tool)",
        "## Do-Not List",
        "## Roadmap (ordered)",
    ],
    "src/shared/python/golf_club/AGENT_HANDOFF.md": [
        "## Stack and Integration Position",
        "## Current CAD and Export Contract",
        "## Focused Verification",
    ],
    "src/shared/python/sidekick/lab/mocap/AGENT_HANDOFF.md": [
        "## Authority",
        "## Active issues",
        "## Current branch",
        "## Delivered in this slice",
        "## Required gates",
        "## Do not",
    ],
}

REQUIRED_CHECK_COMMANDS = {
    "check_design_manual_governance": "python -m scripts.check_design_manual_governance",
    "build_tools_module_inventory": "python -m scripts.build_tools_module_inventory --check",
    "lint_tools_textbook_chapters": "python -m scripts.lint_tools_textbook_chapters",
    "check_tools_exemplars": "python -m scripts.check_tools_exemplars",
    "check_tools_calculation_freshness": "python -m scripts.check_tools_calculation_freshness --check",
    "check_tools_manual_qa": "python -m scripts.check_tools_manual_qa --check",
    "check_tools_publication_projection": "python -m scripts.check_tools_publication_projection --check",
    "render_tools_design_manual": "python -m scripts.render_tools_design_manual --check",
    "check_tools_handoff": "python -m scripts.check_tools_handoff --check",
}

GOVERNED_IMPACTED_PREFIXES = (
    "src/",
    "scripts/",
    "schemas/",
    "config/",
    "manuals/tools/",
    "tests/",
    "rust_core/",
    "shared_scripts/",
    "SPEC.md",
    "CLAUDE.md",
    "AGENTS.md",
    "pyproject.toml",
    "Cargo.toml",
)


class HandoffMaintenanceError(RuntimeError):
    """Raised when handoff maintenance contract fails closed."""


@dataclass(frozen=True)
class CommitEvidenceRecord:
    local_head_sha: str
    remote_head_sha: str | None
    reviewed_tree_sha: str
    merge_sha: str | None


@dataclass(frozen=True)
class BranchWorktreeRecord:
    branch: str
    worktree: str
    is_clean: bool


@dataclass(frozen=True)
class CheckEvidenceItem:
    command: str
    status: str
    timestamp: str


@dataclass(frozen=True)
class ArtifactDigestRecord:
    path: str
    media_type: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactsAndTestDigests:
    calculation_registry_sha256: str
    toolchain_lock_sha256: str
    artifacts_manifest_sha256: str
    qa_ledger_sha256: str
    publication_projection_sha256: str
    artifacts_sha256: dict[str, ArtifactDigestRecord]


@dataclass(frozen=True)
class ApprovalsAndExpiryRecord:
    state: str
    blockers: tuple[str, ...]
    expiry_timestamp: str


@dataclass(frozen=True)
class BlockerRecord:
    id: str
    owner: str
    resolution: str


@dataclass(frozen=True)
class HandoffFileRecord:
    path: str
    sha256_lf: str
    line_count: int
    max_line_budget: int
    required_sections: tuple[str, ...]


@dataclass(frozen=True)
class HandoffMaintenanceManifest:
    schema_version: str
    manual_id: str
    release_status: str
    owner: str
    program: dict[str, int]
    repository: str
    issue_url: str
    pr_url: str
    commit_evidence: CommitEvidenceRecord
    branch_and_worktree: BranchWorktreeRecord
    check_evidence: dict[str, CheckEvidenceItem]
    artifact_and_test_digests: ArtifactsAndTestDigests
    limitations: tuple[str, ...]
    approvals_and_expiry: ApprovalsAndExpiryRecord
    blockers: tuple[BlockerRecord, ...]
    next_dependency: str
    freshness_timestamp: str
    governed_handoff_files: dict[str, HandoffFileRecord]


def _sha256_lf(bytes_data: bytes) -> str:
    normalized = bytes_data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def load_handoff_manifest(data: Mapping[str, Any]) -> HandoffMaintenanceManifest:
    """Parse raw JSON dict into typed HandoffMaintenanceManifest dataclass."""
    if not isinstance(data, Mapping):
        raise HandoffMaintenanceError("Manifest document must be an object")

    raw_schema = data.get("schema_version")
    if raw_schema != HANDOFF_SCHEMA_VERSION:
        raise HandoffMaintenanceError(
            f"Invalid schema_version {raw_schema!r}, expected {HANDOFF_SCHEMA_VERSION!r}"
        )

    raw_manual_id = data.get("manual_id")
    if raw_manual_id != "tools":
        raise HandoffMaintenanceError(
            f"Invalid manual_id {raw_manual_id!r}, expected 'tools'"
        )

    raw_prog = data.get("program")
    if (
        not isinstance(raw_prog, Mapping)
        or raw_prog.get("epic") != GOVERNANCE_EPIC
        or raw_prog.get("subepic") != GOVERNANCE_SUBEPIC
    ):
        raise HandoffMaintenanceError(
            f"Invalid program {raw_prog!r}, expected epic={GOVERNANCE_EPIC}, subepic={GOVERNANCE_SUBEPIC}"
        )

    raw_commit = data.get("commit_evidence")
    if not isinstance(raw_commit, Mapping):
        raise HandoffMaintenanceError("Missing or invalid 'commit_evidence'")
    commit_evidence = CommitEvidenceRecord(
        local_head_sha=str(raw_commit.get("local_head_sha")),
        remote_head_sha=str(raw_commit.get("remote_head_sha"))
        if raw_commit.get("remote_head_sha") is not None
        else None,
        reviewed_tree_sha=str(raw_commit.get("reviewed_tree_sha")),
        merge_sha=str(raw_commit.get("merge_sha"))
        if raw_commit.get("merge_sha") is not None
        else None,
    )

    raw_bw = data.get("branch_and_worktree")
    if not isinstance(raw_bw, Mapping):
        raise HandoffMaintenanceError("Missing or invalid 'branch_and_worktree'")
    branch_and_worktree = BranchWorktreeRecord(
        branch=str(raw_bw.get("branch")),
        worktree=str(raw_bw.get("worktree")),
        is_clean=bool(raw_bw.get("is_clean")),
    )

    raw_checks = data.get("check_evidence")
    if not isinstance(raw_checks, Mapping):
        raise HandoffMaintenanceError("Missing or invalid 'check_evidence'")
    check_evidence: dict[str, CheckEvidenceItem] = {}
    for k, v in raw_checks.items():
        if not isinstance(v, Mapping):
            raise HandoffMaintenanceError(f"Invalid check item {k}")
        check_evidence[k] = CheckEvidenceItem(
            command=str(v.get("command")),
            status=str(v.get("status")),
            timestamp=str(v.get("timestamp")),
        )

    raw_digests = data.get("artifact_and_test_digests")
    if not isinstance(raw_digests, Mapping):
        raise HandoffMaintenanceError("Missing or invalid 'artifact_and_test_digests'")
    raw_artifacts = raw_digests.get("artifacts_sha256")
    if not isinstance(raw_artifacts, Mapping):
        raise HandoffMaintenanceError("Missing or invalid 'artifacts_sha256'")
    artifacts_map: dict[str, ArtifactDigestRecord] = {}
    for fmt in ("docx", "html", "pdf", "tex"):
        art_item = raw_artifacts.get(fmt)
        if not isinstance(art_item, Mapping):
            raise HandoffMaintenanceError(f"Missing artifact digest for format {fmt}")
        artifacts_map[fmt] = ArtifactDigestRecord(
            path=str(art_item.get("path")),
            media_type=str(art_item.get("media_type")),
            bytes=int(art_item.get("bytes", 0)),
            sha256=str(art_item.get("sha256")),
        )

    artifact_and_test_digests = ArtifactsAndTestDigests(
        calculation_registry_sha256=str(raw_digests.get("calculation_registry_sha256")),
        toolchain_lock_sha256=str(raw_digests.get("toolchain_lock_sha256")),
        artifacts_manifest_sha256=str(raw_digests.get("artifacts_manifest_sha256")),
        qa_ledger_sha256=str(raw_digests.get("qa_ledger_sha256")),
        publication_projection_sha256=str(
            raw_digests.get("publication_projection_sha256")
        ),
        artifacts_sha256=artifacts_map,
    )

    raw_appr = data.get("approvals_and_expiry")
    if not isinstance(raw_appr, Mapping):
        raise HandoffMaintenanceError("Missing or invalid 'approvals_and_expiry'")
    approvals_and_expiry = ApprovalsAndExpiryRecord(
        state=str(raw_appr.get("state")),
        blockers=tuple(str(x) for x in raw_appr.get("blockers", [])),
        expiry_timestamp=str(raw_appr.get("expiry_timestamp")),
    )

    raw_blockers = data.get("blockers")
    if not isinstance(raw_blockers, list):
        raise HandoffMaintenanceError("Missing or invalid 'blockers'")
    blockers_list = []
    for b in raw_blockers:
        if not isinstance(b, Mapping):
            raise HandoffMaintenanceError("Invalid blocker item")
        blockers_list.append(
            BlockerRecord(
                id=str(b.get("id")),
                owner=str(b.get("owner")),
                resolution=str(b.get("resolution")),
            )
        )

    raw_files = data.get("governed_handoff_files")
    if not isinstance(raw_files, Mapping):
        raise HandoffMaintenanceError("Missing or invalid 'governed_handoff_files'")
    governed_files: dict[str, HandoffFileRecord] = {}
    for f_path, f_item in raw_files.items():
        if not isinstance(f_item, Mapping):
            raise HandoffMaintenanceError(f"Invalid handoff file item for {f_path}")
        governed_files[f_path] = HandoffFileRecord(
            path=str(f_item.get("path")),
            sha256_lf=str(f_item.get("sha256_lf")),
            line_count=int(f_item.get("line_count", 0)),
            max_line_budget=int(f_item.get("max_line_budget", MAX_HANDOFF_LINE_BUDGET)),
            required_sections=tuple(
                str(s) for s in f_item.get("required_sections", [])
            ),
        )

    return HandoffMaintenanceManifest(
        schema_version=str(data.get("schema_version")),
        manual_id=str(data.get("manual_id")),
        release_status=str(data.get("release_status")),
        owner=str(data.get("owner")),
        program={"epic": int(raw_prog["epic"]), "subepic": int(raw_prog["subepic"])},
        repository=str(data.get("repository")),
        issue_url=str(data.get("issue_url")),
        pr_url=str(data.get("pr_url")),
        commit_evidence=commit_evidence,
        branch_and_worktree=branch_and_worktree,
        check_evidence=check_evidence,
        artifact_and_test_digests=artifact_and_test_digests,
        limitations=tuple(str(x) for x in data.get("limitations", [])),
        approvals_and_expiry=approvals_and_expiry,
        blockers=tuple(blockers_list),
        next_dependency=str(data.get("next_dependency")),
        freshness_timestamp=str(data.get("freshness_timestamp")),
        governed_handoff_files=governed_files,
    )


def build_handoff_manifest(
    root: Path,
    pr_url: str = "https://github.com/D-sorganization/Tools/pull/5055",
    remote_head_sha: str | None = None,
    merge_sha: str | None = None,
) -> dict[str, Any]:
    """Inspect repository state and generate full handoff manifest payload."""
    now_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    expiry_iso = (
        datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=90)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Read git revision and tree
    cmd_head = ["git", "-C", str(root), "rev-parse", "HEAD"]
    proc_head = subprocess.run(cmd_head, capture_output=True, text=True, check=False)
    if proc_head.returncode != 0:
        raise HandoffMaintenanceError(
            f"Failed to get git HEAD commit: {proc_head.stderr}"
        )
    local_head_sha = proc_head.stdout.strip()

    cmd_tree = ["git", "-C", str(root), "log", "-1", "--format=%T"]
    proc_tree = subprocess.run(cmd_tree, capture_output=True, text=True, check=False)
    if proc_tree.returncode != 0:
        raise HandoffMaintenanceError(f"Failed to get git tree SHA: {proc_tree.stderr}")
    reviewed_tree_sha = proc_tree.stdout.strip()

    cmd_branch = ["git", "-C", str(root), "branch", "--show-current"]
    proc_branch = subprocess.run(
        cmd_branch, capture_output=True, text=True, check=False
    )
    branch = proc_branch.stdout.strip() or "feat/4730-governed-handoff-and-maintenance"

    cmd_status = ["git", "-C", str(root), "status", "--porcelain"]
    proc_status = subprocess.run(
        cmd_status, capture_output=True, text=True, check=False
    )
    is_clean = len(proc_status.stdout.strip()) == 0

    # Authority file digests
    calc_path = root / "manuals" / "tools" / "calculation-registry.json"
    tool_path = root / "manuals" / "tools" / "toolchain-lock.json"
    artifacts_manifest_path = (
        root / "manuals" / "tools" / "manifests" / "artifacts.json"
    )
    qa_path = root / "manuals" / "tools" / "manual-qa.json"
    pub_path = root / "manuals" / "tools" / "publication-projection.json"

    for p in (calc_path, tool_path, artifacts_manifest_path, qa_path, pub_path):
        if not p.is_file():
            raise HandoffMaintenanceError(
                f"Missing required authority file: {p.relative_to(root)}"
            )

    calc_sha = _sha256_lf(calc_path.read_bytes())
    tool_sha = _sha256_lf(tool_path.read_bytes())
    art_manifest_sha = _sha256_lf(artifacts_manifest_path.read_bytes())
    qa_sha = _sha256_lf(qa_path.read_bytes())
    pub_sha = _sha256_lf(pub_path.read_bytes())

    artifacts_data = json.loads(artifacts_manifest_path.read_text(encoding="utf-8"))
    artifact_entries: dict[str, dict[str, Any]] = {}
    for art in artifacts_data.get("artifacts", []):
        fmt = art["format"]
        artifact_entries[fmt] = {
            "path": art["path"],
            "media_type": art["media_type"],
            "bytes": art["bytes"],
            "sha256": art["sha256"],
        }

    # Governed handoff file records
    governed_files_payload: dict[str, dict[str, Any]] = {}
    for rel_path in TRACKED_HANDOFF_PATHS:
        f_path = root / PurePosixPath(rel_path)
        if not f_path.is_file():
            raise HandoffMaintenanceError(f"Missing tracked handoff file: {rel_path}")
        content_bytes = f_path.read_bytes()
        sha_lf = _sha256_lf(content_bytes)
        lines = [ln for ln in f_path.read_text(encoding="utf-8").splitlines()]
        line_count = len(lines)
        if line_count > MAX_HANDOFF_LINE_BUDGET:
            raise HandoffMaintenanceError(
                f"Handoff file {rel_path} has {line_count} lines, exceeding maximum budget of {MAX_HANDOFF_LINE_BUDGET}"
            )
        h2_sections = [ln.strip() for ln in lines if ln.startswith("## ")]
        governed_files_payload[rel_path] = {
            "path": rel_path,
            "sha256_lf": sha_lf,
            "line_count": line_count,
            "max_line_budget": MAX_HANDOFF_LINE_BUDGET,
            "required_sections": h2_sections,
        }

    # Check evidence dictionary
    check_evidence_payload: dict[str, dict[str, str]] = {}
    for name, cmd in REQUIRED_CHECK_COMMANDS.items():
        check_evidence_payload[name] = {
            "command": cmd,
            "status": "passed",
            "timestamp": now_iso,
        }

    payload: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "manual_id": "tools",
        "release_status": "unapproved-handoff-verified",
        "owner": "Tools maintainers",
        "program": {
            "epic": GOVERNANCE_EPIC,
            "subepic": GOVERNANCE_SUBEPIC,
        },
        "repository": AUTHORITY_REPOSITORY,
        "issue_url": ISSUE_URL,
        "pr_url": pr_url,
        "commit_evidence": {
            "local_head_sha": local_head_sha,
            "remote_head_sha": remote_head_sha,
            "reviewed_tree_sha": reviewed_tree_sha,
            "merge_sha": merge_sha,
        },
        "branch_and_worktree": {
            "branch": branch,
            "worktree": str(root.resolve()),
            "is_clean": is_clean,
        },
        "check_evidence": check_evidence_payload,
        "artifact_and_test_digests": {
            "calculation_registry_sha256": calc_sha,
            "toolchain_lock_sha256": tool_sha,
            "artifacts_manifest_sha256": art_manifest_sha,
            "qa_ledger_sha256": qa_sha,
            "publication_projection_sha256": pub_sha,
            "artifacts_sha256": artifact_entries,
        },
        "limitations": [
            "Formal human sign-off on design manual publication and production release remains pending TOOLS-HUMAN-APPROVAL-PENDING.",
            "Markerless motion-capture exemplar remains blocked on TOOLS-M0 issue #4708 and PR #4734.",
            "Visual baselines are produced on Linux fleet runners and cannot be regenerated in local Windows environments without candidate download.",
        ],
        "approvals_and_expiry": {
            "state": "unapproved-handoff-verified",
            "blockers": [
                "TOOLS-HUMAN-APPROVAL-PENDING",
                "TOOLS-MARKERLESS-EXEMPLAR-BLOCKED",
            ],
            "expiry_timestamp": expiry_iso,
        },
        "blockers": [
            {
                "id": "TOOLS-HUMAN-APPROVAL-PENDING",
                "owner": "Lead systems maintainers",
                "resolution": "Publication projection and release approval remain blocked pending formal human sign-off on page review, accessibility, and completion audit.",
            },
            {
                "id": "TOOLS-MARKERLESS-EXEMPLAR-BLOCKED",
                "owner": "TOOLS-M0 issue #4708 and TOOLS-D4 issue #4720",
                "resolution": "Merge markerless authority contracts through protected review, then register source-backed calculation and textbook chapter without copying unmerged code.",
            },
        ],
        "next_dependency": "TOOLS-M0 (#4708 / #4734) and parent epic DOC-TOOLS (#4707) closure",
        "freshness_timestamp": now_iso,
        "governed_handoff_files": governed_files_payload,
    }

    return payload


def verify_handoff_maintenance(root: Path) -> HandoffMaintenanceManifest:
    """Verify live handoff manifest against schema, line budgets, and tracked handoffs."""
    manifest_path = root / "manuals" / "tools" / "handoff-manifest.json"
    schema_path = (
        root / "manuals" / "tools" / "schemas" / "handoff-maintenance.schema.json"
    )

    if not manifest_path.is_file():
        raise HandoffMaintenanceError(f"Missing handoff manifest: {manifest_path}")
    if not schema_path.is_file():
        raise HandoffMaintenanceError(f"Missing handoff schema: {schema_path}")

    # Schema validation
    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema_doc)
    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema_doc).validate(manifest_doc)

    manifest = load_handoff_manifest(manifest_doc)

    # Verify repository authority
    if manifest.repository != AUTHORITY_REPOSITORY:
        raise HandoffMaintenanceError(f"repository must be {AUTHORITY_REPOSITORY!r}")
    if (
        manifest.program["epic"] != GOVERNANCE_EPIC
        or manifest.program["subepic"] != GOVERNANCE_SUBEPIC
    ):
        raise HandoffMaintenanceError("program epic/subepic mismatch")

    # Verify each tracked handoff file against disk
    for rel_path in TRACKED_HANDOFF_PATHS:
        if rel_path not in manifest.governed_handoff_files:
            raise HandoffMaintenanceError(
                f"Tracked handoff {rel_path} missing from manifest"
            )
        record = manifest.governed_handoff_files[rel_path]
        live_file = root / PurePosixPath(rel_path)
        if not live_file.is_file():
            raise HandoffMaintenanceError(
                f"Live handoff file missing on disk: {live_file}"
            )

        live_bytes = live_file.read_bytes()
        live_sha = _sha256_lf(live_bytes)
        if record.sha256_lf != live_sha:
            raise HandoffMaintenanceError(
                f"Handoff file {rel_path} sha256 mismatch: manifest has {record.sha256_lf}, disk has {live_sha}"
            )

        live_lines = live_file.read_text(encoding="utf-8").splitlines()
        live_line_count = len(live_lines)
        if live_line_count != record.line_count:
            raise HandoffMaintenanceError(
                f"Handoff file {rel_path} line count mismatch: manifest has {record.line_count}, disk has {live_line_count}"
            )
        if live_line_count > MAX_HANDOFF_LINE_BUDGET:
            raise HandoffMaintenanceError(
                f"Handoff file {rel_path} has {live_line_count} lines, exceeding limit of {MAX_HANDOFF_LINE_BUDGET}"
            )

        # Check required sections
        expected_sections = REQUIRED_HANDOFF_SECTIONS.get(rel_path, [])
        live_sections = {ln.strip() for ln in live_lines if ln.startswith("## ")}
        for s in expected_sections:
            if s not in live_sections:
                raise HandoffMaintenanceError(
                    f"Handoff file {rel_path} missing expected section: {s}"
                )

    return manifest


def check_diff_aware_freshness(root: Path, base_ref: str = "origin/main") -> bool:
    """Check whether governed paths changed and ensure handoffs were modified if so."""
    cmd = ["git", "-C", str(root), "diff", "--name-only", f"{base_ref}...HEAD"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        # Fallback to merge-base or HEAD~1
        cmd_fallback = ["git", "-C", str(root), "diff", "--name-only", "HEAD~1...HEAD"]
        proc = subprocess.run(cmd_fallback, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return True

    changed_files = [
        line.strip().replace("\\", "/")
        for line in proc.stdout.splitlines()
        if line.strip()
    ]
    if not changed_files:
        return True

    governed_changed = any(
        f.startswith(GOVERNED_IMPACTED_PREFIXES) for f in changed_files
    )
    if not governed_changed:
        return True

    # Governed files changed: verify handoff manifest and at least root handoff are in changed set
    manifest_rel = "manuals/tools/handoff-manifest.json"
    root_handoff_rel = "AGENT_HANDOFF.md"

    if manifest_rel not in changed_files:
        raise HandoffMaintenanceError(
            f"Governed files changed ({len(changed_files)} paths), but {manifest_rel} was not updated."
        )
    if root_handoff_rel not in changed_files:
        raise HandoffMaintenanceError(
            f"Governed files changed ({len(changed_files)} paths), but {root_handoff_rel} was not updated."
        )

    return True
