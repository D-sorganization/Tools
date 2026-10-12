#!/usr/bin/env python3
"""Build a reviewed, provenance-bound repair of an inherited development log.

The source file is never edited. Every changed or removed entry is anchored to
its exact source line and byte hash; the receipt retains the complete source.
Only an explicit, locally resolvable commit plus a hash-pinned validation
receipt can supply a new ``Last verified`` value. This tool does not run or
qualify the feature described by an entry.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from shared_scripts.changes_fragment_collate import _set_fields
from shared_scripts.development_log import (
    ENTRY_HEADING,
    TERMINAL_STATES,
    parse_entries,
    validate_devlog_content,
)

SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
ISSUE_ID = re.compile(r"^DL-#(\d+)$")
ALLOWED_FIELDS = frozenset(
    {"Owner", "Issue", "Paths", "Started", "Summary", "Next step", "Branch", "Parked"}
)


class MigrationError(ValueError):
    """An explicit migration decision or its provenance is invalid."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nonempty(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise MigrationError(f"{name} must be a nonempty single line")
    return value.strip()


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise MigrationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _record_spans(lines: list[str]) -> dict[int, tuple[int, int]]:
    """Map original one-based heading lines to complete entry line spans."""
    starts = [
        index
        for index, line in enumerate(lines)
        if ENTRY_HEADING.match(line.rstrip("\r\n"))
    ]
    spans: dict[int, tuple[int, int]] = {}
    for start in starts:
        end = start + 1
        while end < len(lines) and not lines[end].startswith("#"):
            end += 1
        spans[start + 1] = (start, end)
    return spans


def _verification(value: Any, repo_root: Path) -> str:
    if not isinstance(value, dict) or set(value) != {
        "date",
        "commit",
        "scope",
        "receipt_path",
        "receipt_sha256",
    }:
        raise MigrationError("verification needs date, commit, scope and receipt")
    if value["scope"] != "log_metadata_only":
        raise MigrationError("migration verification scope must be log_metadata_only")
    day = _nonempty(value["date"], "verification date")
    try:
        date.fromisoformat(day)
    except ValueError as exc:
        raise MigrationError("verification date is not ISO-8601") from exc
    commit = _nonempty(value["commit"], "verification commit")
    if not COMMIT.fullmatch(commit):
        raise MigrationError("verification commit is not a Git SHA")
    result = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise MigrationError("verification commit is not locally resolvable")
    relative = Path(_nonempty(value["receipt_path"], "receipt_path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise MigrationError(
            "verification receipt_path must stay inside the repository"
        )
    receipt = (repo_root / relative).resolve()
    if not receipt.is_relative_to(repo_root.resolve()):
        raise MigrationError("verification receipt_path escapes the repository")
    try:
        content = receipt.read_bytes()
    except OSError as exc:
        raise MigrationError("verification receipt is not readable") from exc
    if _sha(content) != _hash(value["receipt_sha256"], "receipt_sha256"):
        raise MigrationError("verification receipt hash does not match")
    if commit.encode() not in content:
        raise MigrationError("verification receipt does not bind the commit")
    if b"log metadata only; feature acceptance unverified" not in content.lower():
        raise MigrationError("verification receipt lacks metadata-only scope")
    return (
        f"{day} (`{commit}`; log metadata only; feature acceptance unverified; "
        f"reviewed receipt `{relative.as_posix()}`)"
    )


def _amend(block: list[str], decision: dict[str, Any], repo_root: Path) -> list[str]:
    entry = parse_entries("".join(block))[0]
    heading_id = decision.get("entry_id", entry.entry_id)
    if not isinstance(heading_id, str) or not ISSUE_ID.fullmatch(heading_id):
        raise MigrationError("amended entry_id must be issue-keyed (DL-#N)")
    fields = decision.get("fields", {})
    if not isinstance(fields, dict):
        raise MigrationError("fields must be a mapping")
    updates: dict[str, str] = {}
    for key, change in fields.items():
        if key not in ALLOWED_FIELDS:
            raise MigrationError(f"field {key!r} cannot be changed by migration")
        if not isinstance(change, dict) or set(change) != {"value", "evidence"}:
            raise MigrationError(f"field {key!r} needs value and evidence")
        updates[key] = _nonempty(change["value"], key)
        _nonempty(change["evidence"], f"{key} evidence")
    if "verification" in decision:
        if entry.state in TERMINAL_STATES:
            raise MigrationError(
                "terminal history cannot receive new metadata-only verification"
            )
        updates["Last verified"] = _verification(decision["verification"], repo_root)
    if heading_id != entry.entry_id:
        match = ENTRY_HEADING.match(block[0].rstrip("\r\n"))
        assert match is not None
        block = [f"### {heading_id} · {match.group(2)}\n", *block[1:]]
    issue = updates.get("Issue", entry.fields.get("Issue", ""))
    issue_match = ISSUE_ID.fullmatch(heading_id)
    assert issue_match is not None
    issue_number = issue_match.group(1)
    if not re.search(rf"(?<!\d)#{issue_number}(?!\d)", issue):
        raise MigrationError("Issue field must name the amended heading issue")
    return _set_fields(block, updates)


def migrate(
    source_bytes: bytes, plan: dict[str, Any], repo_root: Path
) -> tuple[bytes, dict[str, Any]]:
    """Return a strictly valid candidate and an audit receipt; never write input.

    Every decision addresses one original record by line and exact byte hash.
    Unverified terminal history may be quarantined in the receipt, but live
    work may not be hidden. A candidate with any validator finding is refused.
    """
    if (
        not isinstance(plan, dict)
        or not {"schema_version", "source_sha256", "decisions"} <= set(plan)
        or set(plan)
        - {"schema_version", "source_sha256", "decisions", "quarantine_receipt"}
        or plan["schema_version"] != 1
    ):
        raise MigrationError("plan must use schema_version 1 and exact required keys")
    if _sha(source_bytes) != _hash(plan["source_sha256"], "source_sha256"):
        raise MigrationError("source_sha256 does not match source bytes")
    if not isinstance(plan["decisions"], list):
        raise MigrationError("decisions must be a list")
    try:
        lines = source_bytes.decode("utf-8").splitlines(keepends=True)
    except UnicodeError as exc:
        raise MigrationError("source must be UTF-8") from exc
    spans = _record_spans(lines)
    replacements: dict[int, tuple[int, list[str]]] = {}
    audited: list[dict[str, Any]] = []
    quarantined: list[tuple[str, int]] = []
    for decision in plan["decisions"]:
        if not isinstance(decision, dict):
            raise MigrationError("each decision must be an object")
        line = decision.get("source_line")
        if type(line) is not int or line not in spans or line in replacements:
            raise MigrationError("source_line must name one unused original entry")
        start, end = spans[line]
        block = lines[start:end]
        raw = "".join(block).encode("utf-8")
        if _sha(raw) != _hash(decision.get("source_sha256"), "record hash"):
            raise MigrationError(f"record hash mismatch at source_line {line}")
        action = decision.get("action")
        _nonempty(decision.get("rationale"), "rationale")
        if action == "amend":
            if set(decision) - {
                "source_line",
                "source_sha256",
                "action",
                "entry_id",
                "fields",
                "verification",
                "rationale",
            }:
                raise MigrationError("unknown amend decision key")
            rendered = _amend(block, decision, repo_root)
        elif action == "quarantine":
            if set(decision) != {"source_line", "source_sha256", "action", "rationale"}:
                raise MigrationError("quarantine accepts only its anchored rationale")
            entry = parse_entries("".join(block))[0]
            if entry.state not in TERMINAL_STATES:
                raise MigrationError("only terminal historical rows may be quarantined")
            quarantined.append((entry.entry_id, line))
            rendered = []
        else:
            raise MigrationError("action must be amend or quarantine")
        replacements[line] = (end, rendered)
        audited.append(
            {
                "source_line": line,
                "source_sha256": _sha(raw),
                "source_bytes_b64": base64.b64encode(raw).decode("ascii"),
                "action": action,
                "decision": decision,
                "output_sha256": _sha("".join(rendered).encode()),
            }
        )
    source_findings = validate_devlog_content(
        source_bytes.decode("utf-8"), Path("DEVELOPMENT_LOG.md")
    )
    undecided = sorted(
        {finding.line for finding in source_findings if finding.line is not None}
        - replacements.keys()
    )
    if undecided:
        raise MigrationError(
            f"source findings lack explicit decisions at lines {undecided}"
        )
    candidate: list[str] = []
    index = 0
    while index < len(lines):
        replacement = replacements.get(index + 1)
        if replacement is None:
            candidate.append(lines[index])
            index += 1
        else:
            end, rendered = replacement
            candidate.extend(rendered)
            index = end
    output = "".join(candidate).encode("utf-8")
    if quarantined:
        pointer = plan.get("quarantine_receipt")
        if (
            not isinstance(pointer, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", pointer)
            or ".." in Path(pointer).parts
        ):
            raise MigrationError(
                "quarantine_receipt must name a safe repository-relative path"
            )
        notice = (
            "\n## Quarantined Historical Records\n\n"
            "These terminal records remain unverified. Their exact original bytes "
            f"and decisions are preserved in `{pointer}`; quarantine does not "
            "qualify their features or change issue status.\n\n"
            + "".join(
                f"- {entry_id} (original line {line}): unresolved historical record.\n"
                for entry_id, line in quarantined
            )
        )
        output += notice.encode("utf-8")
    findings = validate_devlog_content(output.decode(), Path("DEVELOPMENT_LOG.md"))
    if findings:
        first = findings[0]
        raise MigrationError(
            f"candidate still has {len(findings)} development-log findings: "
            f"{first.kind} at line {first.line}"
        )
    receipt = {
        "schema_version": 1,
        "source_sha256": _sha(source_bytes),
        "source_bytes_b64": base64.b64encode(source_bytes).decode("ascii"),
        "plan_sha256": _sha(json.dumps(plan, sort_keys=True).encode()),
        "output_sha256": _sha(output),
        "decisions": audited,
    }
    return output, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.bundle_dir.exists():
        parser.error("bundle-dir must not already exist")
    if args.source.resolve().is_relative_to(args.bundle_dir.resolve()):
        parser.error("source must remain outside bundle-dir")
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        output, receipt = migrate(args.source.read_bytes(), plan, args.repo_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".devlog-migration-", dir=args.bundle_dir.parent)
    )
    try:
        (temporary / "DEVELOPMENT_LOG.candidate.md").write_bytes(output)
        (temporary / "migration-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.bundle_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI exercised separately
    raise SystemExit(main())
