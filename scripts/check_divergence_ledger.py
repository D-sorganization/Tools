#!/usr/bin/env python3
"""Shadowed-module divergence ledger: paired-PR gate, renderer and freshness check.

The ledger ``docs/shared/divergence_ledger.v1.json`` is the single place the
Tools <-> UpstreamDrift shadowed-module rulings live (Tools #4915, supersedes
#4496; D1-D31 rulings formerly a prose row in ``AGENT_HANDOFF.md``).

Modes::

    python scripts/check_divergence_ledger.py            # paired-PR gate (CI)
    python scripts/check_divergence_ledger.py --check    # schema + rendered markdown fresh
    python scripts/check_divergence_ledger.py --render   # rewrite docs/shared/divergence_ledger.md

Gate rule
---------
A PR whose diff (``git diff --name-only origin/main...HEAD``) touches a
ledgered ``tools_path`` must carry ``UD-PAIR: D-sorganization/UpstreamDrift#N``
in its body (read from ``GITHUB_EVENT_PATH``) unless the matching row's
``ruling`` is ``tools-canonical`` or its ``status`` is ``ud-copy-deleted``.
Outside a pull_request event (no event file / no PR body) the gate reports and
exits 0, so ``push`` runs never fail on it. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "docs" / "shared" / "divergence_ledger.v1.json"
RENDER_PATH = ROOT / "docs" / "shared" / "divergence_ledger.md"
SCHEMA_VERSION = "divergence-ledger/1.0.0"
RULINGS = frozenset({"tools-canonical", "ud-canonical", "split", "deferred"})
STATUSES = frozenset(
    {
        "ported",
        "paired-open",
        "pinned",
        "in-sync",
        "tools-only",
        "ud-copy-deleted",
        "pending-inventory",
    }
)
ROW_FIELDS = frozenset(
    {
        "module",
        "tools_path",
        "ud_path",
        "ruling",
        "owner",
        "source_issue",
        "target_pr",
        "status",
        "rulings",
        "inventory",
        "notes",
    }
)
EXEMPT_RULINGS = frozenset({"tools-canonical"})
EXEMPT_STATUSES = frozenset({"ud-copy-deleted"})
PAIR_PATTERN = re.compile(r"UD-PAIR:\s*D-sorganization/UpstreamDrift#(\d+)")


class LedgerError(ValueError):
    """Raised when the ledger violates its own contract."""


def load_ledger(path: Path = LEDGER_PATH) -> dict[str, object]:
    """Load and validate the ledger document."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise LedgerError("ledger must be a JSON object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise LedgerError("ledger schema_version is unsupported")
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise LedgerError("ledger requires a non-empty rows array")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise LedgerError(f"row {index} must be an object")
        actual = set(row)
        if actual != ROW_FIELDS:
            raise LedgerError(
                f"row {index} fields differ: missing={sorted(ROW_FIELDS - actual)}, "
                f"extra={sorted(actual - ROW_FIELDS)}"
            )
        module = row["module"]
        if not isinstance(module, str) or not module or module in seen:
            raise LedgerError(f"row {index} module must be a unique non-empty string")
        seen.add(module)
        tools_path = row["tools_path"]
        if not isinstance(tools_path, str) or not tools_path:
            raise LedgerError(f"{module}: tools_path must be a non-empty string")
        posix = PurePosixPath(tools_path)
        if posix.is_absolute() or ".." in posix.parts or posix.as_posix() != tools_path:
            raise LedgerError(
                f"{module}: tools_path must be a normalized relative path"
            )
        if row["ruling"] not in RULINGS:
            raise LedgerError(f"{module}: ruling {row['ruling']!r} is unsupported")
        if row["status"] not in STATUSES:
            raise LedgerError(f"{module}: status {row['status']!r} is unsupported")
        if not isinstance(row["rulings"], list):
            raise LedgerError(f"{module}: rulings must be a list")
        if row["ud_path"] is not None and not isinstance(row["ud_path"], str):
            raise LedgerError(f"{module}: ud_path must be text or null")
    return document


def rows_of(document: dict[str, object]) -> list[dict[str, object]]:
    rows = document["rows"]
    assert isinstance(rows, list)
    return [row for row in rows if isinstance(row, dict)]


def _covers(tools_path: str, changed: str) -> bool:
    return changed == tools_path or changed.startswith(tools_path.rstrip("/") + "/")


def touched_rows(
    document: dict[str, object], changed_files: Iterable[str]
) -> dict[str, list[dict[str, object]]]:
    """Map each changed file to the ledger rows whose tools_path covers it."""
    rows = rows_of(document)
    hits: dict[str, list[dict[str, object]]] = {}
    for changed in changed_files:
        matches = [row for row in rows if _covers(str(row["tools_path"]), changed)]
        if matches:
            # The most specific (longest) path wins; package rows are fallbacks.
            matches.sort(key=lambda row: len(str(row["tools_path"])), reverse=True)
            hits[changed] = matches
    return hits


def rows_requiring_pair(
    hits: dict[str, list[dict[str, object]]],
) -> dict[str, dict[str, object]]:
    """Return changed_file -> governing row for rows that are not exempt."""
    required: dict[str, dict[str, object]] = {}
    for changed, matches in hits.items():
        governing = matches[0]
        if (
            governing["ruling"] in EXEMPT_RULINGS
            or governing["status"] in EXEMPT_STATUSES
        ):
            continue
        required[changed] = governing
    return required


def paired_pr(body: str | None) -> str | None:
    """Return the UpstreamDrift PR number named by ``UD-PAIR:``, if any."""
    if not body:
        return None
    match = PAIR_PATTERN.search(body)
    return match.group(1) if match else None


def changed_files(root: Path = ROOT, base: str = "origin/main") -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def pr_body_from_event(event_path: str | None) -> str | None:
    """Read the pull request body from the GitHub event payload, if present."""
    if not event_path or not Path(event_path).is_file():
        return None
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pull = event.get("pull_request") if isinstance(event, dict) else None
    if not isinstance(pull, dict):
        return None
    body = pull.get("body")
    return body if isinstance(body, str) else ""


def gate(
    document: dict[str, object], changed: Sequence[str], body: str | None
) -> tuple[int, list[str]]:
    """Evaluate the paired-PR rule; return (exit code, report lines)."""
    hits = touched_rows(document, changed)
    if not hits:
        return 0, ["divergence ledger: no ledgered module touched"]
    required = rows_requiring_pair(hits)
    lines = [f"divergence ledger: {len(hits)} ledgered file(s) touched"]
    if not required:
        lines.append(
            "all touched rows are tools-canonical or ud-copy-deleted; no pair needed"
        )
        return 0, lines
    if body is None:
        lines.append(
            "not a pull_request event (no PR body available); pair rule not enforced"
        )
        return 0, lines
    pair = paired_pr(body)
    if pair:
        lines.append(f"paired UpstreamDrift PR #{pair} referenced; ok")
        return 0, lines
    lines.append(
        "FAIL: these files belong to ledger rows whose ruling requires a paired "
        "UpstreamDrift PR; add 'UD-PAIR: D-sorganization/UpstreamDrift#N' to the "
        "PR body (or have the owner rule the row tools-canonical in "
        "docs/shared/divergence_ledger.v1.json):"
    )
    for changed_file, row in sorted(required.items()):
        lines.append(
            f"- {changed_file} -> {row['module']} (ruling={row['ruling']}, "
            f"status={row['status']})"
        )
    return 1, lines


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render(document: dict[str, object]) -> str:
    """Render the ledger as markdown (generated; do not edit by hand)."""
    pins = document.get("pins", {})
    assert isinstance(pins, dict)
    rows = rows_of(document)
    by_ruling: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for row in rows:
        by_ruling[str(row["ruling"])] = by_ruling.get(str(row["ruling"]), 0) + 1
        by_status[str(row["status"])] = by_status.get(str(row["status"]), 0) + 1
    out = [
        f"# {document.get('title', 'Divergence ledger')}",
        "",
        "Generated from `docs/shared/divergence_ledger.v1.json` by "
        "`python scripts/check_divergence_ledger.py --render`; do not edit by hand.",
        "",
        f"- Updated: {document.get('updated')}",
        f"- Pins: Tools `{pins.get('tools')}` / UpstreamDrift `{pins.get('upstreamdrift')}`",
        f"- Rows: {len(rows)}",
        "- Rulings: "
        + ", ".join(f"{key} {value}" for key, value in sorted(by_ruling.items())),
        "- Statuses: "
        + ", ".join(f"{key} {value}" for key, value in sorted(by_status.items())),
        "",
        "## Gate",
        "",
        str(document.get("gate", {}).get("rule", "")),  # type: ignore[union-attr]
        "",
        "## Ruling vocabulary",
        "",
    ]
    rulings = document.get("rulings", {})
    assert isinstance(rulings, dict)
    for key, value in rulings.items():
        out.append(f"- `{key}`: {value}")
    out += [
        "",
        "## Rows",
        "",
        "| Module | Tools path | UD path | Ruling | Status | Rulings | Owner | Source | Target PR | Inventory (id/div/ud/tools) | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        inventory = row["inventory"]
        inv = (
            f"{inventory['identical']}/{inventory['diverged']}/"
            f"{inventory['ud_only']}/{inventory['tools_only']}"
            if isinstance(inventory, dict)
            else ""
        )
        rulings_text = ", ".join(str(item) for item in row["rulings"])  # type: ignore[union-attr]
        out.append(
            "| "
            + " | ".join(
                [
                    f"`{_cell(row['module'])}`",
                    f"`{_cell(row['tools_path'])}`",
                    f"`{_cell(row['ud_path'])}`" if row["ud_path"] else "—",
                    _cell(row["ruling"]),
                    _cell(row["status"]),
                    _cell(rulings_text),
                    _cell(row["owner"]),
                    _cell(row["source_issue"]),
                    _cell(row["target_pr"]),
                    inv,
                    _cell(row["notes"]),
                ]
            )
            + " |"
        )
    return "\n".join(out) + "\n"


def _report(lines: Iterable[str]) -> None:
    for line in lines:
        sys.stdout.write(line + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Divergence ledger gate/renderer")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate + freshness")
    mode.add_argument("--render", action="store_true", help="rewrite the markdown")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--markdown", type=Path, default=RENDER_PATH)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        document = load_ledger(args.ledger)
    except (LedgerError, OSError, ValueError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    if args.render:
        args.markdown.write_text(render(document), encoding="utf-8", newline="\n")
        _report([f"wrote {args.markdown.as_posix()}"])
        return 0
    if args.check:
        expected = render(document)
        actual = (
            args.markdown.read_text(encoding="utf-8").replace("\r\n", "\n")
            if args.markdown.is_file()
            else ""
        )
        if actual != expected:
            sys.stderr.write(
                "ERROR: docs/shared/divergence_ledger.md is stale; run "
                "python scripts/check_divergence_ledger.py --render\n"
            )
            return 1
        _report([f"divergence ledger ok: {len(rows_of(document))} rows"])
        return 0
    code, lines = gate(
        document,
        changed_files(args.root, args.base),
        pr_body_from_event(os.environ.get("GITHUB_EVENT_PATH")),
    )
    _report(lines)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
