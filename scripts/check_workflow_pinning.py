#!/usr/bin/env python3
"""Reject new mutable workflow actions and unsafe installer patterns."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

WORKFLOW_ROOT = Path(".github") / "workflows"
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
USES_RE = re.compile(r"\buses:\s*([^\s#]+)")
CURL_PIPE_RE = re.compile(r"\b(curl|wget)\b.*\|\s*(sh|bash)\b")
NPM_GLOBAL_RE = re.compile(r"\bnpm\s+install\s+-g\s+([^\s#]+)")


@dataclass(frozen=True)
class WorkflowPinningViolation:
    """One workflow supply-chain pinning violation."""

    path: str
    line: int
    kind: str
    value: str

    @property
    def baseline_key(self) -> str:
        return f"{self.path}|{self.kind}|{self.value}"


def _workflow_paths(paths: list[Path]) -> list[Path]:
    if paths:
        return [path for path in paths if path.suffix in {".yml", ".yaml"}]
    return sorted(WORKFLOW_ROOT.glob("*.yml")) + sorted(WORKFLOW_ROOT.glob("*.yaml"))


def _is_local_action(ref: str) -> bool:
    return ref.startswith("./") or ref.startswith(".github/")


def _is_pinned_action(ref: str) -> bool:
    if _is_local_action(ref):
        return True
    if "@" not in ref:
        return False
    return bool(SHA_RE.fullmatch(ref.rsplit("@", 1)[1]))


def _npm_package_has_exact_version(package: str) -> bool:
    if package.startswith("@"):
        if package.count("@") < 2:
            return False
        version = package.rsplit("@", 1)[1]
    elif "@" in package:
        version = package.rsplit("@", 1)[1]
    else:
        return False
    return not any(marker in version for marker in ("*", "^", "~", ">", "<", "latest"))


def scan_workflow(path: Path) -> list[WorkflowPinningViolation]:
    """Scan one workflow for mutable action/download/install patterns."""
    violations: list[WorkflowPinningViolation] = []
    rel_path = path.as_posix()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        uses_match = USES_RE.search(line)
        if uses_match:
            action_ref = uses_match.group(1).strip("'\"")
            if not _is_pinned_action(action_ref):
                violations.append(
                    WorkflowPinningViolation(
                        rel_path, line_number, "mutable-action", action_ref
                    )
                )

        if CURL_PIPE_RE.search(line):
            violations.append(
                WorkflowPinningViolation(
                    rel_path, line_number, "curl-pipe", line.strip()
                )
            )

        npm_match = NPM_GLOBAL_RE.search(line)
        if npm_match:
            package = npm_match.group(1).strip("'\"")
            if not _npm_package_has_exact_version(package):
                violations.append(
                    WorkflowPinningViolation(
                        rel_path, line_number, "unpinned-global-npm", package
                    )
                )
    return violations


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("allowlisted_violations", [])
    if not isinstance(entries, list):
        raise ValueError("allowlisted_violations must be a list")
    return {str(entry) for entry in entries}


def check(paths: list[Path], baseline_file: Path) -> list[WorkflowPinningViolation]:
    baseline = _load_baseline(baseline_file)
    violations: list[WorkflowPinningViolation] = []
    for path in _workflow_paths(paths):
        violations.extend(
            violation
            for violation in scan_workflow(path)
            if violation.baseline_key not in baseline
        )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("config/workflow_pinning_baseline.json"),
    )
    args = parser.parse_args()

    violations = check(args.paths, args.baseline_file)
    if violations:
        print("Workflow pinning policy failed:", file=sys.stderr)
        for violation in violations:
            print(
                f"- {violation.path}:{violation.line}: "
                f"{violation.kind}: {violation.value}",
                file=sys.stderr,
            )
        return 1
    print("Workflow pinning policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
