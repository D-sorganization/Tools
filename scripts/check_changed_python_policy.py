#!/usr/bin/env python3
"""Guard changed production Python files against local policy regressions."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_REF = "origin/main"
DEFAULT_ALLOWLIST = ROOT / "scripts" / "changed_python_policy_allowlist.txt"
ALLOWLIST_SEPARATOR = "|"
PRINT_POLICY = "print"
SYS_PATH_POLICY = "sys_path"
SYS_PATH_MUTATORS = frozenset({"append", "extend", "insert"})


@dataclass(frozen=True)
class PolicyViolation:
    """A changed-file policy violation found in Python source."""

    path: str
    line_number: int
    policy: str
    message: str


def load_allowlist(path: Path) -> dict[str, str]:
    """Load exact-path policy exemptions with required documented reasons."""
    if not path.exists():
        return {}

    allowlist: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ALLOWLIST_SEPARATOR not in line:
            raise ValueError(
                f"{path}:{line_number}: allowlist entries need a documented reason"
            )
        item_path, reason = (
            part.strip() for part in line.split(ALLOWLIST_SEPARATOR, 1)
        )
        if not item_path or not reason:
            raise ValueError(
                f"{path}:{line_number}: allowlist entries need a documented reason"
            )
        allowlist[_normalize_path(item_path)] = reason
    return allowlist


def changed_files(base_ref: str = DEFAULT_BASE_REF) -> list[str]:
    """Return files changed relative to base_ref."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def find_policy_violations(
    root: Path,
    changed_paths: list[str],
    allowlist: dict[str, str],
) -> list[PolicyViolation]:
    """Find disallowed print() calls and sys.path mutations in changed files."""
    violations: list[PolicyViolation] = []
    for relative_path in changed_paths:
        normalized_path = _normalize_path(relative_path)
        if not _should_check_path(normalized_path, allowlist):
            continue
        file_path = root / normalized_path
        if not file_path.exists():
            continue
        violations.extend(_violations_for_file(file_path, normalized_path))
    return violations


def _should_check_path(path: str, allowlist: dict[str, str]) -> bool:
    return path.endswith(".py") and path not in allowlist and not _is_test_path(path)


def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


def _violations_for_file(file_path: Path, relative_path: str) -> list[PolicyViolation]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=relative_path)
    visitor = _PolicyVisitor(relative_path)
    visitor.visit(tree)
    return visitor.violations


def _normalize_path(path: str) -> str:
    return Path(path).as_posix()


def _is_sys_path(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _is_sys_path_mutator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr in SYS_PATH_MUTATORS
        and _is_sys_path(node.value)
    )


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self._relative_path = relative_path
        self.violations: list[PolicyViolation] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == PRINT_POLICY:
            self.violations.append(
                PolicyViolation(
                    path=self._relative_path,
                    line_number=node.lineno,
                    policy=PRINT_POLICY,
                    message="Use logging instead of production print().",
                )
            )
        if _is_sys_path_mutator(node.func):
            self.violations.append(
                PolicyViolation(
                    path=self._relative_path,
                    line_number=node.lineno,
                    policy=SYS_PATH_POLICY,
                    message="Use package imports instead of mutating sys.path.",
                )
            )
        self.generic_visit(node)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when changed production Python files add print() calls or sys.path "
            "mutations without a documented allowlist entry."
        )
    )
    parser.add_argument(
        "--base-ref",
        default=DEFAULT_BASE_REF,
        help="Git ref used for changed-file detection.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="Path to exact-path allowlist with documented reasons.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    allowlist = load_allowlist(args.allowlist)
    violations = find_policy_violations(ROOT, changed_files(args.base_ref), allowlist)
    if not violations:
        sys.stdout.write("Changed Python policy guard passed.\n")
        return 0

    sys.stderr.write("Changed Python policy guard failed:\n")
    for violation in violations:
        sys.stderr.write(
            f"- {violation.path}:{violation.line_number}: "
            f"{violation.policy}: {violation.message}\n"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
