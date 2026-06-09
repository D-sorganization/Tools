#!/usr/bin/env python3
"""Enforce behavioral assertions in changed Python test files.

Design by Contract:
    Precondition: checked paths are repository-relative or absolute Python files.
    Postcondition: returns 0 only when each changed Python test file has an AST
    assertion, a unittest/mock assert method call, or an exception assertion.
    Invariant: fixture-only exemptions are explicit allowlist patterns.

Usage:
    python scripts/check_test_assertions.py
    python scripts/check_test_assertions.py --changed-files changed_files.txt
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = ROOT / "scripts" / "test_assertion_allowlist.txt"
DIFF_BASE = "origin/main...HEAD"
ASSERT_METHOD_PREFIX = "assert"
EXCEPTION_ASSERTION_NAMES = {"raises", "assertRaises"}


def _repo_relative(path: Path, root: Path) -> str:
    """Return a stable repository-relative POSIX path."""
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.resolve().relative_to(root.resolve())
    except ValueError:
        relative = path
    return relative.as_posix()


def _is_assert_method_call(node: ast.Call) -> bool:
    """Return True when the call is a unittest/mock-style assertion method."""
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr.startswith(
        ASSERT_METHOD_PREFIX
    )


def _is_exception_assertion_call(node: ast.Call) -> bool:
    """Return True when the call is an exception assertion context manager."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in EXCEPTION_ASSERTION_NAMES
    if isinstance(func, ast.Attribute):
        return func.attr in EXCEPTION_ASSERTION_NAMES
    return False


def _with_item_has_exception_assertion(item: ast.withitem) -> bool:
    """Return True when a with-item context expression asserts an exception."""
    expression = item.context_expr
    return isinstance(expression, ast.Call) and _is_exception_assertion_call(expression)


def has_behavioral_assertion(source: str) -> bool:
    """Return True if Python source contains a behavioral assertion marker."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call) and _is_assert_method_call(node):
            return True
        if isinstance(node, (ast.With, ast.AsyncWith)):
            if any(_with_item_has_exception_assertion(item) for item in node.items):
                return True
    return False


def load_allowlist(path: Path = DEFAULT_ALLOWLIST) -> tuple[str, ...]:
    """Load fixture-only exemption patterns from an allowlist file."""
    if not path.exists():
        return ()

    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return tuple(patterns)


def is_allowlisted(path: Path, patterns: Sequence[str], root: Path = ROOT) -> bool:
    """Return True when a path matches an explicit fixture-only allowlist."""
    relative = _repo_relative(path, root)
    return any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


def is_python_test_file(path: Path, root: Path = ROOT) -> bool:
    """Return True when a path is a Python file belonging to the test surface."""
    if path.suffix != ".py":
        return False

    relative_parts = Path(_repo_relative(path, root)).parts
    name = path.name
    if "tests" in relative_parts:
        return True
    return name.startswith("test_") or name.endswith("_test.py")


def select_python_test_files(paths: Iterable[Path], root: Path = ROOT) -> list[Path]:
    """Filter changed paths down to Python test files that still exist."""
    selected: list[Path] = []
    for path in paths:
        candidate = path if path.is_absolute() else root / path
        if candidate.exists() and is_python_test_file(candidate, root):
            selected.append(candidate)
    return selected


def check_test_files(
    paths: Iterable[Path],
    allowlist_patterns: Sequence[str],
    root: Path = ROOT,
) -> list[Path]:
    """Return changed Python test files that lack behavioral assertions."""
    violations: list[Path] = []
    for path in select_python_test_files(paths, root):
        if is_allowlisted(path, allowlist_patterns, root):
            continue
        source = path.read_text(encoding="utf-8")
        if not has_behavioral_assertion(source):
            violations.append(path)
    return violations


def changed_files(base: str = DIFF_BASE, root: Path = ROOT) -> list[Path]:
    """Return paths changed between base and HEAD using git diff."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _read_changed_files(path: Path) -> list[Path]:
    """Read newline-delimited changed paths from a file."""
    return [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--changed-files",
        type=Path,
        help="Newline-delimited changed paths. Defaults to git diff origin/main...HEAD.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="Fixture-only allowlist pattern file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the changed-test assertion gate and return an exit code."""
    args = parse_args(argv)
    paths = (
        _read_changed_files(args.changed_files)
        if args.changed_files
        else changed_files()
    )
    allowlist_patterns = load_allowlist(args.allowlist)
    violations = check_test_files(paths, allowlist_patterns)

    if not violations:
        sys.stdout.write("Changed Python test assertion check passed.\n")
        return 0

    sys.stderr.write("Changed Python test files without behavioral assertions:\n")
    for path in violations:
        sys.stderr.write(f"- {_repo_relative(path, ROOT)}\n")
    sys.stderr.write(
        "Add an assert/pytest.raises/unittest assert, or add a fixture-only "
        "pattern to scripts/test_assertion_allowlist.txt.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
