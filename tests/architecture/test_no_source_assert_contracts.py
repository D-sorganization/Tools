"""Source contract checks for -O-safe boundary validation."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

EXCLUDED_PARTS = {
    "benchmark",
    "benchmarks",
    "test",
    "tests",
}


def _iter_source_files() -> list[Path]:
    return [
        path
        for path in SRC_ROOT.rglob("*.py")
        if EXCLUDED_PARTS.isdisjoint(path.relative_to(SRC_ROOT).parts)
    ]


def test_source_boundary_validation_does_not_use_stripped_asserts() -> None:
    """Boundary contracts must raise explicit exceptions, not AssertionError."""
    violations: list[str] = []

    for path in _iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            if not isinstance(node.msg, ast.Constant) or not isinstance(
                node.msg.value, str
            ):
                continue
            if "must be provided" not in node.msg.value:
                continue
            violations.append(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.msg.value}"
            )

    assert not violations, (
        "Use explicit ValueError/TypeError boundary validation instead of "
        "assert ... 'must be provided':\n" + "\n".join(violations)
    )
