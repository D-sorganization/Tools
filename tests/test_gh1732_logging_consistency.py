"""Tests for GH1732: Operational output is inconsistent.

Verifies that no unguarded print() calls exist in library source files after the
GH1655 migration and that the enforcement remains in place.

Scope: all src/**/*.py files except ruff-excluded directories
(data_processing, document_processing, scientific_modeling, pendulum_simulator)
and test sub-directories.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Directories excluded from ruff T201 enforcement (maintained in ruff.toml).
# If ruff.toml exclude list changes, update this set to match.
_RUFF_EXCLUDED_SRC_DIRS = frozenset(
    [
        "data_processing",
        "document_processing",
        "scientific_modeling",
        "pendulum_simulator",
    ]
)

_REPO_ROOT = Path(__file__).parents[1]
_SRC_ROOT = _REPO_ROOT / "src"


def _collect_library_py_files() -> list[Path]:
    """Collect .py files subject to T201 enforcement.

    Returns all src/**/*.py files that are:
    - NOT inside an ruff-excluded top-level directory
    - NOT inside a tests/ subdirectory
    - NOT __pycache__ entries
    """
    result = []
    for f in sorted(_SRC_ROOT.rglob("*.py")):
        parts = f.relative_to(_SRC_ROOT).parts
        # Skip ruff-excluded directories
        if any(part in _RUFF_EXCLUDED_SRC_DIRS for part in parts):
            continue
        # Skip test subdirectories
        if "tests" in parts:
            continue
        # Skip __pycache__ and node_modules
        if "__pycache__" in parts or "node_modules" in parts:
            continue
        result.append(f)
    return result


def _find_print_calls(source: str, filename: str) -> list[int]:
    """Parse source with AST and return line numbers of top-level print() calls.

    Only counts ast.Expr(ast.Call(ast.Name('print'))) — i.e., calls to the
    built-in print() as a statement. Does NOT flag:
    - Docstring examples (>>> print(...)) — they are inside ast.Constant nodes
    - console.print() — that is an ast.Attribute call, not ast.Name
    - String literals containing "print(" — code generators, templates
    - Attribute access (obj.print()) — not ast.Name with id='print'
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    violations: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
        ):
            violations.append(node.lineno)
    return violations


class TestNoUnguardedPrintInLibrarySrc:
    """No unguarded print() calls exist in library source files.

    Files inside ruff-excluded directories and test subdirectories are out of scope.
    The ruff T201 rule and # noqa suppressions are the enforcement mechanism — this
    test verifies the AST-level reality, independent of ruff.
    """

    def test_no_print_calls_in_library_source(self) -> None:
        """Sweep all in-scope src/**/*.py files for unguarded print() calls."""
        files = _collect_library_py_files()
        assert len(files) > 0, "Expected at least one library source file to check"

        violations: list[str] = []
        for f in files:
            try:
                source = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = _find_print_calls(source, str(f))
            for line in lines:
                rel = f.relative_to(_REPO_ROOT)
                violations.append(f"{rel}:{line}")

        assert violations == [], (
            f"Found {len(violations)} unguarded print() call(s) in library source.\n"
            "Use logging instead, or add # noqa: T201 for intentional CLI output.\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )

    def test_collection_covers_shared_python(self) -> None:
        """Sweep includes src/shared/python — the shared library layer."""
        files = _collect_library_py_files()
        shared_files = [f for f in files if "shared" in f.parts and "python" in f.parts]
        assert len(shared_files) > 0, (
            "Expected at least one file from src/shared/python/ in the sweep"
        )

    def test_collection_excludes_ruff_excluded_dirs(self) -> None:
        """Files from ruff-excluded directories are not in the sweep."""
        files = _collect_library_py_files()
        for f in files:
            parts = f.relative_to(_SRC_ROOT).parts
            excluded = [p for p in parts if p in _RUFF_EXCLUDED_SRC_DIRS]
            assert not excluded, (
                f"File from excluded directory should not be in sweep: {f}"
            )

    def test_collection_excludes_test_subdirs(self) -> None:
        """Test subdirectories are not in the sweep."""
        files = _collect_library_py_files()
        for f in files:
            parts = f.relative_to(_SRC_ROOT).parts
            assert "tests" not in parts, (
                f"File from tests/ subdirectory should not be in sweep: {f}"
            )


class TestLoggingConsistencyRuffConfig:
    """Ruff is configured to enforce the print-free policy."""

    def test_t201_in_ruff_select(self) -> None:
        """ruff.toml must include T201 in lint select."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        ruff_toml = _REPO_ROOT / "ruff.toml"
        config = tomllib.loads(ruff_toml.read_text())
        lint_select = config["lint"]["select"]
        assert "T201" in lint_select, (
            "T201 must be in [lint] select in ruff.toml to enforce the no-print policy"
        )

    def test_notebooks_excluded_from_t201(self) -> None:
        """Notebooks must have T201 suppressed (print is valid in notebooks)."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        ruff_toml = _REPO_ROOT / "ruff.toml"
        config = tomllib.loads(ruff_toml.read_text())
        per_file = config["lint"].get("per-file-ignores", {})
        notebook_ignores = per_file.get("**/*.ipynb", [])
        assert "T201" in notebook_ignores, (
            "**/*.ipynb must have T201 in per-file-ignores in ruff.toml "
            "(print is valid for display output in Jupyter notebooks)"
        )

    def test_scripts_excluded_from_t201(self) -> None:
        """Scripts directories must have T201 suppressed (CLI output is intentional)."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        ruff_toml = _REPO_ROOT / "ruff.toml"
        config = tomllib.loads(ruff_toml.read_text())
        per_file = config["lint"].get("per-file-ignores", {})
        # Either scripts/**/*.py or scripts/*.py must have T201 ignored
        scripts_patterns = [k for k in per_file if "scripts" in k]
        any_has_t201 = any("T201" in per_file[k] for k in scripts_patterns)
        assert any_has_t201, (
            "scripts/ must have T201 suppressed in ruff.toml per-file-ignores "
            "(scripts use print() for intentional CLI output)"
        )


class TestPrintCallDetectionAccuracy:
    """The AST-based detector correctly identifies only true print() calls."""

    def test_docstring_print_not_flagged(self) -> None:
        """>>> print(...) in docstrings is not an executable call — not flagged."""
        source = '''
def convert(x):
    """Convert x.

    Examples:
        >>> print(f"Result: {convert(1)}")
    """
    return x
'''
        assert _find_print_calls(source, "<test>") == []

    def test_string_literal_print_not_flagged(self) -> None:
        """String containing 'print(' is not a call — not flagged."""
        source = """
lines = [
    "    print(f'Processing complete: {result}')",
    "    print(f'Error: {e}')",
]
"""
        assert _find_print_calls(source, "<test>") == []

    def test_console_print_not_flagged(self) -> None:
        """console.print() is a method call on an object — not flagged."""
        source = """
console.print("[green]Done[/green]")
"""
        assert _find_print_calls(source, "<test>") == []

    def test_bare_print_is_flagged(self) -> None:
        """A bare print() call at statement level IS flagged."""
        source = """
import os

print("debug output")
"""
        lines = _find_print_calls(source, "<test>")
        assert lines == [4], f"Expected line 4 to be flagged, got {lines}"

    def test_print_in_function_is_flagged(self) -> None:
        """A print() inside a function body IS flagged."""
        source = """
def run():
    print("running")
"""
        lines = _find_print_calls(source, "<test>")
        assert lines == [3], f"Expected line 3 to be flagged, got {lines}"
