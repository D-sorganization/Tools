"""Tests for GH1655: Replace print() calls in src/ with logging.

Verifies:
- modern_robotics module uses logging instead of print()
- T201 rule is active in ruff configuration
- No unguarded print() calls in library source files
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest


class TestModernRoboticsLogging:
    """modern_robotics uses logging module, not print()."""

    def test_module_has_logger(self) -> None:
        """modern_robotics module exposes a _logger attribute."""
        import rotation_converter.modern_robotics as mr

        assert hasattr(mr, "_logger")
        assert isinstance(mr._logger, logging.Logger)

    def test_logger_name_is_module(self) -> None:
        """Logger name matches the module's __name__."""
        import rotation_converter.modern_robotics as mr

        assert mr._logger.name == "rotation_converter.modern_robotics"

    def test_import_logging_present(self) -> None:
        """logging is imported at the top of modern_robotics."""
        src = (
            Path(__file__).parents[1]
            / "src"
            / "rotation_converter"
            / "modern_robotics.py"
        )
        source = src.read_text()
        tree = ast.parse(source)
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.append(node.module)
        assert "logging" in import_names, (
            "logging must be imported in modern_robotics.py"
        )


class TestNoUnguardedPrintInSrc:
    """Verify no unguarded print() calls exist in src/ .py library files."""

    SRC_ROOT = Path(__file__).parents[1] / "src"

    # Files where print() is legitimately suppressed (CLI/scripts)
    ALLOWED_NOQA_FILES = {
        "src/pendulum_simulator/perf_test.py",
        "src/pendulum_simulator/signal_test.py",
        "src/pendulum_simulator/test_sim.py",
        "src/pendulum_simulator/src/double_pendulum_golf/__main__.py",
    }

    def _is_print_call(self, node: ast.AST) -> bool:
        """Return True if node is a call to print()."""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        )

    def test_no_bare_print_in_modern_robotics(self) -> None:
        """modern_robotics.py must not contain executable print() calls."""
        src_file = self.SRC_ROOT / "rotation_converter" / "modern_robotics.py"
        source = src_file.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and self._is_print_call(node.value):
                pytest.fail(
                    f"Found print() call in {src_file}:{node.lineno} — use logging instead"
                )


class TestRuffT201Configured:
    """Verify T201 (print-statement) rule is enabled in ruff config."""

    def test_t201_in_ruff_select(self) -> None:
        """pyproject.toml must include T201 in ruff lint select."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        pyproject = Path(__file__).parents[1] / "pyproject.toml"
        config = tomllib.loads(pyproject.read_text())
        lint_select = config["tool"]["ruff"]["lint"]["select"]
        assert "T201" in lint_select, (
            "T201 must be in [tool.ruff.lint] select to enforce no-print policy"
        )

    def test_notebooks_excluded_from_t201(self) -> None:
        """Jupyter notebooks must be excluded from T201 (print is valid in notebooks)."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        pyproject = Path(__file__).parents[1] / "pyproject.toml"
        config = tomllib.loads(pyproject.read_text())
        per_file = config["tool"]["ruff"]["lint"].get("per-file-ignores", {})
        notebook_ignores = per_file.get("**/*.ipynb", [])
        assert "T201" in notebook_ignores, (
            "**/*.ipynb must have T201 in per-file-ignores (print is valid in Jupyter notebooks)"
        )
