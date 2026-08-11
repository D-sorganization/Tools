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
        assert (
            "logging" in import_names
        ), "logging must be imported in modern_robotics.py"


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
                    f"Found print() call in {src_file}:"
                    f"{node.lineno} — use logging instead"
                )


class TestPrintMigrationTooling:
    """The print-to-logging migration path has one canonical script."""

    REPO_ROOT = Path(__file__).parents[1]

    def test_root_migration_script_is_not_reintroduced(self) -> None:
        """The retained migrator lives under scripts/, not the repo root."""
        assert not (self.REPO_ROOT / "migrate_print_to_logging.py").exists()
        assert (self.REPO_ROOT / "scripts" / "convert_print_to_logging.py").is_file()

    def test_converter_preserves_inline_comment_outside_call(self) -> None:
        """Inline comments after print() must not become logger call arguments."""
        from scripts.convert_print_to_logging import _convert_print_line

        converted, changed = _convert_print_line("    print('done')  # keep me")

        assert changed is True
        assert converted == "    logger.info('done')  # keep me"

    def test_converter_handles_nested_parentheses_before_comment(self) -> None:
        """Nested calls should convert without greedy comment capture."""
        from scripts.convert_print_to_logging import _convert_print_line

        converted, changed = _convert_print_line(
            "print(format_value(func(')')))  # nested"
        )

        assert changed is True
        assert converted == "logger.info(format_value(func(')')))  # nested"

    def test_converter_uses_word_boundaries_for_log_level(self) -> None:
        """Substring matches like 'no errors' should not force error logging."""
        from scripts.convert_print_to_logging import _convert_print_line

        converted, changed = _convert_print_line("print('no errors detected')")

        assert changed is True
        assert converted == "logger.info('no errors detected')"

    def test_converter_still_detects_error_word(self) -> None:
        """A standalone error token should still map to logger.error."""
        from scripts.convert_print_to_logging import _convert_print_line

        converted, changed = _convert_print_line("print('error detected')")

        assert changed is True
        assert converted == "logger.error('error detected')"


class TestRuffT201Configured:
    """Verify T201 (print-statement) rule is enabled in ruff config.

    ruff.toml is the authoritative ruff configuration (takes precedence over
    pyproject.toml). Tests read from ruff.toml directly.
    """

    def test_t201_in_ruff_select(self) -> None:
        """ruff.toml must include T201 in lint select."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        ruff_toml = Path(__file__).parents[1] / "ruff.toml"
        config = tomllib.loads(ruff_toml.read_text())
        lint_select = config["lint"]["select"]
        assert (
            "T201" in lint_select
        ), "T201 must be in [lint] select in ruff.toml to enforce no-print policy"

    def test_notebooks_excluded_from_t201(self) -> None:
        """Notebooks must be excluded from T201 (print is valid)."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        ruff_toml = Path(__file__).parents[1] / "ruff.toml"
        config = tomllib.loads(ruff_toml.read_text())
        per_file = config["lint"].get("per-file-ignores", {})
        notebook_ignores = per_file.get("**/*.ipynb", [])
        assert "T201" in notebook_ignores, (
            "**/*.ipynb must have T201 in per-file-ignores in ruff.toml "
            "(print is valid in Jupyter notebooks)"
        )
