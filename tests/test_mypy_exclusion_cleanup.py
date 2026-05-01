"""Tests for mypy.ini exclusion cleanup (issue #2351).

These tests verify that the mypy exclusion audit script correctly identifies
phantom exclusions and calculates coverage impact.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The script under test lives in scripts/validate_mypy_exclusions.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_mypy_exclusions as vma  # noqa: E402


class TestParseMypyExcludes:
    """Unit tests for parse_mypy_excludes()."""

    def test_extracts_patterns_from_valid_ini(self, tmp_path: Path) -> None:
        ini = tmp_path / "mypy.ini"
        ini.write_text(
            "[mypy]\n"
            "exclude = (\\.*/tests/|src/foo\\.py)\n"
            "warn_return_any = True\n"
        )
        patterns = vma.parse_mypy_excludes(ini)
        # strip("\\") removes the leading backslash from \.*/tests/
        assert patterns == {".*/tests/", "src/foo\\.py"}

    def test_raises_when_no_exclude_line(self, tmp_path: Path) -> None:
        ini = tmp_path / "mypy.ini"
        ini.write_text("[mypy]\nwarn_return_any = True\n")
        with pytest.raises(ValueError, match="No exclude line found"):
            vma.parse_mypy_excludes(ini)

    def test_handles_multiline_exclude(self, tmp_path: Path) -> None:
        ini = tmp_path / "mypy.ini"
        ini.write_text(
            "[mypy]\n"
            "exclude = (\n"
            "    \\.*/tests/|\n"
            "    src/foo\\.py|\n"
            "    config/project_template/\n"
            ")\n"
        )
        patterns = vma.parse_mypy_excludes(ini)
        # strip("\\") removes the leading backslash from \.*/tests/
        assert patterns == {".*/tests/", "src/foo\\.py", "config/project_template/"}


class TestCountAffectedFiles:
    """Unit tests for count_affected_files()."""

    def test_zero_for_phantom_pattern(self, tmp_path: Path) -> None:
        """A pattern matching nothing should report 0 files."""
        # We mock git ls-files to return empty for a phantom directory
        with patch.object(
            vma.subprocess, "run", return_value=MagicMock(stdout="", returncode=0)
        ):
            results = vma.count_affected_files({"phantom_dir/"})
        assert results["phantom_dir/"] == 0

    def test_positive_count_for_real_files(self, tmp_path: Path) -> None:
        """A pattern matching real tracked files should report >0."""
        fake_output = "src/data_processing/__init__.py\nsrc/data_processing/cli.py\n"
        with patch.object(
            vma.subprocess,
            "run",
            return_value=MagicMock(stdout=fake_output, returncode=0),
        ):
            results = vma.count_affected_files({"src/data_processing/"})
        assert results["src/data_processing/"] == 2


class TestMainSmoke:
    """Smoke tests for the CLI entry point."""

    def test_main_returns_zero_on_valid_ini(self, tmp_path: Path) -> None:
        ini = tmp_path / "mypy.ini"
        ini.write_text(
            "[mypy]\n"
            "exclude = (\\.*/tests/)\n"
            "warn_return_any = True\n"
        )
        with patch.object(vma, "REPO_ROOT", tmp_path):
            rc = vma.main()
        assert rc == 0

    def test_main_returns_one_on_missing_ini(self, tmp_path: Path) -> None:
        # main() catches ValueError and returns 1, but FileNotFoundError
        # is raised when mypy.ini doesn't exist. Patch the path to a
        # directory that exists but has no mypy.ini.
        with patch.object(vma, "REPO_ROOT", tmp_path), pytest.raises(FileNotFoundError):
            vma.main()


class TestIntegration:
    """Integration test against the real repo mypy.ini."""

    def test_real_ini_has_no_phantom_dirs(self) -> None:
        """After the cleanup, no directory patterns in mypy.ini should be phantoms."""
        ini_path = vma.REPO_ROOT / "mypy.ini"
        assert ini_path.exists(), "mypy.ini must exist"

        patterns = vma.parse_mypy_excludes(ini_path)
        results = vma.count_affected_files(patterns)

        phantoms = [
            p for p, c in results.items() if c == 0 and not p.endswith("tests/")
        ]
        assert phantoms == [], f"Phantom exclusions found: {phantoms}"
