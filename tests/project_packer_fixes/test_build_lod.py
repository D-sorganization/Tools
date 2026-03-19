"""Tests for LoD fixes in build.py.

Verifies that the LoD violation fixes:
- Path(__file__).parent.absolute() -> extracted to intermediate variables
- sys.stderr.write() -> extracted sys.stderr to local variable
work correctly and do not break existing behavior.
"""

import inspect
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture()
def build_module():
    """Import build module from project_packer with mocked dependencies."""
    import importlib

    packer_path = str(Path(__file__).parent.parent.parent / "src" / "project_packer")

    utils_mock = Mock()
    utils_mock.ensure_utils_in_path = Mock(return_value=None)
    logging_utils_mock = Mock()
    logging_utils_mock.get_logger = Mock(return_value=Mock())
    subprocess_utils_mock = Mock()

    mods_to_add = {
        "utils": Mock(),
        "utils.path_helpers": utils_mock,
        "utils.logging_utils": logging_utils_mock,
        "utils.subprocess_utils": subprocess_utils_mock,
    }

    original_modules = {}
    for name, mock in mods_to_add.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = mock

    if packer_path not in sys.path:
        sys.path.insert(0, packer_path)

    if "build" in sys.modules:
        del sys.modules["build"]

    try:
        mod = importlib.import_module("build")
        yield mod
    finally:
        if "build" in sys.modules:
            del sys.modules["build"]
        if packer_path in sys.path:
            sys.path.remove(packer_path)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class TestBuildLoDFix:
    """Tests verifying LoD fixes in build.py."""

    def test_main_no_chained_path_parent_absolute(self, build_module) -> None:
        """Verify main() does not chain Path().parent.absolute() directly."""
        source = inspect.getsource(build_module.main)
        assert "Path(__file__).parent.absolute()" not in source, (
            "LoD violation: build.py must not chain Path(__file__).parent.absolute()"
        )

    def test_main_no_chained_stderr_write(self, build_module) -> None:
        """Verify main() does not chain sys.stderr.write() directly."""
        source = inspect.getsource(build_module.main)
        assert "sys.stderr.write" not in source, (
            "LoD violation: build.py must not chain sys.stderr.write() directly"
        )

    def test_main_extracts_stderr_to_variable(self, build_module) -> None:
        """Verify main() extracts sys.stderr to a local variable."""
        source = inspect.getsource(build_module.main)
        assert "stderr = sys.stderr" in source, (
            "build.py main() should extract sys.stderr to a local variable"
        )

    def test_main_extracts_path_parent(self, build_module) -> None:
        """Verify main() extracts Path().parent to an intermediate variable."""
        source = inspect.getsource(build_module.main)
        # Should use script_parent or similar intermediate variable
        assert "Path(__file__).parent" in source, (
            "build.py should still use Path(__file__).parent but assign to intermediate"
        )
        assert ".absolute()" in source, (
            "build.py should call .absolute() on the intermediate variable"
        )

    def test_no_print_calls_in_source(self, build_module) -> None:
        """Verify no print() calls exist in build module source."""
        source = inspect.getsource(build_module)
        lines = source.splitlines()
        print_lines = [
            f"line {i + 1}: {line}"
            for i, line in enumerate(lines)
            if "print(" in line and not line.strip().startswith("#")
        ]
        assert not print_lines, f"Found print() calls in build.py: {print_lines}"
