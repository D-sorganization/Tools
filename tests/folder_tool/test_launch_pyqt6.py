"""Unit tests for folder_tool.launch_pyqt6 module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestLaunchMain:
    """Tests for the main() launcher function."""

    def test_module_importable(self):
        """Verify the launch_pyqt6 module imports without errors."""
        import folder_tool.launch_pyqt6 as mod

        assert hasattr(mod, "main")
        assert callable(mod.main)

    def test_main_exits_on_missing_script(self, tmp_path):
        """main() should sys.exit(1) when tool script doesn't exist."""
        import folder_tool.launch_pyqt6 as mod

        fake_parent = tmp_path / "nonexistent"
        fake_parent.mkdir()

        with (
            patch.object(
                Path, "parent", new_callable=lambda: property(lambda s: fake_parent)
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            # Patch __file__ so script_dir points to an empty directory
            with patch.object(mod, "__file__", str(fake_parent / "launch_pyqt6.py")):
                mod.main()

        assert exc_info.value.code == 1

    def test_logger_exists(self):
        """Module should expose a logger."""
        import folder_tool.launch_pyqt6 as mod

        assert hasattr(mod, "logger")
