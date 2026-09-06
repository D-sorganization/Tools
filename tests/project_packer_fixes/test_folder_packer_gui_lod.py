"""Tests for LoD fixes in folder_packer_gui.py.

Verifies that the LoD violation fixes (file_path.suffix.lower() and
dir_path.name.lower() extracted into intermediate variables) work correctly
and do not break existing behavior.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_packer_dependencies():
    """Mock all project_packer internal dependencies before importing."""
    packer_path = str(Path(__file__).parent.parent.parent / "src" / "project_packer")

    # Mock utils modules
    utils_mock = MagicMock()
    utils_path_mock = MagicMock()
    utils_path_mock.ensure_utils_in_path = MagicMock(return_value=None)
    utils_compat_mock = MagicMock()
    utils_compat_mock.UTC = None

    constants_mock = MagicMock()
    constants_mock.BOLD_HEADER_FONT_SIZE = 12
    constants_mock.DEFAULT_LISTBOX_HEIGHT = 6
    constants_mock.DEFAULT_PADDING = 10
    constants_mock.DEFAULT_WINDOW_HEIGHT = 600
    constants_mock.DEFAULT_WINDOW_WIDTH = 800
    constants_mock.GRID_WEIGHT_MAIN = 1
    constants_mock.HEADER_FONT_SIZE = 11
    constants_mock.SMALL_PADDING = 5
    constants_mock.STATUS_TEXT_HEIGHT = 8
    constants_mock.TINY_PADDING = 2
    constants_mock.TITLE_FONT_SIZE = 14

    mods_to_add = {
        "utils": utils_mock,
        "utils.path_helpers": utils_path_mock,
        "utils.logging_utils": MagicMock(),
        "utils.subprocess_utils": MagicMock(),
        "utils.compatibility": utils_compat_mock,
        "constants": constants_mock,
    }

    original_modules = {}
    for name, mock in mods_to_add.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = mock

    if packer_path not in sys.path:
        sys.path.insert(0, packer_path)

    # Remove cached module to force reimport with mocks
    if "folder_packer_gui" in sys.modules:
        del sys.modules["folder_packer_gui"]

    yield

    # Cleanup
    if "folder_packer_gui" in sys.modules:
        del sys.modules["folder_packer_gui"]
    if packer_path in sys.path:
        sys.path.remove(packer_path)
    for name, original in original_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture()
def gui_module():
    """Import and return the folder_packer_gui module."""
    import importlib

    return importlib.import_module("folder_packer_gui")


@pytest.fixture()
def mock_root():
    """Create a mock Tkinter root window."""
    root = Mock()
    root.title = Mock()
    root.geometry = Mock()
    root.resizable = Mock()
    root.columnconfigure = Mock()
    root.rowconfigure = Mock()
    root.update_idletasks = Mock()
    return root


@pytest.fixture()
def gui_instance(gui_module, mock_root):
    """Create a FolderPackerGUI instance with all Tkinter widgets mocked."""
    FolderPackerGUI = gui_module.FolderPackerGUI
    with (
        patch("folder_packer_gui.ttk"),
        patch("folder_packer_gui.tk"),
    ):
        return FolderPackerGUI(mock_root)


class TestFolderPackerGuiLoDFix:
    """Tests verifying LoD fixes in folder_packer_gui.py."""

    def test_should_include_file_no_chained_suffix_lower(self, gui_module) -> None:
        """Verify should_include_file does not use file_path.suffix.lower() chain."""
        import inspect

        source = inspect.getsource(gui_module.FolderPackerGUI.should_include_file)
        assert "file_path.suffix.lower()" not in source, (
            "LoD violation: should_include_file must not chain .suffix.lower()"
        )

    def test_should_include_directory_no_chained_name_lower(self, gui_module) -> None:
        """Verify should_include_directory does not use dir_path.name.lower() chain."""
        import inspect

        source = inspect.getsource(gui_module.FolderPackerGUI.should_include_directory)
        assert "dir_path.name.lower()" not in source, (
            "LoD violation: should_include_directory must not chain .name.lower()"
        )

    def test_should_include_file_python_file(self, gui_instance) -> None:
        """Test that .py files are included."""
        result = gui_instance.should_include_file(Path("test.py"))
        assert result is True

    def test_should_include_file_txt_file(self, gui_instance) -> None:
        """Test that .txt files are included."""
        result = gui_instance.should_include_file(Path("readme.txt"))
        assert result is True

    def test_should_include_file_cfg_config(self, gui_instance) -> None:
        """Test that .cfg configuration files are always included."""
        result = gui_instance.should_include_file(Path("app.cfg"))
        assert result is True

    def test_should_include_file_toml_config(self, gui_instance) -> None:
        """Test that .toml configuration files are always included."""
        result = gui_instance.should_include_file(Path("pyproject.toml"))
        assert result is True

    def test_should_include_file_uppercase_extension(self, gui_instance) -> None:
        """Test that file inclusion is case-insensitive for extensions."""
        result_lower = gui_instance.should_include_file(Path("test.py"))
        result_upper = gui_instance.should_include_file(Path("test.PY"))
        assert result_lower is True
        assert result_upper is True

    def test_should_include_file_excluded_extension(self, gui_instance) -> None:
        """Test that non-whitelisted extensions are excluded."""
        result = gui_instance.should_include_file(Path("binary.bin"))
        assert result is False

    def test_should_include_directory_excluded_pattern(
        self, gui_instance, tmp_path
    ) -> None:
        """Test that directories matching exclusion patterns are rejected."""
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        result = gui_instance.should_include_directory(cache_dir)
        assert result is False

    def test_should_include_directory_git_excluded(
        self, gui_instance, tmp_path
    ) -> None:
        """Test that .git directories are excluded."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        result = gui_instance.should_include_directory(git_dir)
        assert result is False

    def test_should_include_directory_with_valid_files(
        self, gui_instance, tmp_path
    ) -> None:
        """Test that directories containing valid files are included."""
        valid_dir = tmp_path / "my_module"
        valid_dir.mkdir()
        (valid_dir / "module.py").write_text("# module code")
        result = gui_instance.should_include_directory(valid_dir)
        assert result is True

    def test_should_include_directory_empty_dir(self, gui_instance, tmp_path) -> None:
        """Test that empty directories are excluded (no includable files)."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = gui_instance.should_include_directory(empty_dir)
        assert result is False

    def test_no_print_calls_in_source(self, gui_module) -> None:
        """Verify no print() calls exist in folder_packer_gui module source."""
        import inspect

        source = inspect.getsource(gui_module)
        lines = source.splitlines()
        print_lines = [
            f"line {i + 1}: {line}"
            for i, line in enumerate(lines)
            if "print(" in line and not line.strip().startswith("#")
        ]
        assert not print_lines, (
            f"Found print() calls in folder_packer_gui.py: {print_lines}"
        )


class TestFolderPackerGuiDbCContracts:
    """Tests verifying DbC preconditions in folder_packer_gui.py."""

    def test_init_requires_root(self, gui_module) -> None:
        """Test that __init__ raises an error when root is None."""
        FolderPackerGUI = gui_module.FolderPackerGUI
        with pytest.raises((AssertionError, ValueError), match="root must be provided"):
            FolderPackerGUI(None)

    def test_should_include_file_requires_path(self, gui_instance) -> None:
        """Test that should_include_file raises an error when file_path is None."""
        with pytest.raises(
            (AssertionError, ValueError), match="file_path must be provided"
        ):
            gui_instance.should_include_file(None)

    def test_should_include_directory_requires_path(self, gui_instance) -> None:
        """Test that should_include_directory raises an error when dir_path is None."""
        with pytest.raises(
            (AssertionError, ValueError), match="dir_path must be provided"
        ):
            gui_instance.should_include_directory(None)

    def test_update_status_requires_message(self, gui_instance) -> None:
        """Test that update_status raises an error when message is None."""
        gui_instance.status_text = Mock()
        gui_instance.root = Mock()
        with pytest.raises(
            (AssertionError, ValueError), match="message must be provided"
        ):
            gui_instance.update_status(None)
