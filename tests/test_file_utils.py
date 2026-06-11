"""Tests for file_utils - Shared file I/O utilities.

These tests verify the file utility functions using
Design by Contract principles.
"""

import json
from pathlib import Path


class TestSafeReadJsonContract:
    """Design by Contract tests for safe_read_json function."""

    def test_returns_default_for_missing_file(self, tmp_path):
        """Postcondition: Returns default when file doesn't exist."""
        from utils.file_utils import safe_read_json

        result = safe_read_json(tmp_path / "nonexistent.json", default={"key": "value"})
        assert result == {"key": "value"}

    def test_returns_parsed_json_for_valid_file(self, tmp_path):
        """Postcondition: Returns parsed JSON for valid file."""
        from utils.file_utils import safe_read_json

        test_file = tmp_path / "test.json"
        test_file.write_text('{"name": "test", "value": 42}')

        result = safe_read_json(test_file)
        assert result == {"name": "test", "value": 42}


class TestSafeReadJson:
    """Functional tests for safe_read_json."""

    def test_reads_valid_json_file(self, tmp_path):
        """Test reading a valid JSON file."""
        from utils.file_utils import safe_read_json

        data = {"users": [{"id": 1, "name": "Alice"}], "count": 1}
        test_file = tmp_path / "data.json"
        test_file.write_text(json.dumps(data))

        result = safe_read_json(test_file)
        assert result == data

    def test_returns_default_for_invalid_json(self, tmp_path):
        """Test returning default for invalid JSON."""
        from utils.file_utils import safe_read_json

        test_file = tmp_path / "invalid.json"
        test_file.write_text("{ invalid json }")

        result = safe_read_json(test_file, default=None)
        assert result is None

    def test_returns_default_for_nonexistent_file(self, tmp_path):
        """Test returning default for nonexistent file."""
        from utils.file_utils import safe_read_json

        result = safe_read_json(tmp_path / "missing.json", default=[])
        assert result == []

    def test_accepts_string_path(self, tmp_path):
        """Test accepting string path."""
        from utils.file_utils import safe_read_json

        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')

        result = safe_read_json(str(test_file))
        assert result == {"key": "value"}

    def test_default_is_none_when_not_specified(self, tmp_path):
        """Test default is None when not specified."""
        from utils.file_utils import safe_read_json

        result = safe_read_json(tmp_path / "missing.json")
        assert result is None


class TestSafeWriteJsonContract:
    """Design by Contract tests for safe_write_json function."""

    def test_returns_bool(self, tmp_path):
        """Postcondition: Returns a boolean."""
        from utils.file_utils import safe_write_json

        result = safe_write_json(tmp_path / "test.json", {"key": "value"})
        assert isinstance(result, bool)

    def test_returns_true_on_success(self, tmp_path):
        """Postcondition: Returns True on successful write."""
        from utils.file_utils import safe_write_json

        result = safe_write_json(tmp_path / "test.json", {"key": "value"})
        assert result is True


class TestSafeWriteJson:
    """Functional tests for safe_write_json."""

    def test_writes_valid_json(self, tmp_path):
        """Test writing valid JSON data."""
        from utils.file_utils import safe_read_json, safe_write_json

        data = {"name": "test", "items": [1, 2, 3]}
        test_file = tmp_path / "output.json"

        result = safe_write_json(test_file, data)
        assert result is True
        assert safe_read_json(test_file) == data

    def test_creates_parent_directories(self, tmp_path):
        """Test creating parent directories."""
        from utils.file_utils import safe_write_json

        nested_file = tmp_path / "a" / "b" / "c" / "test.json"
        result = safe_write_json(nested_file, {"nested": True})

        assert result is True
        assert nested_file.exists()

    def test_respects_create_parents_false(self, tmp_path):
        """Test respecting create_parents=False."""
        from utils.file_utils import safe_write_json

        nested_file = tmp_path / "nonexistent" / "test.json"
        result = safe_write_json(nested_file, {"data": 1}, create_parents=False)

        assert result is False
        assert not nested_file.exists()

    def test_returns_false_for_non_serializable(self, tmp_path):
        """Test returning False for non-serializable data."""
        from utils.file_utils import safe_write_json

        class NotSerializable:
            pass

        result = safe_write_json(tmp_path / "test.json", NotSerializable())
        assert result is False

    def test_uses_specified_indent(self, tmp_path):
        """Test using specified indent."""
        from utils.file_utils import safe_write_json

        test_file = tmp_path / "indented.json"
        safe_write_json(test_file, {"key": "value"}, indent=4)

        content = test_file.read_text()
        assert "    " in content  # 4-space indent


class TestEnsureDirectoryContract:
    """Design by Contract tests for ensure_directory function."""

    def test_returns_bool(self, tmp_path):
        """Postcondition: Returns a boolean."""
        from utils.file_utils import ensure_directory

        result = ensure_directory(tmp_path / "new_dir")
        assert isinstance(result, bool)


class TestEnsureDirectory:
    """Functional tests for ensure_directory."""

    def test_creates_directory_when_missing(self, tmp_path):
        """Test creating directory when it doesn't exist."""
        from utils.file_utils import ensure_directory

        new_dir = tmp_path / "new_directory"
        result = ensure_directory(new_dir)

        assert result is True
        assert new_dir.is_dir()

    def test_returns_true_for_existing_directory(self, tmp_path):
        """Test returning True for existing directory."""
        from utils.file_utils import ensure_directory

        result = ensure_directory(tmp_path)
        assert result is True

    def test_returns_false_for_file_path(self, tmp_path):
        """Test returning False when path is a file."""
        from utils.file_utils import ensure_directory

        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        result = ensure_directory(test_file)
        assert result is False

    def test_respects_create_false(self, tmp_path):
        """Test respecting create=False."""
        from utils.file_utils import ensure_directory

        new_dir = tmp_path / "will_not_create"
        result = ensure_directory(new_dir, create=False)

        assert result is False
        assert not new_dir.exists()

    def test_creates_nested_directories(self, tmp_path):
        """Test creating nested directories."""
        from utils.file_utils import ensure_directory

        nested = tmp_path / "a" / "b" / "c"
        result = ensure_directory(nested)

        assert result is True
        assert nested.is_dir()


class TestFindFileUpwardsContract:
    """Design by Contract tests for find_file_upwards function."""

    def test_returns_path_or_none(self, tmp_path):
        """Postcondition: Returns Path or None."""
        from utils.file_utils import find_file_upwards

        result = find_file_upwards("nonexistent.txt", tmp_path)
        assert result is None or isinstance(result, Path)


class TestFindFileUpwards:
    """Functional tests for find_file_upwards."""

    def test_finds_file_in_start_directory(self, tmp_path):
        """Test finding file in start directory."""
        from utils.file_utils import find_file_upwards

        test_file = tmp_path / "config.json"
        test_file.write_text("{}")

        result = find_file_upwards("config.json", tmp_path)
        assert result == test_file

    def test_finds_file_in_parent_directory(self, tmp_path):
        """Test finding file in parent directory."""
        from utils.file_utils import find_file_upwards

        # Create file in parent
        config_file = tmp_path / "pyproject.toml"
        config_file.write_text("[tool.pytest]")

        # Search from child directory
        child_dir = tmp_path / "src" / "module"
        child_dir.mkdir(parents=True)

        result = find_file_upwards("pyproject.toml", child_dir)
        assert result == config_file

    def test_returns_none_when_not_found(self, tmp_path):
        """Test returning None when file not found."""
        from utils.file_utils import find_file_upwards

        result = find_file_upwards("nonexistent_file.xyz", tmp_path, max_depth=3)
        assert result is None

    def test_respects_max_depth(self, tmp_path):
        """Test respecting max_depth parameter."""
        from utils.file_utils import find_file_upwards

        # Create file in root
        root_file = tmp_path / "marker.txt"
        root_file.write_text("marker")

        # Create deep directory
        deep_dir = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep_dir.mkdir(parents=True)

        # With low max_depth, shouldn't find file
        result = find_file_upwards("marker.txt", deep_dir, max_depth=2)
        assert result is None


class TestSafeReadTextContract:
    """Design by Contract tests for safe_read_text function."""

    def test_returns_string(self, tmp_path):
        """Postcondition: Returns a string."""
        from utils.file_utils import safe_read_text

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = safe_read_text(test_file)
        assert isinstance(result, str)


class TestSafeReadText:
    """Functional tests for safe_read_text."""

    def test_reads_text_file(self, tmp_path):
        """Test reading text file."""
        from utils.file_utils import safe_read_text

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = safe_read_text(test_file)
        assert result == "Hello, World!"

    def test_returns_default_for_missing_file(self, tmp_path):
        """Test returning default for missing file."""
        from utils.file_utils import safe_read_text

        result = safe_read_text(tmp_path / "missing.txt", default="fallback")
        assert result == "fallback"

    def test_returns_empty_string_default(self, tmp_path):
        """Test returning empty string as default."""
        from utils.file_utils import safe_read_text

        result = safe_read_text(tmp_path / "missing.txt")
        assert result == ""


class TestSafeWriteTextContract:
    """Design by Contract tests for safe_write_text function."""

    def test_returns_bool(self, tmp_path):
        """Postcondition: Returns a boolean."""
        from utils.file_utils import safe_write_text

        result = safe_write_text(tmp_path / "test.txt", "content")
        assert isinstance(result, bool)


class TestSafeWriteText:
    """Functional tests for safe_write_text."""

    def test_writes_text_file(self, tmp_path):
        """Test writing text file."""
        from utils.file_utils import safe_write_text

        test_file = tmp_path / "output.txt"
        result = safe_write_text(test_file, "Test content")

        assert result is True
        assert test_file.read_text() == "Test content"

    def test_creates_parent_directories(self, tmp_path):
        """Test creating parent directories."""
        from utils.file_utils import safe_write_text

        nested_file = tmp_path / "deep" / "path" / "file.txt"
        result = safe_write_text(nested_file, "Nested content")

        assert result is True
        assert nested_file.read_text() == "Nested content"

    def test_respects_create_parents_false(self, tmp_path):
        """Test respecting create_parents=False."""
        from utils.file_utils import safe_write_text

        nested_file = tmp_path / "nonexistent" / "file.txt"
        result = safe_write_text(nested_file, "content", create_parents=False)

        assert result is False
