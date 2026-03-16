"""Unit tests for folder_tool/file_validation.py."""

from pathlib import Path
from unittest.mock import patch

import pytest

from folder_tool.file_validation import FileValidationMixin


class DummyVar:
    def __init__(self, value=""):
        self._val = value

    def get(self):
        return self._val

    def set(self, value):
        self._val = value


class DummyApp(FileValidationMixin):
    def __init__(self):
        self.cancel_operation = False
        self.filter_extensions = DummyVar("")
        self.min_file_size = DummyVar("")
        self.max_file_size = DummyVar("")
        self.organize_by_type_var = DummyVar(False)
        self.organize_by_date_var = DummyVar(False)


@pytest.fixture
def app():
    return DummyApp()


@pytest.fixture
def sample_file(tmp_path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Hello World!")
    return str(file_path)


class TestFileValidation:
    def test_validate_file_filters_cancel(self, app, sample_file):
        app.cancel_operation = True
        assert not app.validate_file_filters(sample_file)

    def test_validate_file_filters_extension(self, app, sample_file):
        app.filter_extensions.set(".jpg, .png")
        assert not app.validate_file_filters(sample_file)

        app.filter_extensions.set(".txt")
        assert app.validate_file_filters(sample_file)

    def test_validate_file_filters_size_min(self, app, sample_file):
        # File is 12 bytes
        app.min_file_size.set("1")  # 1 MB
        assert not app.validate_file_filters(sample_file)

        app.min_file_size.set("-1")
        assert app.validate_file_filters(sample_file)
        assert app.min_file_size.get() == "0"

    def test_validate_file_filters_size_max(self, app, sample_file):
        app.max_file_size.set("0.000001")  # Small max size (<12 bytes)
        assert not app.validate_file_filters(sample_file)

        app.max_file_size.set("-1")
        assert app.validate_file_filters(sample_file)
        # Should reset to max
        assert app.max_file_size.get() == "1024"

        app.max_file_size.set("2000")  # Exceeds MAX
        assert not app.validate_file_filters(sample_file)
        assert app.max_file_size.get() == "1024"

    def test_validate_file_filters_invalid_size_str(self, app, sample_file):
        app.max_file_size.set("abc")
        assert not app.validate_file_filters(sample_file)
        assert app.max_file_size.get() == ""

    def test_validate_file_filters_exception(self, app):
        # Provide invalid path to force OSError/ValueError mapping to false
        assert not app.validate_file_filters("non_existent_file.txt")

    @patch("folder_tool.file_validation.messagebox")
    def test_validate_size_inputs(self, mock_mb, app):
        # Valid bounds
        app.min_file_size.set("1")
        app.max_file_size.set("10")
        assert app.validate_size_inputs()

        # Min > Max
        app.min_file_size.set("10")
        app.max_file_size.set("1")
        assert not app.validate_size_inputs()
        mock_mb.showwarning.assert_called()

        # Min negative
        app.min_file_size.set("-1")
        assert not app.validate_size_inputs()
        assert app.min_file_size.get() == "0"

        # Min excessive
        app.min_file_size.set("2000")
        assert not app.validate_size_inputs()
        assert app.min_file_size.get() == "0"

        # Max negative
        app.min_file_size.set("0")
        app.max_file_size.set("-1")
        assert not app.validate_size_inputs()
        assert app.max_file_size.get() == "1024"

        # Max excessive
        app.max_file_size.set("2000")
        assert not app.validate_size_inputs()
        assert app.max_file_size.get() == "1024"

        # Invalid str
        app.max_file_size.set("abc")
        assert not app.validate_size_inputs()
        mock_mb.showerror.assert_called()

    def test_get_organized_path(self, app, sample_file, tmp_path):
        dest_base = str(tmp_path / "dest")

        # Original behavior
        res = app.get_organized_path(sample_file, dest_base)
        assert res == str(Path(dest_base) / "test_file.txt")

        # Organize by type
        app.organize_by_type_var.set(True)
        res = app.get_organized_path(sample_file, dest_base)
        assert res == str(Path(dest_base) / "Documents" / "test_file.txt")

        # Organize by date
        app.organize_by_type_var.set(False)
        app.organize_by_date_var.set(True)
        res = app.get_organized_path(sample_file, dest_base)
        # We can't be sure of the exact date, but it should be %Y/%m
        assert "test_file.txt" in res

        # Test unknown date gracefully falling back to "Unknown_Date"
        with patch("os.path.getmtime", side_effect=OSError):
            res = app.get_organized_path(sample_file, dest_base)
            assert str(Path("Unknown_Date") / "test_file.txt") in res
