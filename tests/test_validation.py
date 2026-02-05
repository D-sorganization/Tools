"""Tests for validation - Shared validation utilities.

These tests verify the validation functions using
Design by Contract principles.
"""


class TestValidatePathContract:
    """Design by Contract tests for validate_path function."""

    def test_returns_tuple(self):
        """Postcondition: Returns a tuple."""
        from utils.validation import validate_path

        result = validate_path("/some/path", must_exist=False)
        assert isinstance(result, tuple)

    def test_returns_tuple_of_two(self):
        """Postcondition: Returns tuple of (bool, str|None)."""
        from utils.validation import validate_path

        is_valid, error_msg = validate_path("/some/path", must_exist=False)
        assert isinstance(is_valid, bool)
        assert error_msg is None or isinstance(error_msg, str)


class TestValidatePath:
    """Functional tests for validate_path function."""

    def test_existing_file_passes(self, tmp_path):
        """Test that existing file passes validation."""
        from utils.validation import validate_path

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        is_valid, error = validate_path(test_file, must_exist=True)

        assert is_valid is True
        assert error is None

    def test_nonexistent_path_fails_when_must_exist(self, tmp_path):
        """Test that nonexistent path fails when must_exist=True."""
        from utils.validation import validate_path

        nonexistent = tmp_path / "does_not_exist.txt"

        is_valid, error = validate_path(nonexistent, must_exist=True)

        assert is_valid is False
        assert "does not exist" in error.lower()

    def test_nonexistent_path_passes_when_not_required(self, tmp_path):
        """Test that nonexistent path passes when must_exist=False."""
        from utils.validation import validate_path

        nonexistent = tmp_path / "does_not_exist.txt"

        is_valid, error = validate_path(nonexistent, must_exist=False)

        assert is_valid is True
        assert error is None

    def test_must_be_file_passes_for_file(self, tmp_path):
        """Test that must_be_file passes for actual file."""
        from utils.validation import validate_path

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        is_valid, error = validate_path(test_file, must_be_file=True)

        assert is_valid is True

    def test_must_be_file_fails_for_directory(self, tmp_path):
        """Test that must_be_file fails for directory."""
        from utils.validation import validate_path

        is_valid, error = validate_path(tmp_path, must_be_file=True)

        assert is_valid is False
        assert "not a file" in error.lower()

    def test_must_be_dir_passes_for_directory(self, tmp_path):
        """Test that must_be_dir passes for directory."""
        from utils.validation import validate_path

        is_valid, error = validate_path(tmp_path, must_be_dir=True)

        assert is_valid is True

    def test_must_be_dir_fails_for_file(self, tmp_path):
        """Test that must_be_dir fails for file."""
        from utils.validation import validate_path

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        is_valid, error = validate_path(test_file, must_be_dir=True)

        assert is_valid is False
        assert "not a directory" in error.lower()

    def test_must_be_within_passes_for_subpath(self, tmp_path):
        """Test path within base directory passes."""
        from utils.validation import validate_path

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        is_valid, error = validate_path(
            subdir, must_exist=True, must_be_within=tmp_path
        )

        assert is_valid is True

    def test_must_be_within_fails_for_outside_path(self, tmp_path):
        """Test path outside base directory fails (security check)."""
        from utils.validation import validate_path

        # Parent directory is outside tmp_path
        outside = tmp_path.parent

        is_valid, error = validate_path(
            outside, must_exist=True, must_be_within=tmp_path
        )

        assert is_valid is False
        assert "security" in error.lower() or "outside" in error.lower()

    def test_accepts_string_path(self, tmp_path):
        """Test that string paths are accepted."""
        from utils.validation import validate_path

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        is_valid, error = validate_path(str(test_file), must_exist=True)

        assert is_valid is True


class TestValidateFileExtensionContract:
    """Design by Contract tests for validate_file_extension."""

    def test_returns_tuple(self):
        """Postcondition: Returns a tuple."""
        from utils.validation import validate_file_extension

        result = validate_file_extension("test.txt", [".txt"])
        assert isinstance(result, tuple)

    def test_returns_tuple_of_two(self):
        """Postcondition: Returns tuple of (bool, str|None)."""
        from utils.validation import validate_file_extension

        is_valid, error_msg = validate_file_extension("test.txt", [".txt"])
        assert isinstance(is_valid, bool)
        assert error_msg is None or isinstance(error_msg, str)


class TestValidateFileExtension:
    """Functional tests for validate_file_extension."""

    def test_valid_extension_passes(self):
        """Test that valid extension passes."""
        from utils.validation import validate_file_extension

        is_valid, error = validate_file_extension("document.pdf", [".pdf", ".txt"])

        assert is_valid is True
        assert error is None

    def test_invalid_extension_fails(self):
        """Test that invalid extension fails."""
        from utils.validation import validate_file_extension

        is_valid, error = validate_file_extension("script.exe", [".pdf", ".txt"])

        assert is_valid is False
        assert ".exe" in error
        assert "not allowed" in error.lower()

    def test_case_insensitive_by_default(self):
        """Test case-insensitive matching by default."""
        from utils.validation import validate_file_extension

        is_valid, error = validate_file_extension("FILE.PDF", [".pdf"])

        assert is_valid is True

    def test_case_sensitive_option(self):
        """Test case-sensitive matching when specified."""
        from utils.validation import validate_file_extension

        is_valid, error = validate_file_extension(
            "FILE.PDF", [".pdf"], case_sensitive=True
        )

        assert is_valid is False

    def test_extension_without_dot_handled(self):
        """Test that extensions without leading dot are handled."""
        from utils.validation import validate_file_extension

        is_valid, error = validate_file_extension("file.txt", ["txt", "pdf"])

        assert is_valid is True

    def test_mixed_dot_and_no_dot_extensions(self):
        """Test mix of dotted and non-dotted extensions."""
        from utils.validation import validate_file_extension

        is_valid, error = validate_file_extension("file.csv", [".txt", "csv"])

        assert is_valid is True


class TestValidatePythonVersionContract:
    """Design by Contract tests for validate_python_version."""

    def test_returns_tuple(self):
        """Postcondition: Returns a tuple."""
        from utils.validation import validate_python_version

        result = validate_python_version()
        assert isinstance(result, tuple)

    def test_returns_tuple_of_bool_and_string(self):
        """Postcondition: Returns (bool, str)."""
        from utils.validation import validate_python_version

        is_valid, version_str = validate_python_version()
        assert isinstance(is_valid, bool)
        assert isinstance(version_str, str)


class TestValidatePythonVersion:
    """Functional tests for validate_python_version."""

    def test_current_python_passes_low_requirements(self):
        """Test current Python passes low version requirements."""
        from utils.validation import validate_python_version

        # Python 3.0 should be satisfied by any modern Python
        is_valid, version_str = validate_python_version(min_major=3, min_minor=0)

        assert is_valid is True
        assert "3." in version_str

    def test_fails_for_future_version(self):
        """Test that future version requirement fails."""
        from utils.validation import validate_python_version

        # Python 99.0 doesn't exist yet
        is_valid, error_msg = validate_python_version(min_major=99, min_minor=0)

        assert is_valid is False
        assert "required" in error_msg.lower()

    def test_version_string_format(self):
        """Test that version string has correct format."""
        from utils.validation import validate_python_version

        is_valid, version_str = validate_python_version()

        # Should be in format X.Y.Z
        parts = version_str.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class TestValidateNotNoneContract:
    """Design by Contract tests for validate_not_none."""

    def test_returns_tuple(self):
        """Postcondition: Returns a tuple."""
        from utils.validation import validate_not_none

        result = validate_not_none("value")
        assert isinstance(result, tuple)


class TestValidateNotNone:
    """Functional tests for validate_not_none."""

    def test_non_none_value_passes(self):
        """Test that non-None values pass."""
        from utils.validation import validate_not_none

        is_valid, error = validate_not_none("value")

        assert is_valid is True
        assert error is None

    def test_none_value_fails(self):
        """Test that None fails."""
        from utils.validation import validate_not_none

        is_valid, error = validate_not_none(None, name="config")

        assert is_valid is False
        assert "config" in error
        assert "None" in error

    def test_empty_string_passes(self):
        """Test that empty string (not None) passes."""
        from utils.validation import validate_not_none

        is_valid, error = validate_not_none("")

        assert is_valid is True

    def test_zero_passes(self):
        """Test that zero (not None) passes."""
        from utils.validation import validate_not_none

        is_valid, error = validate_not_none(0)

        assert is_valid is True

    def test_false_passes(self):
        """Test that False (not None) passes."""
        from utils.validation import validate_not_none

        is_valid, error = validate_not_none(False)

        assert is_valid is True


class TestValidateNotEmptyContract:
    """Design by Contract tests for validate_not_empty."""

    def test_returns_tuple(self):
        """Postcondition: Returns a tuple."""
        from utils.validation import validate_not_empty

        result = validate_not_empty("value")
        assert isinstance(result, tuple)


class TestValidateNotEmpty:
    """Functional tests for validate_not_empty."""

    def test_non_empty_string_passes(self):
        """Test that non-empty string passes."""
        from utils.validation import validate_not_empty

        is_valid, error = validate_not_empty("content")

        assert is_valid is True
        assert error is None

    def test_empty_string_fails(self):
        """Test that empty string fails."""
        from utils.validation import validate_not_empty

        is_valid, error = validate_not_empty("", name="username")

        assert is_valid is False
        assert "username" in error
        assert "empty" in error.lower()

    def test_non_empty_list_passes(self):
        """Test that non-empty list passes."""
        from utils.validation import validate_not_empty

        is_valid, error = validate_not_empty([1, 2, 3])

        assert is_valid is True

    def test_empty_list_fails(self):
        """Test that empty list fails."""
        from utils.validation import validate_not_empty

        is_valid, error = validate_not_empty([], name="items")

        assert is_valid is False

    def test_non_empty_dict_passes(self):
        """Test that non-empty dict passes."""
        from utils.validation import validate_not_empty

        is_valid, error = validate_not_empty({"key": "value"})

        assert is_valid is True

    def test_empty_dict_fails(self):
        """Test that empty dict fails."""
        from utils.validation import validate_not_empty

        is_valid, error = validate_not_empty({}, name="config")

        assert is_valid is False


class TestValidateInRangeContract:
    """Design by Contract tests for validate_in_range."""

    def test_returns_tuple(self):
        """Postcondition: Returns a tuple."""
        from utils.validation import validate_in_range

        result = validate_in_range(5, min_val=0, max_val=10)
        assert isinstance(result, tuple)


class TestValidateInRange:
    """Functional tests for validate_in_range."""

    def test_value_in_range_passes(self):
        """Test that value within range passes."""
        from utils.validation import validate_in_range

        is_valid, error = validate_in_range(5, min_val=0, max_val=10)

        assert is_valid is True
        assert error is None

    def test_value_at_minimum_passes(self):
        """Test that value at minimum (inclusive) passes."""
        from utils.validation import validate_in_range

        is_valid, error = validate_in_range(0, min_val=0, max_val=10)

        assert is_valid is True

    def test_value_at_maximum_passes(self):
        """Test that value at maximum (inclusive) passes."""
        from utils.validation import validate_in_range

        is_valid, error = validate_in_range(10, min_val=0, max_val=10)

        assert is_valid is True

    def test_value_below_minimum_fails(self):
        """Test that value below minimum fails."""
        from utils.validation import validate_in_range

        is_valid, error = validate_in_range(-1, min_val=0, max_val=10, name="count")

        assert is_valid is False
        assert "count" in error
        assert "less than minimum" in error.lower()

    def test_value_above_maximum_fails(self):
        """Test that value above maximum fails."""
        from utils.validation import validate_in_range

        is_valid, error = validate_in_range(11, min_val=0, max_val=10, name="count")

        assert is_valid is False
        assert "count" in error
        assert "greater than maximum" in error.lower()

    def test_only_min_specified(self):
        """Test with only minimum specified."""
        from utils.validation import validate_in_range

        is_valid, error = validate_in_range(100, min_val=0)

        assert is_valid is True

    def test_only_max_specified(self):
        """Test with only maximum specified."""
        from utils.validation import validate_in_range

        is_valid, error = validate_in_range(-100, max_val=0)

        assert is_valid is True

    def test_float_values(self):
        """Test with float values."""
        from utils.validation import validate_in_range

        is_valid, error = validate_in_range(0.5, min_val=0.0, max_val=1.0)

        assert is_valid is True

    def test_no_bounds_always_passes(self):
        """Test that no bounds always passes."""
        from utils.validation import validate_in_range

        is_valid, error = validate_in_range(float("inf"))

        assert is_valid is True
