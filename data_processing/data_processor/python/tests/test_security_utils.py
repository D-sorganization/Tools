"""Tests for security_utils.py - comprehensive security testing."""

from pathlib import Path
from unittest.mock import patch

import pytest
from data_processor.security_utils import (
    ExpressionValidationError,
    FileSizeError,
    PathValidationError,
    SecurityError,
    check_file_size,
    get_safe_file_info,
    validate_and_check_file,
    validate_file_path,
    validate_python_expression,
)


class TestExceptionHierarchy:
    """Test the security exception hierarchy."""

    def test_security_error_base(self) -> None:
        """Test SecurityError is base class for all security exceptions."""
        assert issubclass(PathValidationError, SecurityError)
        assert issubclass(FileSizeError, SecurityError)
        assert issubclass(ExpressionValidationError, SecurityError)

    def test_exceptions_can_be_raised(self) -> None:
        """Test that all security exceptions can be raised and caught."""
        with pytest.raises(SecurityError):
            raise SecurityError("Test error")

        with pytest.raises(PathValidationError):
            raise PathValidationError("Path error")

        with pytest.raises(FileSizeError):
            raise FileSizeError("File size error")

        with pytest.raises(ExpressionValidationError):
            raise ExpressionValidationError("Expression error")


class TestValidatePythonExpression:
    """Test Python expression validation for formula builder."""

    def test_simple_arithmetic_allowed(self) -> None:
        """Test simple arithmetic expressions are allowed."""
        validate_python_expression("1 + 2")
        validate_python_expression("10 - 5")
        validate_python_expression("3 * 4")
        validate_python_expression("8 / 2")
        validate_python_expression("2 ** 3")
        validate_python_expression("10 % 3")

    def test_complex_arithmetic_allowed(self) -> None:
        """Test complex arithmetic expressions are allowed."""
        validate_python_expression("(1 + 2) * 3")
        validate_python_expression("((10 - 5) * 2) + 3")
        validate_python_expression("-5 + 10")
        validate_python_expression("+5")

    def test_variables_with_allowed_names(self) -> None:
        """Test expressions with variables when names are allowed."""
        allowed = {"x", "y", "z", "data", "value"}
        validate_python_expression("x + y", allowed_names=allowed)
        validate_python_expression("data * 2", allowed_names=allowed)
        validate_python_expression("(x + y) / z", allowed_names=allowed)

    def test_variables_without_allowed_names(self) -> None:
        """Test expressions with variables when names not provided."""
        # When allowed_names is None, any variable name is allowed
        validate_python_expression("x + y", allowed_names=None)
        validate_python_expression("foo * bar", allowed_names=None)

    def test_unknown_variable_rejected(self) -> None:
        """Test expressions with unknown variables are rejected."""
        allowed = {"x", "y"}
        with pytest.raises(ExpressionValidationError, match="Unknown identifier: z"):
            validate_python_expression("x + z", allowed_names=allowed)

    def test_function_calls_allowed(self) -> None:
        """Test function calls with allowed names."""
        allowed = {"abs", "max", "min", "sum", "x", "y"}
        validate_python_expression("abs(x)", allowed_names=allowed)
        validate_python_expression("max(1, 2)", allowed_names=allowed)
        validate_python_expression("sum(x, y)", allowed_names=allowed)

    def test_unknown_function_rejected(self) -> None:
        """Test unknown functions are rejected."""
        allowed = {"x", "y"}
        with pytest.raises(
            ExpressionValidationError, match="Unknown function: unknown_func"
        ):
            validate_python_expression("unknown_func(x)", allowed_names=allowed)

    def test_comparisons_allowed(self) -> None:
        """Test comparison operations are allowed."""
        validate_python_expression("1 < 2")
        validate_python_expression("5 > 3")
        validate_python_expression("4 == 4")
        validate_python_expression("3 != 2")
        validate_python_expression("1 <= 1")
        validate_python_expression("2 >= 1")

    def test_boolean_operations_allowed(self) -> None:
        """Test boolean operations are allowed."""
        validate_python_expression("True and False")
        validate_python_expression("True or False")
        validate_python_expression("not True")
        validate_python_expression("1 < 2 and 3 > 1")

    def test_strings_and_numbers_allowed(self) -> None:
        """Test string and number constants are allowed."""
        validate_python_expression("'hello'")
        validate_python_expression('"world"')
        validate_python_expression("123")
        validate_python_expression("45.67")
        validate_python_expression("True")
        validate_python_expression("False")
        validate_python_expression("None")

    def test_invalid_syntax_rejected(self) -> None:
        """Test invalid Python syntax is rejected."""
        with pytest.raises(ExpressionValidationError, match="Invalid syntax"):
            validate_python_expression("1 +")

        with pytest.raises(ExpressionValidationError, match="Invalid syntax"):
            validate_python_expression("(1 + 2")

    def test_imports_rejected(self) -> None:
        """Test import statements are rejected."""
        with pytest.raises(ExpressionValidationError, match="Invalid syntax"):
            validate_python_expression("import os")

    def test_attribute_access_rejected(self) -> None:
        """Test attribute access is rejected."""
        with pytest.raises(
            ExpressionValidationError,
            match="Complex function calls are not allowed|Disallowed syntax",
        ):
            validate_python_expression("obj.method()")

    def test_list_comprehension_rejected(self) -> None:
        """Test list comprehensions are rejected."""
        with pytest.raises(ExpressionValidationError, match="Disallowed syntax"):
            validate_python_expression("[x for x in range(10)]")

    def test_complex_calls_rejected(self) -> None:
        """Test complex function calls are rejected."""
        with pytest.raises(
            ExpressionValidationError, match="Complex function calls are not allowed"
        ):
            validate_python_expression("f()()")


class TestValidateFilePath:
    """Test file path validation."""

    def test_valid_file_in_cwd(self, tmp_path: Path) -> None:
        """Test valid file in current working directory."""
        test_file = tmp_path / "test.csv"
        test_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            validated = validate_file_path(test_file)
            assert validated == test_file.resolve()

    def test_valid_file_with_allowed_extension(self, tmp_path: Path) -> None:
        """Test file with allowed extension."""
        test_file = tmp_path / "test.csv"
        test_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            validated = validate_file_path(
                test_file, allowed_extensions={".csv", ".xlsx"}
            )
            assert validated == test_file.resolve()

    def test_file_with_disallowed_extension_rejected(self, tmp_path: Path) -> None:
        """Test file with disallowed extension is rejected."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(PathValidationError, match="Unsupported file extension"):
                validate_file_path(test_file, allowed_extensions={".csv", ".xlsx"})

    def test_nonexistent_file_rejected(self, tmp_path: Path) -> None:
        """Test nonexistent file is rejected."""
        test_file = tmp_path / "nonexistent.csv"

        with pytest.raises(PathValidationError, match="File does not exist"):
            validate_file_path(test_file)

    def test_directory_rejected(self, tmp_path: Path) -> None:
        """Test directory path is rejected."""
        with pytest.raises(PathValidationError, match="Path is not a file"):
            validate_file_path(tmp_path)

    def test_file_outside_cwd_rejected(self, tmp_path: Path) -> None:
        """Test file outside CWD is rejected when allow_anywhere=False."""
        test_file = tmp_path / "test.csv"
        test_file.touch()

        # Create a different directory as CWD
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        with patch("pathlib.Path.cwd", return_value=other_dir):
            with patch("pathlib.Path.home", return_value=other_dir):
                with pytest.raises(
                    PathValidationError, match="Path outside allowed directories"
                ):
                    validate_file_path(test_file, allow_anywhere=False)

    def test_file_outside_cwd_allowed_with_flag(self, tmp_path: Path) -> None:
        """Test file outside CWD is allowed when allow_anywhere=True."""
        test_file = tmp_path / "test.csv"
        test_file.touch()

        # Create a different directory as CWD
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        with patch("pathlib.Path.cwd", return_value=other_dir):
            validated = validate_file_path(test_file, allow_anywhere=True)
            assert validated == test_file.resolve()

    def test_string_path_converted_to_path(self, tmp_path: Path) -> None:
        """Test string path is converted to Path object."""
        test_file = tmp_path / "test.csv"
        test_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            validated = validate_file_path(str(test_file))
            assert isinstance(validated, Path)
            assert validated == test_file.resolve()


class TestCheckFileSize:
    """Test file size checking."""

    def test_small_file_passes(self, tmp_path: Path) -> None:
        """Test file within size limit passes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Small content")

        # Should not raise exception
        check_file_size(test_file, max_size_bytes=1024)

    def test_large_file_rejected(self, tmp_path: Path) -> None:
        """Test file exceeding size limit is rejected."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("X" * 2000)

        with pytest.raises(FileSizeError, match="File too large"):
            check_file_size(test_file, max_size_bytes=1000)

    def test_exact_size_limit_passes(self, tmp_path: Path) -> None:
        """Test file at exact size limit passes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("X" * 1000)

        # Should not raise exception
        check_file_size(test_file, max_size_bytes=1000)

    def test_nonexistent_file_rejected(self, tmp_path: Path) -> None:
        """Test nonexistent file is rejected."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileSizeError, match="File does not exist"):
            check_file_size(test_file)

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """Test string path is accepted."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        # Should not raise exception
        check_file_size(str(test_file), max_size_bytes=1024)

    def test_error_message_includes_sizes(self, tmp_path: Path) -> None:
        """Test error message includes human-readable sizes."""
        test_file = tmp_path / "test.txt"
        # Write 2 GB worth of data (simulated with seek)
        with test_file.open("wb") as f:
            f.seek(2 * 1024**3 - 1)
            f.write(b"\0")

        with pytest.raises(FileSizeError, match=r"\d+\.\d+ GB.*max:.*\d+\.\d+ GB"):
            check_file_size(test_file, max_size_bytes=1024**3)


class TestValidateAndCheckFile:
    """Test combined validation and size checking."""

    def test_valid_file_passes_all_checks(self, tmp_path: Path) -> None:
        """Test file that passes all checks."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            validated = validate_and_check_file(
                test_file,
                allowed_extensions={".csv"},
                max_size_bytes=1024,
                allow_anywhere=False,
            )
            assert validated == test_file.resolve()

    def test_file_fails_path_validation(self, tmp_path: Path) -> None:
        """Test file that fails path validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(PathValidationError):
                validate_and_check_file(
                    test_file,
                    allowed_extensions={".csv"},  # Wrong extension
                    max_size_bytes=1024,
                )

    def test_file_fails_size_check(self, tmp_path: Path) -> None:
        """Test file that fails size check."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("X" * 2000)

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(FileSizeError):
                validate_and_check_file(
                    test_file,
                    allowed_extensions={".csv"},
                    max_size_bytes=1000,  # Too small
                )


class TestGetSafeFileInfo:
    """Test safe file information retrieval."""

    def test_valid_file_returns_info(self, tmp_path: Path) -> None:
        """Test valid file returns complete information."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data content")

        info = get_safe_file_info(test_file)

        assert info["name"] == "test.csv"
        assert info["absolute_path"] == str(test_file.resolve())
        assert info["size_bytes"] == 12
        assert info["size_mb"] >= 0  # Small files may round to 0.0
        assert info["size_gb"] >= 0  # Small files may round to 0.0
        assert info["extension"] == ".csv"
        assert info["is_file"] is True
        assert "modified_timestamp" in info
        assert "within_size_limit" in info

    def test_nonexistent_file_returns_error(self, tmp_path: Path) -> None:
        """Test nonexistent file returns error dict."""
        test_file = tmp_path / "nonexistent.csv"

        info = get_safe_file_info(test_file)

        assert "error" in info
        assert info["error"] == "File does not exist"

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """Test string path is accepted."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        info = get_safe_file_info(str(test_file))

        assert info["name"] == "test.txt"
        assert info["is_file"] is True

    def test_size_calculations_correct(self, tmp_path: Path) -> None:
        """Test size calculations are accurate."""
        test_file = tmp_path / "test.txt"
        # Write exactly 1 MB
        test_file.write_bytes(b"X" * (1024 * 1024))

        info = get_safe_file_info(test_file)

        assert info["size_bytes"] == 1024 * 1024
        assert info["size_mb"] == 1.0
        assert abs(info["size_gb"] - 0.000977) < 0.001  # ~0.000977 GB

    def test_within_size_limit_flag(self, tmp_path: Path) -> None:
        """Test within_size_limit flag is set correctly."""
        # Create a small file
        test_file = tmp_path / "small.txt"
        test_file.write_text("small")

        info = get_safe_file_info(test_file)
        assert info["within_size_limit"] is True

    def test_exception_returns_error_dict(self, tmp_path: Path) -> None:
        """Test exceptions are caught and returned as error dict."""
        # Pass an invalid path object to trigger an exception
        with patch("pathlib.Path.resolve", side_effect=RuntimeError("Test error")):
            info = get_safe_file_info(tmp_path / "test.txt")
            assert "error" in info
            assert "Test error" in info["error"]
