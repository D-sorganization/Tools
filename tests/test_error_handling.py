"""Tests for error_handling - Shared error handling utilities.

These tests verify the error handling decorators and functions
using Design by Contract principles.
"""

import logging

import pytest


class TestHandleFileErrorsContract:
    """Design by Contract tests for handle_file_errors decorator."""

    def test_returns_callable(self):
        """Postcondition: Returns a callable decorator."""
        from utils.error_handling import handle_file_errors

        decorator = handle_file_errors()
        assert callable(decorator)

    def test_decorated_function_callable(self):
        """Postcondition: Decorated function is callable."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors()
        def test_func():
            return "result"

        assert callable(test_func)

    def test_preserves_function_name(self):
        """Postcondition: Decorator preserves function name."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors()
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestHandleFileErrors:
    """Functional tests for handle_file_errors decorator."""

    def test_returns_result_on_success(self):
        """Test that successful function returns its result."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors()
        def successful_func():
            return "success"

        assert successful_func() == "success"

    def test_returns_default_on_file_not_found(self):
        """Test that FileNotFoundError returns default."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors(default="default_value", log_error=False)
        def file_error_func():
            raise FileNotFoundError("test file not found")

        assert file_error_func() == "default_value"

    def test_returns_default_on_permission_error(self):
        """Test that PermissionError returns default."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors(default=None, log_error=False)
        def permission_func():
            raise PermissionError("access denied")

        assert permission_func() is None

    def test_returns_default_on_os_error(self):
        """Test that OSError returns default."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors(default=-1, log_error=False)
        def os_error_func():
            raise OSError("disk error")

        assert os_error_func() == -1

    def test_returns_default_on_unexpected_error(self):
        """Test that unexpected errors return default."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors(default="fallback", log_error=False)
        def unexpected_error_func():
            raise ValueError("unexpected")

        assert unexpected_error_func() == "fallback"

    def test_reraise_option_works(self):
        """Test that reraise=True re-raises exceptions."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors(reraise=True, log_error=False)
        def reraise_func():
            raise FileNotFoundError("test")

        with pytest.raises(FileNotFoundError):
            reraise_func()

    def test_passes_arguments_to_function(self):
        """Test that arguments are passed correctly."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_passes_kwargs_to_function(self):
        """Test that kwargs are passed correctly."""
        from utils.error_handling import handle_file_errors

        @handle_file_errors()
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        assert greet("World", greeting="Hi") == "Hi, World!"


class TestSafeExecuteContract:
    """Design by Contract tests for safe_execute function."""

    def test_returns_result_on_success(self):
        """Postcondition: Returns function result on success."""
        from utils.error_handling import safe_execute

        result = safe_execute(lambda: 42)
        assert result == 42

    def test_returns_default_on_error(self):
        """Postcondition: Returns default on error."""
        from utils.error_handling import safe_execute

        def raise_error():
            raise RuntimeError("test")

        result = safe_execute(raise_error, default=-1, log_error=False)
        assert result == -1


class TestSafeExecute:
    """Functional tests for safe_execute function."""

    def test_executes_function_with_args(self):
        """Test function execution with positional arguments."""
        from utils.error_handling import safe_execute

        result = safe_execute(pow, 2, 3)
        assert result == 8

    def test_executes_function_with_kwargs(self):
        """Test function execution with keyword arguments."""
        from utils.error_handling import safe_execute

        def subtract(a, b=0):
            return a - b

        result = safe_execute(subtract, 10, b=3)
        assert result == 7

    def test_returns_none_by_default_on_error(self):
        """Test that default is None when not specified."""
        from utils.error_handling import safe_execute

        def fail():
            raise Exception("fail")

        result = safe_execute(fail, log_error=False)
        assert result is None

    def test_handles_any_exception(self):
        """Test that any exception is handled."""
        from utils.error_handling import safe_execute

        def custom_error():
            raise KeyError("missing key")

        result = safe_execute(custom_error, default="handled", log_error=False)
        assert result == "handled"


class TestHandleImportErrorContract:
    """Design by Contract tests for handle_import_error function."""

    def test_returns_module_on_success(self):
        """Postcondition: Returns module on successful import."""
        from utils.error_handling import handle_import_error

        result = handle_import_error("os")
        assert result is not None

    def test_returns_default_on_failure(self):
        """Postcondition: Returns default on import failure."""
        from utils.error_handling import handle_import_error

        result = handle_import_error("nonexistent_module_xyz_123", default="not_found")
        assert result == "not_found"


class TestHandleImportError:
    """Functional tests for handle_import_error function."""

    def test_imports_standard_library(self):
        """Test importing standard library module."""
        from utils.error_handling import handle_import_error

        result = handle_import_error("json")
        import json

        assert result is json

    def test_returns_none_by_default(self):
        """Test that default is None when not specified."""
        from utils.error_handling import handle_import_error

        result = handle_import_error("definitely_not_a_real_module")
        assert result is None


class TestLogAndContinueContract:
    """Design by Contract tests for log_and_continue decorator."""

    def test_returns_callable(self):
        """Postcondition: Returns a callable decorator."""
        from utils.error_handling import log_and_continue

        decorator = log_and_continue("test error")
        assert callable(decorator)

    def test_preserves_function_name(self):
        """Postcondition: Preserves decorated function name."""
        from utils.error_handling import log_and_continue

        @log_and_continue("error message")
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


class TestLogAndContinue:
    """Functional tests for log_and_continue decorator."""

    def test_returns_result_on_success(self):
        """Test successful execution returns result."""
        from utils.error_handling import log_and_continue

        @log_and_continue("error")
        def success():
            return "ok"

        assert success() == "ok"

    def test_returns_default_on_error(self):
        """Test that error returns default value."""
        from utils.error_handling import log_and_continue

        @log_and_continue("operation failed", default=0)
        def failing_func():
            raise ValueError("test")

        assert failing_func() == 0

    def test_logs_at_specified_level(self, caplog):
        """Test that logging occurs at specified level."""
        from utils.error_handling import log_and_continue

        @log_and_continue("test error", log_level=logging.ERROR)
        def error_func():
            raise RuntimeError("test")

        with caplog.at_level(logging.ERROR):
            error_func()

        assert "test error" in caplog.text


class TestExitOnErrorContract:
    """Design by Contract tests for exit_on_error decorator."""

    def test_returns_callable(self):
        """Postcondition: Returns a callable decorator."""
        from utils.error_handling import exit_on_error

        decorator = exit_on_error("error")
        assert callable(decorator)


class TestExitOnError:
    """Functional tests for exit_on_error decorator."""

    def test_returns_result_on_success(self):
        """Test successful execution returns result."""
        from utils.error_handling import exit_on_error

        @exit_on_error("should not exit")
        def success():
            return "result"

        assert success() == "result"

    def test_exits_on_error(self):
        """Test that error causes system exit."""
        from utils.error_handling import exit_on_error

        @exit_on_error("operation failed", exit_code=42, log_error=False)
        def failing_func():
            raise ValueError("test")

        with pytest.raises(SystemExit) as exc_info:
            failing_func()

        assert exc_info.value.code == 42
