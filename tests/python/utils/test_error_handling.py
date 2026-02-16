"""Tests for python.src.utils.error_handling module.

Covers:
- handle_file_errors decorator
- safe_execute function
- handle_import_error function
- log_and_continue decorator
"""

from __future__ import annotations

import pytest
from utils.error_handling import (
    handle_file_errors,
    handle_import_error,
    log_and_continue,
    safe_execute,
)


class TestHandleFileErrors:
    """Tests for handle_file_errors decorator."""

    def test_successful_function(self) -> None:
        @handle_file_errors(default=-1)
        def good_func() -> int:
            return 42

        assert good_func() == 42

    def test_file_not_found_returns_default(self) -> None:
        @handle_file_errors(default="fallback")
        def bad_func() -> str:
            raise FileNotFoundError("missing")

        assert bad_func() == "fallback"

    def test_permission_error_returns_default(self) -> None:
        @handle_file_errors(default=None)
        def locked_func() -> None:
            raise PermissionError("no access")

        assert locked_func() is None

    def test_reraise_propagates(self) -> None:
        @handle_file_errors(default=None, reraise=True)
        def failing_func() -> None:
            raise OSError("disk error")

        with pytest.raises(OSError):
            failing_func()

    def test_generic_exception_returns_default(self) -> None:
        @handle_file_errors(default="safe")
        def broken_func() -> str:
            raise RuntimeError("boom")

        assert broken_func() == "safe"


class TestSafeExecute:
    """Tests for safe_execute function."""

    def test_successful_execution(self) -> None:
        result = safe_execute(lambda: 42, default=-1)
        assert result == 42

    def test_error_returns_default(self) -> None:
        def fail() -> None:
            raise ValueError("bad")

        result = safe_execute(fail, default="oops")
        assert result == "oops"

    def test_with_args(self) -> None:
        result = safe_execute(lambda x, y: x + y, 3, 4, default=0)
        assert result == 7

    def test_none_default(self) -> None:
        result = safe_execute(lambda: 1 / 0)
        assert result is None


class TestHandleImportError:
    """Tests for handle_import_error function."""

    def test_successful_import(self) -> None:
        result = handle_import_error("os")
        assert result is not None

    def test_missing_module_returns_default(self) -> None:
        result = handle_import_error(
            "nonexistent_module_xyz", default="not_found"
        )
        assert result == "not_found"

    def test_missing_module_returns_none(self) -> None:
        result = handle_import_error("nonexistent_module_xyz")
        assert result is None


class TestLogAndContinue:
    """Tests for log_and_continue decorator."""

    def test_successful_function(self) -> None:
        @log_and_continue("something went wrong", default=-1)
        def good_func() -> int:
            return 99

        assert good_func() == 99

    def test_error_returns_default(self) -> None:
        @log_and_continue("caught error", default=0)
        def bad_func() -> int:
            raise ValueError("fail")

        assert bad_func() == 0

    def test_preserves_function_name(self) -> None:
        @log_and_continue("err")
        def named_func() -> None:
            pass

        assert named_func.__name__ == "named_func"
