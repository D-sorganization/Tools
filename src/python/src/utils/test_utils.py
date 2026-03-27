# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Comprehensive test utilities for consistent testing across the repository.

This module provides standardized testing utilities including:
- Common fixtures and fixture factories
- Custom assertions for complex comparisons
- Test data generators
- Mock helpers and factories
- Performance testing utilities
- Test case base classes
"""

import functools
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import MagicMock, patch

# Type variables for generic functions
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Constants
DEFAULT_TIMEOUT: float = 30.0  # Default timeout for async operations
SAMPLE_DATA_SIZE: int = 100  # Default sample data size


# =============================================================================
# Test Data Generators
# =============================================================================


@dataclass
class DataGeneratorConfig:
    """Configuration for test data generation."""

    size: int = SAMPLE_DATA_SIZE
    seed: int = 42
    include_nulls: bool = False
    include_edge_cases: bool = True


# Alias for backwards compatibility
TestDataConfig = DataGeneratorConfig


def generate_sample_data(
    data_type: str = "mixed",
    size: int = SAMPLE_DATA_SIZE,
    seed: int = 42,
) -> list[Any]:
    """Generate sample test data.

    Args:
        data_type: Type of data to generate ('int', 'float', 'string', 'mixed')
        size: Number of elements to generate
        seed: Random seed for reproducibility

    Returns:
        List of generated test data
    """
    import random

    random.seed(seed)

    if data_type == "int":
        return [random.randint(-1000, 1000) for _ in range(size)]
    elif data_type == "float":
        return [random.uniform(-1000.0, 1000.0) for _ in range(size)]
    elif data_type == "string":
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return [
            "".join(random.choices(chars, k=random.randint(5, 20))) for _ in range(size)
        ]
    else:  # mixed
        generators = [
            lambda: random.randint(-1000, 1000),
            lambda: random.uniform(-1000.0, 1000.0),
            lambda: "".join(
                random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(5, 15))
            ),
            lambda: random.choice([True, False]),
            lambda: None,
        ]
        return [random.choice(generators)() for _ in range(size)]


def generate_edge_case_data(data_type: str = "int") -> list[Any]:
    """Generate edge case test data for boundary testing.

    Args:
        data_type: Type of data ('int', 'float', 'string')

    Returns:
        List of edge case values
    """
    if data_type == "int":
        return [
            0,
            1,
            -1,
            2**31 - 1,  # Max 32-bit signed int
            -(2**31),  # Min 32-bit signed int
            2**63 - 1,  # Max 64-bit signed int
            -(2**63),  # Min 64-bit signed int
        ]
    elif data_type == "float":
        return [
            0.0,
            -0.0,
            1.0,
            -1.0,
            float("inf"),
            float("-inf"),
            1e-10,
            1e10,
            sys.float_info.min,
            sys.float_info.max,
            sys.float_info.epsilon,
        ]
    elif data_type == "string":
        return [
            "",
            " ",
            "   ",
            "\t",
            "\n",
            "\r\n",
            "a" * 10000,  # Long string
            "🎉🎊🎁",  # Unicode
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection attempt
            "\x00\x01\x02",  # Binary characters
            "path/../../../etc/passwd",  # Path traversal
        ]
    return []


# =============================================================================
# Custom Assertions
# =============================================================================


class AssertionHelpers:
    """Collection of custom assertion helpers for testing."""

    @staticmethod
    def assert_approx_equal(
        actual: float,
        expected: float,
        rel_tol: float = 1e-9,
        abs_tol: float = 0.0,
        msg: str | None = None,
    ) -> None:
        """Assert two floats are approximately equal.

        Args:
            actual: Actual value
            expected: Expected value
            rel_tol: Relative tolerance
            abs_tol: Absolute tolerance
            msg: Optional message on failure
        """
        import math

        if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            error_msg = msg or f"Values not approximately equal: {actual} != {expected}"
            raise AssertionError(error_msg)

    @staticmethod
    def assert_dict_subset(
        subset: dict[str, Any],
        superset: dict[str, Any],
        msg: str | None = None,
    ) -> None:
        """Assert that subset is contained within superset.

        Args:
            subset: Dictionary that should be contained
            superset: Dictionary that should contain subset
            msg: Optional message on failure
        """
        for key, value in subset.items():
            if key not in superset:
                error_msg = msg or f"Key '{key}' not found in superset"
                raise AssertionError(error_msg)
            if superset[key] != value:
                error_msg = (
                    msg or f"Value mismatch for key '{key}': {superset[key]} != {value}"
                )
                raise AssertionError(error_msg)

    @staticmethod
    def assert_json_equal(
        actual: str | dict[str, Any],
        expected: str | dict[str, Any],
        msg: str | None = None,
    ) -> None:
        """Assert two JSON values are equal (ignores formatting).

        Args:
            actual: Actual JSON (string or dict)
            expected: Expected JSON (string or dict)
            msg: Optional message on failure
        """
        if isinstance(actual, str):
            actual = json.loads(actual)
        if isinstance(expected, str):
            expected = json.loads(expected)

        if actual != expected:
            error_msg = msg or f"JSON values not equal:\n{actual}\n!=\n{expected}"
            raise AssertionError(error_msg)

    @staticmethod
    def assert_raises_with_message(
        exception_type: type[Exception],
        message_pattern: str,
        callable_obj: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Exception:
        """Assert that callable raises exception with matching message.

        Args:
            exception_type: Expected exception type
            message_pattern: Regex pattern to match exception message
            callable_obj: Callable to execute
            *args: Positional arguments for callable
            **kwargs: Keyword arguments for callable

        Returns:
            The caught exception
        """
        try:
            callable_obj(*args, **kwargs)
        except exception_type as e:
            if not re.search(message_pattern, str(e)):
                raise AssertionError(
                    f"Exception message '{e}' does not match pattern '{message_pattern}'"
                ) from e
            return e
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            raise AssertionError(
                f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}"
            ) from e
        else:
            raise AssertionError(f"Expected {exception_type.__name__} was not raised")

    @staticmethod
    def assert_file_contains(
        file_path: Path | str,
        expected_content: str,
        msg: str | None = None,
    ) -> None:
        """Assert that file contains expected content.

        Args:
            file_path: Path to file
            expected_content: Content that should be present
            msg: Optional message on failure
        """
        path = Path(file_path)
        if not path.exists():
            raise AssertionError(f"File does not exist: {file_path}")

        content = path.read_text()
        if expected_content not in content:
            error_msg = (
                msg or f"Expected content not found in {file_path}: {expected_content}"
            )
            raise AssertionError(error_msg)

    @staticmethod
    def assert_logs_contain(
        logs: list[logging.LogRecord],
        level: int,
        message_pattern: str,
        msg: str | None = None,
    ) -> None:
        """Assert that logs contain a message at specified level.

        Args:
            logs: List of log records
            level: Expected log level
            message_pattern: Regex pattern to match message
            msg: Optional message on failure
        """
        for record in logs:
            if record.levelno == level and re.search(
                message_pattern, record.getMessage()
            ):
                return
        error_msg = (
            msg
            or f"No log message matching '{message_pattern}' "
            f"at level {logging.getLevelName(level)}"
        )
        raise AssertionError(error_msg)


# Create a module-level instance for convenience
assert_helpers = AssertionHelpers()


# =============================================================================
# Mock Helpers and Factories
# =============================================================================


class MockFactory:
    """Factory for creating common mock objects."""

    @staticmethod
    def create_mock_file(
        content: str = "",
        name: str = "mock_file.txt",
        encoding: str = "utf-8",
    ) -> MagicMock:
        """Create a mock file object.

        Args:
            content: File content
            name: File name
            encoding: File encoding

        Returns:
            Mock file object
        """
        mock_file = MagicMock()
        mock_file.name = name
        mock_file.read.return_value = content
        mock_file.readlines.return_value = content.split("\n")
        mock_file.encoding = encoding
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        return mock_file

    @staticmethod
    def create_mock_response(
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> MagicMock:
        """Create a mock HTTP response object.

        Args:
            status_code: HTTP status code
            json_data: JSON response data
            text: Text response
            headers: Response headers

        Returns:
            Mock response object
        """
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data or {}
        mock_response.text = text
        mock_response.headers = headers or {}
        mock_response.ok = 200 <= status_code < 300
        mock_response.raise_for_status = MagicMock()
        if not mock_response.ok:
            mock_response.raise_for_status.side_effect = Exception(
                f"HTTP Error: {status_code}"
            )
        return mock_response

    @staticmethod
    def create_mock_logger() -> MagicMock:
        """Create a mock logger with all standard methods.

        Returns:
            Mock logger object
        """
        mock_logger = MagicMock(spec=logging.Logger)
        mock_logger.level = logging.DEBUG
        mock_logger.handlers = []
        mock_logger.name = "mock_logger"
        return mock_logger

    @staticmethod
    def create_mock_path(
        exists: bool = True,
        is_file: bool = True,
        is_dir: bool = False,
        content: str = "",
        stat_size: int = 1024,
    ) -> MagicMock:
        """Create a mock Path object.

        Args:
            exists: Whether the path exists
            is_file: Whether it's a file
            is_dir: Whether it's a directory
            content: File content (for read_text)
            stat_size: File size for stat()

        Returns:
            Mock Path object
        """
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = exists
        mock_path.is_file.return_value = is_file
        mock_path.is_dir.return_value = is_dir
        mock_path.read_text.return_value = content
        mock_path.read_bytes.return_value = content.encode()

        mock_stat = MagicMock()
        mock_stat.st_size = stat_size
        mock_stat.st_mtime = time.time()
        mock_path.stat.return_value = mock_stat

        return mock_path


# =============================================================================
# Context Managers for Testing
# =============================================================================


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after use.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@contextmanager
def temporary_file(
    content: str = "",
    suffix: str = ".txt",
    mode: str = "w",
) -> Generator[Path, None, None]:
    """Create a temporary file with content.

    Args:
        content: Initial file content
        suffix: File suffix
        mode: Write mode

    Yields:
        Path to temporary file
    """
    with tempfile.NamedTemporaryFile(
        mode=mode, suffix=suffix, delete=False
    ) as tmp_file:
        if content:
            tmp_file.write(content)
            tmp_file.flush()
        tmp_path = Path(tmp_file.name)

    try:
        yield tmp_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@contextmanager
def captured_output() -> Generator[tuple[io.StringIO, io.StringIO], None, None]:
    """Capture stdout and stderr output.

    Yields:
        Tuple of (stdout, stderr) StringIO objects
    """
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        yield new_stdout, new_stderr
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


@contextmanager
def captured_logs(
    logger_name: str | None = None,
    level: int = logging.DEBUG,
) -> Generator[list[logging.LogRecord], None, None]:
    """Capture log records from a logger.

    Args:
        logger_name: Name of logger to capture (None for root)
        level: Minimum level to capture

    Yields:
        List of captured LogRecord objects
    """
    records: list[logging.LogRecord] = []

    class RecordCapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(logger_name)
    handler = RecordCapturingHandler()
    handler.setLevel(level)
    original_level = logger.level
    logger.setLevel(level)
    logger.addHandler(handler)

    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


@contextmanager
def environment_variables(**env_vars: str) -> Generator[None, None, None]:
    """Temporarily set environment variables.

    Args:
        **env_vars: Environment variable name/value pairs

    Yields:
        None
    """
    original_vars: dict[str, str | None] = {}
    for key, value in env_vars.items():
        original_vars[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield
    finally:
        for key, original_value in original_vars.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


@contextmanager
def mock_datetime(
    target_module: str,
    frozen_time: str | None = None,
) -> Generator[MagicMock, None, None]:
    """Mock datetime in a specific module.

    Args:
        target_module: Module path to mock datetime in
        frozen_time: ISO format time string to freeze to

    Yields:
        Mock datetime object
    """
    import datetime

    if frozen_time:
        frozen_dt = datetime.datetime.fromisoformat(frozen_time)
    else:
        frozen_dt = datetime.datetime.now()

    mock_dt = MagicMock(wraps=datetime.datetime)
    mock_dt.now.return_value = frozen_dt
    mock_dt.utcnow.return_value = frozen_dt

    with patch(f"{target_module}.datetime", mock_dt):
        yield mock_dt


# =============================================================================
# Performance Testing Utilities
# =============================================================================


@dataclass
class TimingResult:
    """Result of a timing operation."""

    elapsed_seconds: float
    function_name: str
    args_repr: str = ""
    iterations: int = 1

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000

    @property
    def average_seconds(self) -> float:
        """Average time per iteration in seconds."""
        return self.elapsed_seconds / self.iterations

    @property
    def average_ms(self) -> float:
        """Average time per iteration in milliseconds."""
        return self.average_seconds * 1000


def time_function(
    func: Callable[..., T],
    *args: Any,
    iterations: int = 1,
    **kwargs: Any,
) -> tuple[T, TimingResult]:
    """Time a function execution.

    Args:
        func: Function to time
        *args: Positional arguments for function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments for function

    Returns:
        Tuple of (result, TimingResult)
    """
    start_time = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start_time

    timing_result = TimingResult(
        elapsed_seconds=elapsed,
        function_name=func.__name__,
        args_repr=f"args={args}, kwargs={kwargs}",
        iterations=iterations,
    )
    return result, timing_result  # type: ignore[return-value]


def assert_performance(
    func: Callable[..., Any],
    max_seconds: float,
    *args: Any,
    iterations: int = 1,
    **kwargs: Any,
) -> TimingResult:
    """Assert that function executes within time limit.

    Args:
        func: Function to test
        max_seconds: Maximum allowed execution time
        *args: Positional arguments for function
        iterations: Number of iterations
        **kwargs: Keyword arguments for function

    Returns:
        TimingResult with execution details

    Raises:
        AssertionError: If execution time exceeds max_seconds
    """
    _, timing = time_function(func, *args, iterations=iterations, **kwargs)

    if timing.average_seconds > max_seconds:
        raise AssertionError(
            f"Function {func.__name__} took {timing.average_ms:.2f}ms "
            f"(limit: {max_seconds * 1000:.2f}ms)"
        )

    return timing


# =============================================================================
# Test Decorators
# =============================================================================


def skip_if_no_module(module_name: str, reason: str | None = None) -> Callable[[F], F]:
    """Decorator to skip test if module is not available.

    Args:
        module_name: Name of required module
        reason: Custom skip reason

    Returns:
        Decorator function
    """
    import pytest

    def decorator(func: F) -> F:
        try:
            __import__(module_name)
            return func
        except ImportError:
            skip_reason = reason or f"Module '{module_name}' not available"
            return pytest.mark.skip(reason=skip_reason)(func)  # type: ignore

    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay_seconds: float = 0.1,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator to retry test on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay_seconds: Delay between attempts
        exceptions: Exceptions to catch and retry

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay_seconds)
            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def parametrize_with_edge_cases(
    param_name: str,
    base_values: list[Any],
    data_type: str = "string",
) -> Callable[[F], F]:
    """Decorator to parametrize test with base values plus edge cases.

    Args:
        param_name: Name of the parameter
        base_values: Base test values
        data_type: Type of edge cases to add

    Returns:
        Decorator function
    """
    import pytest

    all_values = base_values + generate_edge_case_data(data_type)

    def decorator(func: F) -> F:
        return pytest.mark.parametrize(param_name, all_values)(func)  # type: ignore

    return decorator


# =============================================================================
# Test Base Classes
# =============================================================================


class BaseTestCase:
    """Base class for test cases with common utilities."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up class-level fixtures."""

    @classmethod
    def teardown_class(cls) -> None:
        """Tear down class-level fixtures."""

    def setup_method(self) -> None:
        """Set up method-level fixtures."""

    def teardown_method(self) -> None:
        """Tear down method-level fixtures."""

    @staticmethod
    def create_temp_file(content: str = "", suffix: str = ".txt") -> Path:
        """Create a temporary file.

        Args:
            content: File content
            suffix: File suffix

        Returns:
            Path to temporary file
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            return Path(f.name)

    @staticmethod
    def create_temp_dir() -> Path:
        """Create a temporary directory.

        Returns:
            Path to temporary directory
        """
        return Path(tempfile.mkdtemp())


class IntegrationTestCase(BaseTestCase):
    """Base class for integration tests with extended setup."""

    # Override these in subclasses
    required_services: list[str] = []
    required_env_vars: list[str] = []

    @classmethod
    def setup_class(cls) -> None:
        """Set up integration test environment."""
        super().setup_class()
        cls._check_requirements()

    @classmethod
    def _check_requirements(cls) -> None:
        """Check that required services and env vars are available."""
        import pytest

        # Check environment variables
        missing_vars = [var for var in cls.required_env_vars if var not in os.environ]
        if missing_vars:
            pytest.skip(f"Missing environment variables: {missing_vars}")

        # Check services (override check_service in subclass)
        for service in cls.required_services:
            if not cls.check_service(service):
                pytest.skip(f"Service not available: {service}")

    @classmethod
    def check_service(cls, service_name: str) -> bool:
        """Check if a service is available.

        Override in subclass for specific service checks.

        Args:
            service_name: Name of service to check

        Returns:
            True if service is available
        """
        return True


# =============================================================================
# Fixture Factories
# =============================================================================


def fixture_factory(
    scope: str = "function",
    autouse: bool = False,
) -> Callable[[Callable[..., T]], Any]:
    """Factory for creating pytest fixtures with common configuration.

    Args:
        scope: Fixture scope
        autouse: Whether to auto-use

    Returns:
        Decorator function
    """
    import pytest

    def decorator(func: Callable[..., T]) -> Any:
        # Note: pytest.fixture returns a FixtureFunctionMarker, not just the callable
        return pytest.fixture(scope=scope, autouse=autouse)(func)  # type: ignore[call-overload]

    return decorator


# =============================================================================
# Test Markers (for use with pytest)
# =============================================================================


@dataclass
class PytestMarkers:
    """Collection of common test markers."""

    # Standard markers
    UNIT: str = "unit"
    INTEGRATION: str = "integration"
    E2E: str = "e2e"

    # Performance markers
    SLOW: str = "slow"
    PERFORMANCE: str = "performance"

    # Dependency markers
    REQUIRES_NETWORK: str = "requires_network"
    REQUIRES_DATABASE: str = "requires_database"
    REQUIRES_GPU: str = "requires_gpu"

    # Platform markers
    LINUX_ONLY: str = "linux_only"
    WINDOWS_ONLY: str = "windows_only"
    MACOS_ONLY: str = "macos_only"


# Alias for backwards compatibility
TestMarkers = PytestMarkers
markers = PytestMarkers()
