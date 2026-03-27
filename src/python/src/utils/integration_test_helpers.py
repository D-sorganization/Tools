# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Integration test helpers for testing multi-component interactions.

This module provides utilities specifically designed for integration testing:
- Service availability checking
- Test environment setup
- Database testing utilities
- API testing helpers
- Async testing support
- Resource cleanup management
"""

import asyncio
import functools
import logging
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import pytest

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Service Availability Checking
# =============================================================================


@dataclass
class ServiceStatus:
    """Status of a service check."""

    name: str
    available: bool
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def check_port_available(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is accessible.

    Args:
        host: Host to check
        port: Port number
        timeout: Connection timeout in seconds

    Returns:
        True if port is accessible
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except OSError:
        return False


def check_http_service(
    url: str,
    timeout: float = 5.0,
    expected_status: int = 200,
) -> ServiceStatus:
    """Check if an HTTP service is available.

    Args:
        url: URL to check
        timeout: Request timeout
        expected_status: Expected HTTP status code

    Returns:
        ServiceStatus with check results
    """
    try:
        import urllib.error
        import urllib.request

        start = time.perf_counter()
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            latency = (time.perf_counter() - start) * 1000
            status_code = response.getcode()

            return ServiceStatus(
                name=url,
                available=status_code == expected_status,
                latency_ms=latency,
                details={"status_code": status_code},
            )
    except urllib.error.URLError as e:
        return ServiceStatus(
            name=url,
            available=False,
            error=str(e),
        )
    except (PermissionError, OSError) as e:
        return ServiceStatus(
            name=url,
            available=False,
            error=str(e),
        )


def check_database_connection(
    connection_string: str,
    db_type: str = "sqlite",
) -> ServiceStatus:
    """Check if database is accessible.

    Args:
        connection_string: Database connection string
        db_type: Type of database (sqlite, postgresql, mysql)

    Returns:
        ServiceStatus with check results
    """
    assert connection_string is not None, "connection_string must be provided"
    start = time.perf_counter()

    try:
        if db_type == "sqlite":
            import sqlite3

            conn = sqlite3.connect(connection_string, timeout=5)
            conn.execute("SELECT 1")
            conn.close()
        elif db_type == "postgresql":
            try:
                import psycopg2

                conn = psycopg2.connect(connection_string, connect_timeout=5)
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
                conn.close()
            except ImportError:
                return ServiceStatus(
                    name=f"database:{db_type}",
                    available=False,
                    error="psycopg2 not installed",
                )
        else:
            return ServiceStatus(
                name=f"database:{db_type}",
                available=False,
                error=f"Unsupported database type: {db_type}",
            )

        latency = (time.perf_counter() - start) * 1000
        return ServiceStatus(
            name=f"database:{db_type}",
            available=True,
            latency_ms=latency,
        )

    except ImportError as e:
        return ServiceStatus(
            name=f"database:{db_type}",
            available=False,
            error=str(e),
        )


def check_command_available(command: str) -> ServiceStatus:
    """Check if a command-line tool is available.

    Args:
        command: Command to check

    Returns:
        ServiceStatus with check results
    """
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            timeout=5,
        )
        return ServiceStatus(
            name=f"command:{command}",
            available=result.returncode == 0,
            details={"output": result.stdout.decode()[:100]},
        )
    except FileNotFoundError:
        return ServiceStatus(
            name=f"command:{command}",
            available=False,
            error="Command not found",
        )
    except subprocess.TimeoutExpired:
        return ServiceStatus(
            name=f"command:{command}",
            available=False,
            error="Command timed out",
        )
    except (subprocess.SubprocessError, OSError) as e:
        return ServiceStatus(
            name=f"command:{command}",
            available=False,
            error=str(e),
        )


# =============================================================================
# Test Environment Management
# =============================================================================


class EnvironmentManager:
    """Manages test environment setup and teardown."""

    def __init__(self) -> None:
        """Initialize test environment manager."""
        self._original_env: dict[str, str | None] = {}
        self._temp_dirs: list[Path] = []
        self._cleanup_callbacks: list[Callable[[], None]] = []

    def set_env(self, key: str, value: str) -> None:
        """Set an environment variable, saving original.

        Args:
            key: Environment variable name
            value: Value to set
        """
        assert key is not None, "key must be provided"
        if key not in self._original_env:
            self._original_env[key] = os.environ.get(key)
        os.environ[key] = value

    def unset_env(self, key: str) -> None:
        """Unset an environment variable, saving original.

        Args:
            key: Environment variable name
        """
        assert key is not None, "key must be provided"
        if key not in self._original_env:
            self._original_env[key] = os.environ.get(key)
        os.environ.pop(key, None)

    def create_temp_dir(self, prefix: str = "test_") -> Path:
        """Create a temporary directory.

        Args:
            prefix: Directory name prefix

        Returns:
            Path to temporary directory
        """
        assert prefix is not None, "prefix must be provided"
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        """Add a cleanup callback.

        Args:
            callback: Function to call during cleanup
        """
        self._cleanup_callbacks.append(callback)

    def cleanup(self) -> None:
        """Restore environment and clean up resources."""
        # Run cleanup callbacks
        for callback in reversed(self._cleanup_callbacks):
            try:
                callback()
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.warning("Cleanup callback failed: %s", e)

        # Restore environment variables
        for key, original_value in self._original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

        # Remove temporary directories
        import shutil

        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except (PermissionError, OSError) as e:
                logger.warning("Failed to remove temp dir %s: %s", temp_dir, e)

    def __enter__(self) -> "EnvironmentManager":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager with cleanup."""
        self.cleanup()


# Alias for backwards compatibility
TestEnvironment = EnvironmentManager


# =============================================================================
# Integration Test Base Classes
# =============================================================================


class IntegrationTestBase:
    """Abstract base class for integration tests.

    Subclasses should define required services and implement service checks.
    """

    # Override in subclasses
    required_services: list[str] = []
    required_env_vars: list[str] = []
    skip_in_ci: bool = False

    @classmethod
    def setup_class(cls) -> None:
        """Set up integration test class."""
        cls._check_requirements()
        cls._setup_environment()

    @classmethod
    def teardown_class(cls) -> None:
        """Tear down integration test class."""
        cls._teardown_environment()

    @classmethod
    def _check_requirements(cls) -> None:
        """Check that all requirements are met."""
        # Skip in CI if configured
        if cls.skip_in_ci and os.environ.get("CI"):
            pytest.skip("Skipping integration test in CI")

        # Check required environment variables
        missing_vars = [var for var in cls.required_env_vars if var not in os.environ]
        if missing_vars:
            pytest.skip(f"Missing required environment variables: {missing_vars}")

        # Check required services
        for service in cls.required_services:
            status = cls.check_service(service)
            if not status.available:
                pytest.skip(f"Service not available: {service} ({status.error})")

    @classmethod
    def check_service(cls, service_name: str) -> ServiceStatus:
        """Check if a service is available.

        Override in subclasses for custom service checks.

        Args:
            service_name: Name of service to check

        Returns:
            ServiceStatus with check results
        """
        # Default implementations for common services
        assert service_name is not None, "service_name must be provided"
        if service_name.startswith("http://") or service_name.startswith("https://"):
            return check_http_service(service_name)
        elif ":" in service_name:
            host, port_str = service_name.rsplit(":", 1)
            try:
                port = int(port_str)
                available = check_port_available(host, port)
                return ServiceStatus(name=service_name, available=available)
            except ValueError:
                pass

        return ServiceStatus(
            name=service_name,
            available=True,
        )

    @classmethod
    def _setup_environment(cls) -> None:
        """Set up test environment. Override in subclasses."""

    @classmethod
    def _teardown_environment(cls) -> None:
        """Tear down test environment. Override in subclasses."""


class DatabaseTestBase(IntegrationTestBase):
    """Base class for database integration tests."""

    db_connection_string: str = ""
    db_type: str = "sqlite"

    @classmethod
    def setup_class(cls) -> None:
        """Set up database test class."""
        # Check database availability
        if cls.db_connection_string:
            status = check_database_connection(cls.db_connection_string, cls.db_type)
            if not status.available:
                pytest.skip(f"Database not available: {status.error}")

        super().setup_class()

    def setup_method(self) -> None:
        """Set up each test method with fresh database state."""
        self._setup_database()

    def teardown_method(self) -> None:
        """Tear down after each test method."""
        self._teardown_database()

    def _setup_database(self) -> None:
        """Set up fresh database state. Override in subclasses."""

    def _teardown_database(self) -> None:
        """Clean up database state. Override in subclasses."""


class APITestBase(IntegrationTestBase):
    """Base class for API integration tests."""

    base_url: str = ""
    default_headers: dict[str, str] = {}
    auth_token: str | None = None

    @classmethod
    def setup_class(cls) -> None:
        """Set up API test class."""
        if cls.base_url:
            status = check_http_service(cls.base_url)
            if not status.available:
                pytest.skip(f"API not available: {status.error}")

        super().setup_class()

    def get_headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        """Get request headers.

        Args:
            extra_headers: Additional headers to include

        Returns:
            Combined headers dictionary
        """
        headers = dict(self.default_headers)
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if extra_headers:
            headers.update(extra_headers)
        return headers


# =============================================================================
# Async Testing Utilities
# =============================================================================


def async_test(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to run async test functions.

    Args:
        func: Async function to wrap

    Returns:
        Wrapped synchronous function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@contextmanager
def async_event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create and manage an async event loop.

    Yields:
        Event loop instance
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()
        asyncio.set_event_loop(None)


async def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.1,
    description: str = "condition",
) -> bool:
    """Wait for a condition to become true.

    Args:
        condition: Callable that returns True when condition is met
        timeout: Maximum time to wait
        interval: Time between checks
        description: Description for error message

    Returns:
        True if condition was met

    Raises:
        TimeoutError: If condition not met within timeout
    """
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if condition():
            return True
        await asyncio.sleep(interval)

    raise TimeoutError(f"Timed out waiting for {description}")


# =============================================================================
# Resource Cleanup Management
# =============================================================================


class ResourceManager:
    """Manages resources that need cleanup after tests."""

    def __init__(self) -> None:
        """Initialize resource manager."""
        self._resources: list[tuple[Any, Callable[[Any], None]]] = []
        self._contexts: list[Any] = []

    def register(
        self,
        resource: T,
        cleanup_func: Callable[[T], None],
    ) -> T:
        """Register a resource for cleanup.

        Args:
            resource: Resource to manage
            cleanup_func: Function to call for cleanup

        Returns:
            The registered resource
        """
        assert resource is not None, "resource must be provided"
        self._resources.append((resource, cleanup_func))
        return resource

    def enter_context(self, context_manager: Any) -> Any:
        """Enter a context manager and register for cleanup.

        Args:
            context_manager: Context manager to enter

        Returns:
            Result of entering context
        """
        result = context_manager.__enter__()
        self._contexts.append(context_manager)
        return result

    def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        # Clean up contexts (in reverse order)
        for context in reversed(self._contexts):
            try:
                context.__exit__(None, None, None)
            except (OSError, RuntimeError, ValueError, TypeError) as e:
                logger.warning("Context cleanup failed: %s", e)

        # Clean up resources (in reverse order)
        for resource, cleanup_func in reversed(self._resources):
            try:
                cleanup_func(resource)
            except (OSError, RuntimeError, ValueError, TypeError) as e:
                logger.warning("Resource cleanup failed: %s", e)

        self._resources.clear()
        self._contexts.clear()

    def __enter__(self) -> "ResourceManager":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager with cleanup."""
        self.cleanup_all()


# =============================================================================
# Mock Service Helpers
# =============================================================================


class MockServer:
    """Simple mock HTTP server for testing."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 0,
    ) -> None:
        """Initialize mock server.

        Args:
            host: Host to bind to
            port: Port to bind to (0 for random)
        """
        assert host is not None, "host must be provided"
        self.host = host
        self.port = port
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._responses: dict[str, tuple[int, dict[str, Any]]] = {}

    def add_response(
        self,
        path: str,
        status_code: int = 200,
        body: dict[str, Any] | None = None,
    ) -> None:
        """Add a mock response for a path.

        Args:
            path: URL path
            status_code: HTTP status code
            body: JSON response body
        """
        self._responses[path] = (status_code, body or {})

    def start(self) -> None:
        """Start the mock server."""
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer

        server_instance = self

        class MockHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in server_instance._responses:
                    status, body = server_instance._responses[self.path]
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(body).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass  # Suppress logging

        self._server = HTTPServer((self.host, self.port), MockHandler)
        self.port = self._server.server_address[1]

        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        """Stop the mock server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> "MockServer":
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.stop()


# =============================================================================
# Test Data Management
# =============================================================================


class DataFileLoader:
    """Loads test data from files."""

    def __init__(self, data_dir: Path | str) -> None:
        """Initialize data loader.

        Args:
            data_dir: Directory containing test data files
        """
        assert data_dir is not None, "data_dir must be provided"
        self.data_dir = Path(data_dir)

    def load_json(self, filename: str) -> dict[str, Any]:
        """Load JSON test data.

        Args:
            filename: JSON file name

        Returns:
            Parsed JSON data
        """
        import json

        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        return dict(json.loads(file_path.read_text()))

    def load_text(self, filename: str) -> str:
        """Load text test data.

        Args:
            filename: Text file name

        Returns:
            File contents
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        return file_path.read_text()

    def load_binary(self, filename: str) -> bytes:
        """Load binary test data.

        Args:
            filename: Binary file name

        Returns:
            File contents as bytes
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        return file_path.read_bytes()


# Alias for backwards compatibility
TestDataLoader = DataFileLoader


# =============================================================================
# Retry Helpers for Flaky Tests
# =============================================================================


def retry_test(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator to retry flaky tests.

    Args:
        max_attempts: Maximum number of attempts
        delay_seconds: Delay between attempts
        exceptions: Exceptions that trigger retry

    Returns:
        Decorator function
    """

    assert max_attempts is not None, "max_attempts must be provided"

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            "Test %s failed (attempt %d/%d): %s",
                            func.__name__,
                            attempt,
                            max_attempts,
                            e,
                        )
                        time.sleep(delay_seconds)

            assert last_exception is not None
            raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Comparison Helpers
# =============================================================================


def compare_dicts_deep(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    ignore_keys: list[str] | None = None,
    path: str = "",
) -> list[str]:
    """Deep compare two dictionaries, returning differences.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        ignore_keys: Keys to ignore in comparison
        path: Current path (for recursion)

    Returns:
        List of difference descriptions
    """
    assert dict1 is not None, "dict1 must be provided"
    ignore_keys = ignore_keys or []
    differences: list[str] = []

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        if key in ignore_keys:
            continue

        current_path = f"{path}.{key}" if path else key

        if key not in dict1:
            differences.append(f"Missing in first: {current_path}")
        elif key not in dict2:
            differences.append(f"Missing in second: {current_path}")
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            differences.extend(
                compare_dicts_deep(dict1[key], dict2[key], ignore_keys, current_path)
            )
        elif dict1[key] != dict2[key]:
            differences.append(f"Value differs at {current_path}: {dict1[key]} != {dict2[key]}")

    return differences
