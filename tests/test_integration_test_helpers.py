"""Unit tests for integration_test_helpers module."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

# Path setup handled by conftest.py
from utils.integration_test_helpers import (
    APITestBase,
    IntegrationTestBase,
    MockServer,
    ResourceManager,
    ServiceStatus,
    TestDataLoader,
    TestEnvironment,
    async_event_loop,
    async_test,
    check_command_available,
    check_database_connection,
    check_http_service,
    check_port_available,
    compare_dicts_deep,
    retry_test,
    wait_for_condition,
)


class TestServiceStatus:
    """Tests for ServiceStatus dataclass."""

    def test_service_status_available(self) -> None:
        """Test creating available service status."""
        status = ServiceStatus(
            name="test_service",
            available=True,
            latency_ms=50.0,
        )
        assert status.name == "test_service"
        assert status.available is True
        assert status.latency_ms == 50.0
        assert status.error is None

    def test_service_status_unavailable(self) -> None:
        """Test creating unavailable service status."""
        status = ServiceStatus(
            name="test_service",
            available=False,
            error="Connection refused",
        )
        assert status.available is False
        assert status.error == "Connection refused"


class TestServiceChecks:
    """Tests for service availability checks."""

    def test_check_port_available_localhost(self) -> None:
        """Test checking port availability."""
        # Use a port that's unlikely to be in use but is within valid range
        result = check_port_available("localhost", 59999, timeout=0.1)
        assert isinstance(result, bool)

    def test_check_http_service_invalid(self) -> None:
        """Test checking invalid HTTP service."""
        status = check_http_service("http://localhost:59999", timeout=0.1)
        assert status.available is False

    def test_check_command_available_python(self) -> None:
        """Test checking python command availability."""
        status = check_command_available("python")
        # Python should be available since we're running in it
        assert isinstance(status, ServiceStatus)

    def test_check_command_unavailable(self) -> None:
        """Test checking unavailable command."""
        status = check_command_available("nonexistent_command_12345")
        assert status.available is False
        assert "not found" in status.error.lower()


class TestTestEnvironment:
    """Tests for TestEnvironment class."""

    def test_set_env(self) -> None:
        """Test setting environment variable."""
        env = TestEnvironment()
        original = os.environ.get("TEST_ENV_VAR")

        try:
            env.set_env("TEST_ENV_VAR", "test_value")
            assert os.environ["TEST_ENV_VAR"] == "test_value"

            env.cleanup()
            assert os.environ.get("TEST_ENV_VAR") == original
        finally:
            if original is None:
                os.environ.pop("TEST_ENV_VAR", None)
            else:
                os.environ["TEST_ENV_VAR"] = original

    def test_unset_env(self) -> None:
        """Test unsetting environment variable."""
        env = TestEnvironment()
        os.environ["TEST_UNSET_VAR"] = "original"

        try:
            env.unset_env("TEST_UNSET_VAR")
            assert "TEST_UNSET_VAR" not in os.environ

            env.cleanup()
            assert os.environ.get("TEST_UNSET_VAR") == "original"
        finally:
            os.environ.pop("TEST_UNSET_VAR", None)

    def test_create_temp_dir(self) -> None:
        """Test creating temporary directory."""
        env = TestEnvironment()

        temp_dir = env.create_temp_dir(prefix="test_")
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        env.cleanup()
        assert not temp_dir.exists()

    def test_add_cleanup_callback(self) -> None:
        """Test adding cleanup callback."""
        env = TestEnvironment()
        callback_called = False

        def cleanup_func() -> None:
            nonlocal callback_called
            callback_called = True

        env.add_cleanup(cleanup_func)
        env.cleanup()

        assert callback_called

    def test_context_manager(self) -> None:
        """Test using as context manager."""
        with TestEnvironment() as env:
            env.set_env("TEST_CTX_VAR", "value")
            assert os.environ["TEST_CTX_VAR"] == "value"

        # Should be cleaned up
        assert "TEST_CTX_VAR" not in os.environ


class TestResourceManager:
    """Tests for ResourceManager class."""

    def test_register_resource(self) -> None:
        """Test registering resource for cleanup."""
        manager = ResourceManager()
        cleanup_called = False
        resource = {"data": "value"}

        def cleanup(r: dict[str, str]) -> None:
            nonlocal cleanup_called
            cleanup_called = True
            assert r == resource

        result = manager.register(resource, cleanup)
        assert result == resource

        manager.cleanup_all()
        assert cleanup_called

    def test_enter_context(self) -> None:
        """Test entering context manager."""
        manager = ResourceManager()

        class DummyContext:
            entered = False
            exited = False

            def __enter__(self) -> str:
                self.entered = True
                return "value"

            def __exit__(self, *args: Any) -> None:
                self.exited = True

        ctx = DummyContext()
        result = manager.enter_context(ctx)

        assert result == "value"
        assert ctx.entered

        manager.cleanup_all()
        assert ctx.exited

    def test_context_manager_usage(self) -> None:
        """Test using ResourceManager as context manager."""
        cleanup_called = False

        def cleanup(r: str) -> None:
            nonlocal cleanup_called
            cleanup_called = True

        with ResourceManager() as manager:
            manager.register("resource", cleanup)

        assert cleanup_called


class TestMockServer:
    """Tests for MockServer class."""

    def test_mock_server_start_stop(self) -> None:
        """Test starting and stopping mock server."""
        server = MockServer(host="localhost", port=0)
        server.add_response("/test", status_code=200, body={"key": "value"})

        server.start()
        try:
            assert server.port > 0
            assert server.url.startswith("http://")
        finally:
            server.stop()

    def test_mock_server_context_manager(self) -> None:
        """Test mock server as context manager."""
        with MockServer(host="localhost", port=0) as server:
            server.add_response("/api/data", status_code=200, body={"status": "ok"})
            assert server.port > 0

    def test_mock_server_response(self) -> None:
        """Test mock server returns configured response."""
        import urllib.request

        with MockServer(host="localhost", port=0) as server:
            server.add_response("/test", status_code=200, body={"result": 42})

            # Make request
            url = f"{server.url}/test"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())
                assert data == {"result": 42}


class TestAsyncUtilities:
    """Tests for async testing utilities."""

    def test_async_test_decorator(self) -> None:
        """Test async_test decorator."""

        @async_test
        async def async_func() -> int:
            await asyncio.sleep(0.01)
            return 42

        result = async_func()
        assert result == 42

    def test_async_event_loop_context(self) -> None:
        """Test async_event_loop context manager."""
        with async_event_loop() as loop:

            async def coro() -> int:
                return 42

            result = loop.run_until_complete(coro())
            assert result == 42

    @pytest.mark.asyncio
    async def test_wait_for_condition_success(self) -> None:
        """Test wait_for_condition succeeds."""
        counter = 0

        def check() -> bool:
            nonlocal counter
            counter += 1
            return counter >= 3

        result = await wait_for_condition(check, timeout=1.0, interval=0.01)
        assert result is True
        assert counter >= 3

    @pytest.mark.asyncio
    async def test_wait_for_condition_timeout(self) -> None:
        """Test wait_for_condition times out."""

        def never_true() -> bool:
            return False

        with pytest.raises(TimeoutError, match="Timed out"):
            await wait_for_condition(
                never_true,
                timeout=0.1,
                interval=0.01,
                description="test condition",
            )


class TestTestDataLoader:
    """Tests for TestDataLoader class."""

    def test_load_json(self, tmp_path: Path) -> None:
        """Test loading JSON test data."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        json_file = data_dir / "test.json"
        json_file.write_text('{"key": "value", "number": 42}')

        loader = TestDataLoader(data_dir)
        data = loader.load_json("test.json")

        assert data["key"] == "value"
        assert data["number"] == 42

    def test_load_text(self, tmp_path: Path) -> None:
        """Test loading text test data."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        text_file = data_dir / "test.txt"
        text_file.write_text("Hello World")

        loader = TestDataLoader(data_dir)
        content = loader.load_text("test.txt")

        assert content == "Hello World"

    def test_load_binary(self, tmp_path: Path) -> None:
        """Test loading binary test data."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        bin_file = data_dir / "test.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03")

        loader = TestDataLoader(data_dir)
        content = loader.load_binary("test.bin")

        assert content == b"\x00\x01\x02\x03"

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading nonexistent file raises error."""
        loader = TestDataLoader(tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.load_json("nonexistent.json")


class TestRetryTest:
    """Tests for retry_test decorator."""

    def test_retry_test_succeeds(self) -> None:
        """Test retry_test when function succeeds."""
        call_count = 0

        @retry_test(max_attempts=3)
        def succeeds() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retry_test_retries_on_failure(self) -> None:
        """Test retry_test retries on failure."""
        call_count = 0

        @retry_test(max_attempts=3, delay_seconds=0.01)
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = flaky()
        assert result == "success"
        assert call_count == 3

    def test_retry_test_exhausted(self) -> None:
        """Test retry_test raises after max attempts."""
        call_count = 0

        @retry_test(max_attempts=3, delay_seconds=0.01)
        def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count == 3


class TestCompareDictsDeep:
    """Tests for compare_dicts_deep function."""

    def test_identical_dicts(self) -> None:
        """Test comparing identical dicts."""
        d1 = {"a": 1, "b": {"c": 2}}
        d2 = {"a": 1, "b": {"c": 2}}

        diffs = compare_dicts_deep(d1, d2)
        assert len(diffs) == 0

    def test_different_values(self) -> None:
        """Test comparing dicts with different values."""
        d1 = {"a": 1}
        d2 = {"a": 2}

        diffs = compare_dicts_deep(d1, d2)
        assert len(diffs) == 1
        assert "a" in diffs[0]

    def test_missing_keys(self) -> None:
        """Test comparing dicts with missing keys."""
        d1 = {"a": 1}
        d2 = {"a": 1, "b": 2}

        diffs = compare_dicts_deep(d1, d2)
        assert len(diffs) == 1
        assert "Missing" in diffs[0]

    def test_nested_differences(self) -> None:
        """Test comparing nested dicts."""
        d1 = {"a": {"b": {"c": 1}}}
        d2 = {"a": {"b": {"c": 2}}}

        diffs = compare_dicts_deep(d1, d2)
        assert len(diffs) == 1
        assert "a.b.c" in diffs[0]

    def test_ignore_keys(self) -> None:
        """Test ignoring specific keys."""
        d1 = {"a": 1, "timestamp": 12345}
        d2 = {"a": 1, "timestamp": 67890}

        diffs = compare_dicts_deep(d1, d2, ignore_keys=["timestamp"])
        assert len(diffs) == 0


class TestIntegrationTestBase:
    """Tests for IntegrationTestBase class."""

    def test_check_service_http(self) -> None:
        """Test checking HTTP service."""
        status = IntegrationTestBase.check_service("http://localhost:99999")
        assert isinstance(status, ServiceStatus)

    def test_check_service_port(self) -> None:
        """Test checking port service."""
        status = IntegrationTestBase.check_service("localhost:59999")
        assert isinstance(status, ServiceStatus)

    def test_check_service_default(self) -> None:
        """Test default service check."""
        status = IntegrationTestBase.check_service("some_service")
        assert status.available is True  # Default returns True


class TestDatabaseTestBase:
    """Tests for DatabaseTestBase class."""

    def test_check_sqlite_connection(self, tmp_path: Path) -> None:
        """Test checking SQLite connection."""
        db_path = tmp_path / "test.db"
        status = check_database_connection(str(db_path), "sqlite")
        assert status.available is True

    def test_check_unsupported_database(self) -> None:
        """Test checking unsupported database type."""
        status = check_database_connection("connection_string", "unknown_db")
        assert status.available is False
        assert "Unsupported" in status.error


class TestAPITestBase:
    """Tests for APITestBase class."""

    def test_get_headers(self) -> None:
        """Test getting headers."""

        class TestAPI(APITestBase):
            default_headers = {"Content-Type": "application/json"}
            auth_token = "test_token"

        api = TestAPI()
        headers = api.get_headers({"X-Custom": "value"})

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test_token"
        assert headers["X-Custom"] == "value"

    def test_get_headers_no_auth(self) -> None:
        """Test getting headers without auth."""

        class TestAPI(APITestBase):
            default_headers = {"Content-Type": "application/json"}
            auth_token = None

        api = TestAPI()
        headers = api.get_headers()

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers
