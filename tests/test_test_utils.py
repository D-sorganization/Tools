"""Unit tests for test_utils module."""

import logging
import os
import sys
from pathlib import Path

import pytest

# Path setup handled by conftest.py
from utils.test_utils import (
    BaseTestCase,
    MockFactory,
    TimingResult,
    assert_helpers,
    assert_performance,
    captured_logs,
    captured_output,
    environment_variables,
    generate_edge_case_data,
    generate_sample_data,
    markers,
    retry_on_failure,
    temporary_directory,
    temporary_file,
    time_function,
)


class TestDataGenerators:
    """Tests for data generation functions."""

    def test_generate_sample_data_int(self) -> None:
        """Test generating integer sample data."""
        data = generate_sample_data("int", size=10, seed=42)
        assert len(data) == 10
        assert all(isinstance(x, int) for x in data)

    def test_generate_sample_data_float(self) -> None:
        """Test generating float sample data."""
        data = generate_sample_data("float", size=10, seed=42)
        assert len(data) == 10
        assert all(isinstance(x, float) for x in data)

    def test_generate_sample_data_string(self) -> None:
        """Test generating string sample data."""
        data = generate_sample_data("string", size=10, seed=42)
        assert len(data) == 10
        assert all(isinstance(x, str) for x in data)

    def test_generate_sample_data_mixed(self) -> None:
        """Test generating mixed sample data."""
        data = generate_sample_data("mixed", size=50, seed=42)
        assert len(data) == 50
        # Should contain variety of types
        types = {type(x) for x in data if x is not None}
        assert len(types) >= 2  # At least 2 different types

    def test_generate_sample_data_reproducible(self) -> None:
        """Test that same seed produces same data."""
        data1 = generate_sample_data("int", size=10, seed=42)
        data2 = generate_sample_data("int", size=10, seed=42)
        assert data1 == data2

    def test_generate_edge_case_data_int(self) -> None:
        """Test generating integer edge cases."""
        data = generate_edge_case_data("int")
        assert 0 in data
        assert 1 in data
        assert -1 in data
        assert 2**31 - 1 in data  # Max 32-bit int

    def test_generate_edge_case_data_float(self) -> None:
        """Test generating float edge cases."""
        data = generate_edge_case_data("float")
        assert 0.0 in data
        assert float("inf") in data
        assert float("-inf") in data

    def test_generate_edge_case_data_string(self) -> None:
        """Test generating string edge cases."""
        data = generate_edge_case_data("string")
        assert "" in data
        assert " " in data
        # Check for security test strings
        assert any("script" in s.lower() for s in data if isinstance(s, str))


class TestAssertionHelpers:
    """Tests for custom assertion helpers."""

    def test_assert_approx_equal_pass(self) -> None:
        """Test approximate equality passing."""
        assert_helpers.assert_approx_equal(1.0, 1.0 + 1e-10)
        assert_helpers.assert_approx_equal(100.0, 100.0001, rel_tol=1e-3)

    def test_assert_approx_equal_fail(self) -> None:
        """Test approximate equality failing."""
        with pytest.raises(AssertionError):
            assert_helpers.assert_approx_equal(1.0, 2.0)

    def test_assert_dict_subset_pass(self) -> None:
        """Test dict subset assertion passing."""
        subset = {"a": 1, "b": 2}
        superset = {"a": 1, "b": 2, "c": 3}
        assert_helpers.assert_dict_subset(subset, superset)

    def test_assert_dict_subset_fail_missing_key(self) -> None:
        """Test dict subset fails when key missing."""
        subset = {"a": 1, "d": 4}
        superset = {"a": 1, "b": 2, "c": 3}
        with pytest.raises(AssertionError, match="Key 'd' not found"):
            assert_helpers.assert_dict_subset(subset, superset)

    def test_assert_dict_subset_fail_value_mismatch(self) -> None:
        """Test dict subset fails when value different."""
        subset = {"a": 999}
        superset = {"a": 1, "b": 2}
        with pytest.raises(AssertionError, match="Value mismatch"):
            assert_helpers.assert_dict_subset(subset, superset)

    def test_assert_json_equal_dict(self) -> None:
        """Test JSON equality with dicts."""
        d1 = {"a": 1, "b": [1, 2, 3]}
        d2 = {"a": 1, "b": [1, 2, 3]}
        assert_helpers.assert_json_equal(d1, d2)

    def test_assert_json_equal_string(self) -> None:
        """Test JSON equality with strings."""
        s1 = '{"a": 1}'
        s2 = '{"a":1}'  # Different formatting
        assert_helpers.assert_json_equal(s1, s2)

    def test_assert_json_equal_fail(self) -> None:
        """Test JSON inequality."""
        d1 = {"a": 1}
        d2 = {"a": 2}
        with pytest.raises(AssertionError):
            assert_helpers.assert_json_equal(d1, d2)

    def test_assert_raises_with_message(self) -> None:
        """Test exception with message pattern."""

        def raise_error() -> None:
            raise ValueError("test error 123")

        exc = assert_helpers.assert_raises_with_message(ValueError, r"test error \d+", raise_error)
        assert "123" in str(exc)

    def test_assert_raises_with_message_wrong_exception(self) -> None:
        """Test fails with wrong exception type."""

        def raise_error() -> None:
            raise TypeError("wrong type")

        with pytest.raises(AssertionError, match="Expected ValueError"):
            assert_helpers.assert_raises_with_message(ValueError, ".*", raise_error)

    def test_assert_raises_with_message_no_exception(self) -> None:
        """Test fails when no exception raised."""

        def no_error() -> None:
            pass

        with pytest.raises(AssertionError, match="was not raised"):
            assert_helpers.assert_raises_with_message(ValueError, ".*", no_error)

    def test_assert_file_contains(self, tmp_path: Path) -> None:
        """Test file contains assertion."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\nfoo bar")

        assert_helpers.assert_file_contains(test_file, "hello")
        assert_helpers.assert_file_contains(test_file, "foo bar")

    def test_assert_file_contains_fail(self, tmp_path: Path) -> None:
        """Test file contains fails when content missing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        with pytest.raises(AssertionError):
            assert_helpers.assert_file_contains(test_file, "goodbye")

    def test_assert_file_contains_nonexistent(self) -> None:
        """Test file contains fails for nonexistent file."""
        with pytest.raises(AssertionError, match="does not exist"):
            assert_helpers.assert_file_contains("/nonexistent/file.txt", "test")


class TestMockFactory:
    """Tests for MockFactory."""

    def test_create_mock_file(self) -> None:
        """Test creating mock file."""
        mock_file = MockFactory.create_mock_file(content="test content", name="test.txt")
        assert mock_file.name == "test.txt"
        assert mock_file.read() == "test content"

    def test_create_mock_response_success(self) -> None:
        """Test creating successful mock response."""
        mock_resp = MockFactory.create_mock_response(status_code=200, json_data={"key": "value"})
        assert mock_resp.status_code == 200
        assert mock_resp.json() == {"key": "value"}
        assert mock_resp.ok is True

    def test_create_mock_response_error(self) -> None:
        """Test creating error mock response."""
        mock_resp = MockFactory.create_mock_response(status_code=404)
        assert mock_resp.status_code == 404
        assert mock_resp.ok is False

    def test_create_mock_logger(self) -> None:
        """Test creating mock logger."""
        mock_logger = MockFactory.create_mock_logger()
        assert mock_logger.name == "mock_logger"
        mock_logger.info("test")  # Should not raise

    def test_create_mock_path(self) -> None:
        """Test creating mock path."""
        mock_path = MockFactory.create_mock_path(exists=True, content="file content")
        assert mock_path.exists() is True
        assert mock_path.read_text() == "file content"


class TestContextManagers:
    """Tests for context managers."""

    def test_temporary_directory(self) -> None:
        """Test temporary directory creation and cleanup."""
        with temporary_directory() as tmpdir:
            assert tmpdir.exists()
            assert tmpdir.is_dir()
            # Create a file inside
            test_file = tmpdir / "test.txt"
            test_file.write_text("test")
            assert test_file.exists()

        # Directory should be cleaned up
        assert not tmpdir.exists()

    def test_temporary_file(self) -> None:
        """Test temporary file creation and cleanup."""
        with temporary_file(content="test content", suffix=".txt") as tmp:
            assert tmp.exists()
            assert tmp.suffix == ".txt"
            assert tmp.read_text() == "test content"

        # File should be cleaned up
        assert not tmp.exists()

    def test_captured_output(self) -> None:
        """Test capturing stdout/stderr."""
        with captured_output() as (stdout, stderr):
            print("stdout message")
            print("stderr message", file=sys.stderr)

        assert "stdout message" in stdout.getvalue()
        assert "stderr message" in stderr.getvalue()

    def test_captured_logs(self) -> None:
        """Test capturing log records."""
        test_logger = logging.getLogger("test_logger")

        with captured_logs("test_logger") as logs:
            test_logger.info("test info message")
            test_logger.warning("test warning message")

        assert len(logs) == 2
        assert logs[0].levelno == logging.INFO
        assert "test info message" in logs[0].getMessage()

    def test_environment_variables(self) -> None:
        """Test temporary environment variable setting."""
        original = os.environ.get("TEST_VAR")

        with environment_variables(TEST_VAR="test_value"):
            assert os.environ["TEST_VAR"] == "test_value"

        # Should be restored
        assert os.environ.get("TEST_VAR") == original


class TestTimingUtilities:
    """Tests for timing utilities."""

    def test_time_function(self) -> None:
        """Test timing a function."""
        import time

        def slow_func(n: int) -> int:
            time.sleep(0.01)
            return n * 2

        result, timing = time_function(slow_func, 5)

        assert result == 10
        assert timing.function_name == "slow_func"
        assert timing.elapsed_seconds >= 0.01
        assert timing.elapsed_ms >= 10

    def test_time_function_iterations(self) -> None:
        """Test timing with multiple iterations."""

        def quick_func() -> int:
            return 42

        result, timing = time_function(quick_func, iterations=10)

        assert result == 42
        assert timing.iterations == 10
        assert timing.average_seconds == timing.elapsed_seconds / 10

    def test_assert_performance_pass(self) -> None:
        """Test performance assertion passing."""

        def fast_func() -> int:
            return 42

        timing = assert_performance(fast_func, max_seconds=1.0)
        assert timing.elapsed_seconds < 1.0

    def test_assert_performance_fail(self) -> None:
        """Test performance assertion failing."""
        import time

        def slow_func() -> int:
            time.sleep(0.1)
            return 42

        with pytest.raises(AssertionError, match="took"):
            assert_performance(slow_func, max_seconds=0.001)


class TestDecorators:
    """Tests for test decorators."""

    def test_retry_on_failure_passes(self) -> None:
        """Test retry decorator when function passes."""
        call_count = 0

        @retry_on_failure(max_attempts=3)
        def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure_retries(self) -> None:
        """Test retry decorator actually retries."""
        call_count = 0

        @retry_on_failure(max_attempts=3, delay_seconds=0.01)
        def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_on_failure_exhausted(self) -> None:
        """Test retry decorator gives up after max attempts."""
        call_count = 0

        @retry_on_failure(max_attempts=3, delay_seconds=0.01)
        def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fail()

        assert call_count == 3


class TestBaseTestCase:
    """Tests for BaseTestCase."""

    def test_create_temp_file(self) -> None:
        """Test temporary file creation."""
        temp_file = BaseTestCase.create_temp_file(content="test content", suffix=".txt")
        try:
            assert temp_file.exists()
            assert temp_file.read_text() == "test content"
            assert temp_file.suffix == ".txt"
        finally:
            temp_file.unlink()

    def test_create_temp_dir(self) -> None:
        """Test temporary directory creation."""
        import shutil

        temp_dir = BaseTestCase.create_temp_dir()
        try:
            assert temp_dir.exists()
            assert temp_dir.is_dir()
        finally:
            shutil.rmtree(temp_dir)


class TestTestMarkers:
    """Tests for TestMarkers dataclass."""

    def test_markers_values(self) -> None:
        """Test marker string values."""
        assert markers.UNIT == "unit"
        assert markers.INTEGRATION == "integration"
        assert markers.E2E == "e2e"
        assert markers.SLOW == "slow"
        assert markers.PERFORMANCE == "performance"
        assert markers.REQUIRES_NETWORK == "requires_network"


class TestTimingResult:
    """Tests for TimingResult dataclass."""

    def test_elapsed_ms(self) -> None:
        """Test elapsed_ms property."""
        result = TimingResult(
            elapsed_seconds=1.5,
            function_name="test",
            iterations=1,
        )
        assert result.elapsed_ms == 1500.0

    def test_average_seconds(self) -> None:
        """Test average_seconds property."""
        result = TimingResult(
            elapsed_seconds=3.0,
            function_name="test",
            iterations=3,
        )
        assert result.average_seconds == 1.0

    def test_average_ms(self) -> None:
        """Test average_ms property."""
        result = TimingResult(
            elapsed_seconds=3.0,
            function_name="test",
            iterations=3,
        )
        assert result.average_ms == 1000.0
