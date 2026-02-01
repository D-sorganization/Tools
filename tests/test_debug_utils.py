"""Unit tests for debug_utils module."""

import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python" / "src"))

from utils.debug_utils import (
    ExecutionTracer,
    MemoryStats,
    PerformanceWatchdog,
    ProfileResult,
    StackFrame,
    SystemDiagnostics,
    TimingStats,
    conditional_breakpoint,
    debug_exception,
    debug_log,
    debug_only,
    debug_vars,
    deprecated,
    format_exception_with_locals,
    get_call_stack,
    get_caller_info,
    get_memory_usage,
    get_system_diagnostics,
    get_watchdog,
    is_debug_mode,
    memory_profile,
    memory_tracker,
    profile,
    profile_block,
    set_debug_mode,
    timed,
    timer,
    trace_calls,
)


class TestDebugMode:
    """Tests for debug mode management."""

    def setup_method(self) -> None:
        """Reset debug mode before each test."""
        set_debug_mode(False)

    def teardown_method(self) -> None:
        """Reset debug mode after each test."""
        set_debug_mode(False)

    def test_is_debug_mode_default(self) -> None:
        """Test default debug mode is off."""
        set_debug_mode(False)
        assert is_debug_mode() is False

    def test_set_debug_mode_on(self) -> None:
        """Test enabling debug mode."""
        set_debug_mode(True)
        assert is_debug_mode() is True

    def test_set_debug_mode_off(self) -> None:
        """Test disabling debug mode."""
        set_debug_mode(True)
        set_debug_mode(False)
        assert is_debug_mode() is False

    def test_debug_only_runs_in_debug_mode(self) -> None:
        """Test debug_only decorator runs in debug mode."""
        call_count = 0

        @debug_only
        def tracked_func() -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        set_debug_mode(True)
        result = tracked_func()
        assert result == "result"
        assert call_count == 1

    def test_debug_only_skips_outside_debug_mode(self) -> None:
        """Test debug_only decorator skips outside debug mode."""
        call_count = 0

        @debug_only
        def tracked_func() -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        set_debug_mode(False)
        result = tracked_func()
        assert result is None
        assert call_count == 0


class TestProfilingUtilities:
    """Tests for profiling utilities."""

    def test_profile_decorator(self, tmp_path: Path) -> None:
        """Test profile decorator."""
        output_file = tmp_path / "profile.txt"

        @profile(output_file=str(output_file), top_n=5)
        def profiled_func() -> int:
            total = 0
            for i in range(100):
                total += i
            return total

        result = profiled_func()
        assert result == 4950
        assert output_file.exists()

    def test_profile_block(self) -> None:
        """Test profile_block context manager."""
        with profile_block(name="test_block", log_result=False) as result:
            _total = sum(range(100))  # noqa: F841  # Used for profiling

        assert result.function_name == "test_block"
        assert result.total_time >= 0
        assert isinstance(result.stats_text, str)

    def test_profile_result_str(self) -> None:
        """Test ProfileResult string representation."""
        result = ProfileResult(
            function_name="test_func",
            total_time=1.5,
            cumulative_time=2.0,
            call_count=10,
            stats_text="stats",
        )
        result_str = str(result)
        assert "test_func" in result_str
        assert "1.5" in result_str


class TestTimingUtilities:
    """Tests for timing utilities."""

    def test_timed_decorator(self) -> None:
        """Test timed decorator."""

        @timed(log_level=logging.DEBUG)
        def timed_func() -> int:
            time.sleep(0.01)
            return 42

        result = timed_func()
        assert result == 42

    def test_timed_decorator_with_threshold(self) -> None:
        """Test timed decorator with threshold."""

        @timed(log_level=logging.DEBUG, threshold_ms=1000)
        def fast_func() -> int:
            return 42

        # Should not log since below threshold
        result = fast_func()
        assert result == 42

    def test_timer_context_manager(self) -> None:
        """Test timer context manager."""
        with timer(name="test_operation", log_level=logging.DEBUG) as stats:
            time.sleep(0.01)

        assert stats.name == "test_operation"
        assert stats.elapsed_seconds >= 0.01
        assert stats.elapsed_ms >= 10

    def test_timing_stats_properties(self) -> None:
        """Test TimingStats properties."""
        from datetime import datetime

        stats = TimingStats(
            name="test",
            elapsed_seconds=1.5,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert stats.elapsed_ms == 1500.0
        assert "test" in str(stats)
        assert "1500" in str(stats)


class TestMemoryProfiling:
    """Tests for memory profiling utilities."""

    def test_get_memory_usage(self) -> None:
        """Test getting memory usage."""
        usage = get_memory_usage()
        assert usage >= 0

    def test_memory_tracker(self) -> None:
        """Test memory_tracker context manager."""
        with memory_tracker(name="test_memory", log_result=False) as stats:
            # Allocate some memory
            _data = [0] * 10000  # noqa: F841  # Used for memory allocation

        assert stats.current_mb >= 0
        assert stats.peak_mb >= 0
        assert isinstance(stats.top_allocations, list)

    def test_memory_profile_decorator(self) -> None:
        """Test memory_profile decorator."""

        @memory_profile
        def memory_intensive() -> list[int]:
            return [0] * 10000

        result = memory_intensive()
        assert len(result) == 10000

    def test_memory_stats_str(self) -> None:
        """Test MemoryStats string representation."""
        stats = MemoryStats(
            current_mb=100.5,
            peak_mb=150.2,
            diff_mb=50.3,
            top_allocations=[],
        )
        stats_str = str(stats)
        assert "100.5" in stats_str
        assert "150.2" in stats_str


class TestExecutionTracing:
    """Tests for execution tracing."""

    def test_execution_tracer_basic(self) -> None:
        """Test basic execution tracing."""
        tracer = ExecutionTracer(max_depth=5)

        def inner() -> int:
            return 42

        def outer() -> int:
            return inner()

        with tracer.trace() as trace_log:
            outer()

        # Should have traced the calls
        assert isinstance(trace_log, list)

    def test_trace_calls_decorator(self) -> None:
        """Test trace_calls decorator."""
        set_debug_mode(True)
        try:

            @trace_calls(max_depth=3)
            def traced_func() -> int:
                return 42

            result = traced_func()
            assert result == 42
        finally:
            set_debug_mode(False)


class TestStackInspection:
    """Tests for stack inspection utilities."""

    def test_get_call_stack(self) -> None:
        """Test getting call stack."""

        def inner() -> list[StackFrame]:
            return get_call_stack(skip_frames=1, max_frames=5)

        def outer() -> list[StackFrame]:
            return inner()

        frames = outer()
        assert len(frames) >= 1
        assert all(isinstance(f, StackFrame) for f in frames)

    def test_get_call_stack_with_locals(self) -> None:
        """Test getting call stack with local variables."""
        local_var = 42  # noqa: F841

        frames = get_call_stack(skip_frames=1, max_frames=3, include_locals=True)
        assert len(frames) >= 1

    def test_get_caller_info(self) -> None:
        """Test getting caller information."""

        def inner() -> tuple[str, str, int]:
            return get_caller_info(skip_frames=1)

        def outer() -> tuple[str, str, int]:
            return inner()

        filename, func, lineno = outer()
        assert func == "outer"
        assert lineno > 0

    def test_stack_frame_str(self) -> None:
        """Test StackFrame string representation."""
        frame = StackFrame(
            filename="test.py",
            lineno=42,
            function="test_func",
            code_context="x = 1",
            local_vars={"x": "1"},
        )
        frame_str = str(frame)
        assert "test_func" in frame_str
        assert "test.py:42" in frame_str


class TestDebugLogging:
    """Tests for debug logging helpers."""

    def test_debug_log_in_debug_mode(self) -> None:
        """Test debug_log in debug mode."""
        set_debug_mode(True)
        try:
            # Should not raise
            debug_log("Test message %s", "arg1", extra_key="value")
        finally:
            set_debug_mode(False)

    def test_debug_log_outside_debug_mode(self) -> None:
        """Test debug_log outside debug mode does nothing."""
        set_debug_mode(False)
        # Should not raise
        debug_log("Test message")

    def test_debug_vars_in_debug_mode(self) -> None:
        """Test debug_vars in debug mode."""
        set_debug_mode(True)
        try:
            x = 42
            y = "test"
            # Should not raise
            debug_vars(x=x, y=y)
        finally:
            set_debug_mode(False)


class TestExceptionDebugging:
    """Tests for exception debugging utilities."""

    def test_format_exception_with_locals(self) -> None:
        """Test formatting exception with locals."""

        def raise_error() -> None:
            local_var = 42  # noqa: F841
            raise ValueError("Test error")

        try:
            raise_error()
        except ValueError as e:
            formatted = format_exception_with_locals(e, include_locals=True)
            assert "ValueError" in formatted or "Test error" in formatted

    def test_debug_exception(self) -> None:
        """Test debug_exception function."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            # Should not raise
            debug_exception(e, log_level=logging.DEBUG)


class TestSystemDiagnostics:
    """Tests for system diagnostics."""

    def test_get_system_diagnostics(self) -> None:
        """Test getting system diagnostics."""
        diag = get_system_diagnostics()

        assert isinstance(diag, SystemDiagnostics)
        assert len(diag.python_version) > 0
        assert len(diag.platform) > 0
        assert diag.process_id > 0
        assert diag.thread_count >= 1

    def test_get_system_diagnostics_with_env(self) -> None:
        """Test diagnostics with environment variables."""
        diag = get_system_diagnostics(include_env=True)
        assert isinstance(diag.environment, dict)

    def test_get_system_diagnostics_with_modules(self) -> None:
        """Test diagnostics with loaded modules."""
        diag = get_system_diagnostics(include_modules=True)
        assert isinstance(diag.loaded_modules, list)
        assert len(diag.loaded_modules) > 0

    def test_system_diagnostics_to_dict(self) -> None:
        """Test converting diagnostics to dict."""
        diag = get_system_diagnostics()
        d = diag.to_dict()

        assert "python_version" in d
        assert "platform" in d
        assert "process_id" in d


class TestPerformanceWatchdog:
    """Tests for PerformanceWatchdog."""

    def test_watchdog_record(self) -> None:
        """Test recording timings."""
        watchdog = PerformanceWatchdog(warn_threshold_ms=100, error_threshold_ms=500)
        watchdog.record("operation1", 50.0)
        watchdog.record("operation1", 75.0)

        stats = watchdog.get_stats("operation1")
        assert stats["count"] == 2
        assert stats["min"] == 50.0
        assert stats["max"] == 75.0
        assert stats["avg"] == 62.5

    def test_watchdog_watch_context(self) -> None:
        """Test watch context manager."""
        watchdog = PerformanceWatchdog()

        with watchdog.watch("test_operation"):
            time.sleep(0.01)

        stats = watchdog.get_stats("test_operation")
        assert stats["count"] == 1
        assert stats["min"] >= 10  # At least 10ms

    def test_watchdog_report(self) -> None:
        """Test generating report."""
        watchdog = PerformanceWatchdog()
        watchdog.record("op1", 100.0)
        watchdog.record("op2", 200.0)

        report = watchdog.report()
        assert "Performance Report" in report
        assert "op1" in report
        assert "op2" in report

    def test_get_global_watchdog(self) -> None:
        """Test getting global watchdog instance."""
        watchdog = get_watchdog()
        assert isinstance(watchdog, PerformanceWatchdog)


class TestDeprecatedDecorator:
    """Tests for deprecated decorator."""

    def test_deprecated_warning(self) -> None:
        """Test deprecated decorator emits warning."""

        @deprecated(reason="Use new_func instead", removal_version="2.0")
        def old_func() -> str:
            return "result"

        with pytest.warns(DeprecationWarning, match="old_func is deprecated"):
            result = old_func()

        assert result == "result"

    def test_deprecated_without_reason(self) -> None:
        """Test deprecated decorator without reason."""

        @deprecated()
        def old_func() -> str:
            return "result"

        with pytest.warns(DeprecationWarning):
            result = old_func()

        assert result == "result"


class TestConditionalBreakpoint:
    """Tests for conditional breakpoint."""

    def test_conditional_breakpoint_false(self) -> None:
        """Test conditional breakpoint with false condition."""
        # Should not break
        conditional_breakpoint(False, "Should not trigger")

    @patch("builtins.breakpoint")
    def test_conditional_breakpoint_true(self, mock_breakpoint: MagicMock) -> None:
        """Test conditional breakpoint with true condition."""
        conditional_breakpoint(True, "Should trigger")
        mock_breakpoint.assert_called_once()
