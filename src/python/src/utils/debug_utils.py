"""
Comprehensive debugging utilities for troubleshooting and profiling.

This module provides standardized debugging utilities including:
- Profiling decorators and context managers
- Memory profiling tools
- Execution tracing
- Debug mode management
- Diagnostic utilities
- Stack inspection helpers
- Performance monitoring
"""

import atexit
import cProfile
import functools
import gc
import inspect
import io
import linecache
import logging
import os
import pstats
import sys
import threading
import time
import tracemalloc
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Module-level logger
logger = logging.getLogger(__name__)

# Debug mode flag - can be set globally
_DEBUG_MODE: bool = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


# =============================================================================
# Debug Mode Management
# =============================================================================


def is_debug_mode() -> bool:
    """Check if debug mode is enabled.

    Returns:
        True if debug mode is enabled
    """
    return _DEBUG_MODE


def set_debug_mode(enabled: bool) -> None:
    """Set the debug mode flag.

    Args:
        enabled: Whether to enable debug mode
    """
    global _DEBUG_MODE
    _DEBUG_MODE = enabled
    if enabled:
        # Enable more verbose logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")


def debug_only(func: F) -> F:
    """Decorator that only executes function in debug mode.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that only runs in debug mode
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _DEBUG_MODE:
            return func(*args, **kwargs)
        return None

    return wrapper  # type: ignore[return-value]


# =============================================================================
# Profiling Utilities
# =============================================================================


@dataclass
class ProfileResult:
    """Result of profiling operation."""

    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    stats_text: str
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"Profile: {self.function_name}\n"
            f"  Total time: {self.total_time:.4f}s\n"
            f"  Cumulative time: {self.cumulative_time:.4f}s\n"
            f"  Call count: {self.call_count}\n"
        )


def profile(
    output_file: str | Path | None = None,
    sort_by: str = "cumulative",
    top_n: int = 20,
) -> Callable[[F], F]:
    """Decorator to profile a function execution.

    Args:
        output_file: Optional file to save profile stats
        sort_by: Sort key for stats (cumulative, tottime, calls, etc.)
        top_n: Number of top functions to show

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = cProfile.Profile()
            try:
                return profiler.runcall(func, *args, **kwargs)
            finally:
                # Process stats
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats(sort_by)
                stats.print_stats(top_n)

                # Log or save results
                stats_text = stream.getvalue()
                if _DEBUG_MODE:
                    logger.debug("Profile for %s:\n%s", func.__name__, stats_text)

                if output_file:
                    Path(output_file).write_text(stats_text)
                    profiler.dump_stats(str(output_file) + ".prof")

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def profile_block(
    name: str = "code_block",
    log_result: bool = True,
) -> Generator[ProfileResult, None, None]:
    """Context manager to profile a block of code.

    Args:
        name: Name for the profiled block
        log_result: Whether to log the result

    Yields:
        ProfileResult object (populated after block completes)
    """
    profiler = cProfile.Profile()
    result = ProfileResult(
        function_name=name,
        total_time=0.0,
        cumulative_time=0.0,
        call_count=0,
        stats_text="",
    )

    profiler.enable()
    start_time = time.perf_counter()
    try:
        yield result
    finally:
        profiler.disable()
        elapsed = time.perf_counter() - start_time

        # Process stats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)

        result.total_time = elapsed
        result.cumulative_time = elapsed
        result.stats_text = stream.getvalue()

        if log_result and _DEBUG_MODE:
            logger.debug("Profile for '%s':\n%s", name, result.stats_text)


# =============================================================================
# Timing Utilities
# =============================================================================


@dataclass
class TimingStats:
    """Statistics for timed execution."""

    name: str
    elapsed_seconds: float
    start_time: datetime
    end_time: datetime
    extra_info: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed_ms:.2f}ms"


def timed(
    log_level: int = logging.DEBUG,
    threshold_ms: float | None = None,
) -> Callable[[F], F]:
    """Decorator to time function execution.

    Args:
        log_level: Level to log timing at
        threshold_ms: Only log if execution exceeds this threshold

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                if threshold_ms is None or elapsed >= threshold_ms:
                    logger.log(
                        log_level,
                        "%s executed in %.2fms",
                        func.__name__,
                        elapsed,
                    )

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def timer(
    name: str = "operation",
    log_level: int = logging.DEBUG,
) -> Generator[TimingStats, None, None]:
    """Context manager for timing code blocks.

    Args:
        name: Name for the timed operation
        log_level: Level to log timing at

    Yields:
        TimingStats object (populated after block completes)
    """
    stats = TimingStats(
        name=name,
        elapsed_seconds=0.0,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats.elapsed_seconds = time.perf_counter() - start
        stats.end_time = datetime.now()
        logger.log(log_level, "%s", stats)


# =============================================================================
# Memory Profiling
# =============================================================================


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    current_mb: float
    peak_mb: float
    diff_mb: float
    top_allocations: list[tuple[str, int]]

    def __str__(self) -> str:
        return (
            f"Memory: current={self.current_mb:.2f}MB, "
            f"peak={self.peak_mb:.2f}MB, diff={self.diff_mb:+.2f}MB"
        )


def get_memory_usage() -> float:
    """Get current memory usage in MB.

    Returns:
        Memory usage in megabytes
    """
    try:
        import resource  # Unix only

        # Unix-based systems
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return float(usage.ru_maxrss / (1024 * 1024))
        return float(usage.ru_maxrss / 1024)
    except ImportError:
        # Windows fallback
        try:
            import psutil

            process = psutil.Process()
            return float(process.memory_info().rss / (1024 * 1024))
        except ImportError:
            return 0.0


@contextmanager
def memory_tracker(
    name: str = "memory_block",
    log_result: bool = True,
) -> Generator[MemoryStats, None, None]:
    """Context manager to track memory usage.

    Args:
        name: Name for the tracked block
        log_result: Whether to log the result

    Yields:
        MemoryStats object (populated after block completes)
    """
    # Force garbage collection before measuring
    gc.collect()

    tracemalloc.start()
    start_memory = get_memory_usage()

    stats = MemoryStats(
        current_mb=0.0,
        peak_mb=0.0,
        diff_mb=0.0,
        top_allocations=[],
    )

    try:
        yield stats
    finally:
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Get top allocations
        top_stats = snapshot.statistics("lineno")[:10]
        stats.top_allocations = [(str(stat.traceback), stat.size) for stat in top_stats]

        end_memory = get_memory_usage()
        stats.current_mb = current / (1024 * 1024)
        stats.peak_mb = peak / (1024 * 1024)
        stats.diff_mb = end_memory - start_memory

        if log_result and _DEBUG_MODE:
            logger.debug("Memory stats for '%s': %s", name, stats)


def memory_profile(func: F) -> F:
    """Decorator to profile memory usage of a function.

    Args:
        func: Function to profile

    Returns:
        Wrapped function with memory profiling
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with memory_tracker(func.__name__):
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


# =============================================================================
# Execution Tracing
# =============================================================================


class ExecutionTracer:
    """Traces function calls and execution flow."""

    def __init__(
        self,
        include_modules: list[str] | None = None,
        exclude_modules: list[str] | None = None,
        max_depth: int = 10,
    ):
        """Initialize the tracer.

        Args:
            include_modules: Only trace these modules (if specified)
            exclude_modules: Don't trace these modules
            max_depth: Maximum call depth to trace
        """
        self.include_modules = include_modules or []
        self.exclude_modules = exclude_modules or ["logging", "threading"]
        self.max_depth = max_depth
        self._depth = 0
        self._trace_log: list[str] = []
        self._active = False

    def _should_trace(self, frame: Any) -> bool:
        """Check if frame should be traced."""
        if not frame:
            return False

        module = frame.f_globals.get("__name__", "")

        # Check exclusions
        for exclude in self.exclude_modules:
            if module.startswith(exclude):
                return False

        # Check inclusions
        if self.include_modules:
            for include in self.include_modules:
                if module.startswith(include):
                    return True
            return False

        return True

    def _trace_calls(
        self,
        frame: Any,
        event: str,
        arg: Any,
    ) -> Callable[..., Any] | None:
        """Trace function for sys.settrace."""
        if not self._should_trace(frame):
            return None

        if event == "call":
            if self._depth < self.max_depth:
                indent = "  " * self._depth
                func_name = frame.f_code.co_name
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                self._trace_log.append(f"{indent}-> {func_name} ({filename}:{lineno})")
            self._depth += 1

        elif event == "return":
            self._depth = max(0, self._depth - 1)
            if self._depth < self.max_depth:
                indent = "  " * self._depth
                func_name = frame.f_code.co_name
                self._trace_log.append(f"{indent}<- {func_name}")

        return self._trace_calls

    def start(self) -> None:
        """Start tracing."""
        self._active = True
        self._depth = 0
        self._trace_log = []
        sys.settrace(self._trace_calls)

    def stop(self) -> list[str]:
        """Stop tracing and return trace log.

        Returns:
            List of trace entries
        """
        sys.settrace(None)
        self._active = False
        return self._trace_log

    @contextmanager
    def trace(self) -> Generator[list[str], None, None]:
        """Context manager for tracing.

        Yields:
            List of trace entries (populated after block)
        """
        self.start()
        trace_log = self._trace_log
        try:
            yield trace_log
        finally:
            self.stop()


def trace_calls(
    include_modules: list[str] | None = None,
    max_depth: int = 5,
) -> Callable[[F], F]:
    """Decorator to trace function calls.

    Args:
        include_modules: Modules to include in trace
        max_depth: Maximum trace depth

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = ExecutionTracer(
                include_modules=include_modules,
                max_depth=max_depth,
            )
            with tracer.trace() as trace_log:
                result = func(*args, **kwargs)

            if _DEBUG_MODE and trace_log:
                logger.debug(
                    "Trace for %s:\n%s",
                    func.__name__,
                    "\n".join(trace_log),
                )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Stack Inspection
# =============================================================================


@dataclass
class StackFrame:
    """Information about a stack frame."""

    filename: str
    lineno: int
    function: str
    code_context: str
    local_vars: dict[str, str]

    def __str__(self) -> str:
        return f"{self.function} at {self.filename}:{self.lineno}"


def get_call_stack(
    skip_frames: int = 1,
    max_frames: int = 10,
    include_locals: bool = False,
) -> list[StackFrame]:
    """Get the current call stack.

    Args:
        skip_frames: Number of frames to skip (1 = skip this function)
        max_frames: Maximum number of frames to return
        include_locals: Whether to include local variables

    Returns:
        List of StackFrame objects
    """
    frames: list[StackFrame] = []
    stack = inspect.stack()[skip_frames : skip_frames + max_frames]

    for frame_info in stack:
        local_vars: dict[str, str] = {}
        if include_locals:
            for key, value in frame_info.frame.f_locals.items():
                try:
                    local_vars[key] = repr(value)[:100]
                except (KeyError, ValueError, TypeError):
                    local_vars[key] = "<unprintable>"

        code_context = ""
        if frame_info.code_context:
            code_context = "".join(frame_info.code_context).strip()

        frames.append(
            StackFrame(
                filename=frame_info.filename,
                lineno=frame_info.lineno,
                function=frame_info.function,
                code_context=code_context,
                local_vars=local_vars,
            )
        )

    return frames


def print_call_stack(
    skip_frames: int = 1,
    max_frames: int = 10,
    include_locals: bool = False,
    file: Any = None,
) -> None:
    """Print the current call stack.

    Args:
        skip_frames: Number of frames to skip
        max_frames: Maximum frames to print
        include_locals: Include local variables
        file: File to print to (default: stderr)
    """
    if file is None:
        file = sys.stderr

    frames = get_call_stack(skip_frames + 1, max_frames, include_locals)

    print("\n=== Call Stack ===", file=file)  # noqa: T201
    for i, frame in enumerate(frames):
        print(f"\n[{i}] {frame}", file=file)  # noqa: T201
        if frame.code_context:
            print(f"    > {frame.code_context}", file=file)  # noqa: T201
        if frame.local_vars:
            print("    Locals:", file=file)  # noqa: T201
            for key, value in frame.local_vars.items():
                print(f"      {key} = {value}", file=file)  # noqa: T201
    print("\n==================\n", file=file)  # noqa: T201


def get_caller_info(skip_frames: int = 1) -> tuple[str, str, int]:
    """Get information about the calling function.

    Args:
        skip_frames: Number of frames to skip

    Returns:
        Tuple of (filename, function_name, line_number)
    """
    frame = inspect.currentframe()
    for _ in range(skip_frames + 1):
        if frame is not None:
            frame = frame.f_back

    if frame is None:
        return ("<unknown>", "<unknown>", 0)

    return (
        frame.f_code.co_filename,
        frame.f_code.co_name,
        frame.f_lineno,
    )


# =============================================================================
# Debug Logging Helpers
# =============================================================================


def debug_log(
    message: str,
    *args: Any,
    include_caller: bool = True,
    **kwargs: Any,
) -> None:
    """Log a debug message with optional caller information.

    Args:
        message: Log message
        *args: Format arguments
        include_caller: Include caller info in message
        **kwargs: Additional context to log
    """
    if not _DEBUG_MODE:
        return

    if include_caller:
        filename, func, lineno = get_caller_info(skip_frames=1)
        caller_info = f"[{Path(filename).name}:{func}:{lineno}] "
    else:
        caller_info = ""

    full_message = f"{caller_info}{message}"
    if args:
        full_message = full_message % args

    if kwargs:
        full_message += f" | context: {kwargs}"

    logger.debug(full_message)


def debug_vars(**variables: Any) -> None:
    """Log variable values for debugging.

    Args:
        **variables: Variables to log (name=value pairs)
    """
    if not _DEBUG_MODE:
        return

    filename, func, lineno = get_caller_info(skip_frames=1)
    location = f"[{Path(filename).name}:{func}:{lineno}]"

    var_strs = []
    for name, value in variables.items():
        try:
            value_repr = repr(value)[:200]
        except (KeyError, ValueError, TypeError):
            value_repr = "<unprintable>"
        var_strs.append(f"{name}={value_repr}")

    logger.debug("%s Variables: %s", location, ", ".join(var_strs))


# =============================================================================
# Exception Debugging
# =============================================================================


def format_exception_with_locals(
    exc: BaseException,
    include_locals: bool = True,
    max_frames: int = 10,
) -> str:
    """Format exception with local variable context.

    Args:
        exc: Exception to format
        include_locals: Include local variables
        max_frames: Maximum frames to show

    Returns:
        Formatted exception string
    """
    lines = ["Exception: " + str(exc), ""]

    tb = exc.__traceback__
    frames: list[tuple[Any, int]] = []
    while tb is not None and len(frames) < max_frames:
        frames.append((tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next

    for frame, lineno in frames:
        filename = frame.f_code.co_filename
        func_name = frame.f_code.co_name

        lines.append(f'  File "{filename}", line {lineno}, in {func_name}')

        # Get source line
        line = linecache.getline(filename, lineno).strip()
        if line:
            lines.append(f"    {line}")

        # Include locals if requested
        if include_locals:
            lines.append("    Local variables:")
            for key, value in frame.f_locals.items():
                if not key.startswith("__"):
                    try:
                        value_repr = repr(value)[:100]
                    except (KeyError, ValueError, TypeError):
                        value_repr = "<unprintable>"
                    lines.append(f"      {key} = {value_repr}")
            lines.append("")

    return "\n".join(lines)


def debug_exception(
    exc: BaseException | None = None,
    log_level: int = logging.ERROR,
) -> None:
    """Log exception with enhanced debugging information.

    Args:
        exc: Exception to debug (or None to use current exception)
        log_level: Level to log at
    """
    if exc is None:
        exc_info = sys.exc_info()
        if exc_info[1] is not None:
            exc = exc_info[1]
        else:
            return

    formatted = format_exception_with_locals(exc, include_locals=_DEBUG_MODE)
    logger.log(log_level, "Exception details:\n%s", formatted)


@contextmanager
def debug_on_error() -> Generator[None, None, None]:
    """Context manager that provides debug info on error.

    Yields:
        None
    """
    try:
        yield
    except Exception as e:
        debug_exception(e)
        raise


# =============================================================================
# Diagnostic Utilities
# =============================================================================


@dataclass
class SystemDiagnostics:
    """System diagnostic information."""

    python_version: str
    platform: str
    process_id: int
    thread_count: int
    memory_mb: float
    cpu_count: int | None
    environment: dict[str, str]
    loaded_modules: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "process_id": self.process_id,
            "thread_count": self.thread_count,
            "memory_mb": self.memory_mb,
            "cpu_count": self.cpu_count,
            "environment": self.environment,
            "loaded_modules": self.loaded_modules,
        }


def get_system_diagnostics(
    include_env: bool = True,
    include_modules: bool = True,
    env_filter: list[str] | None = None,
) -> SystemDiagnostics:
    """Get system diagnostic information.

    Args:
        include_env: Include environment variables
        include_modules: Include loaded modules
        env_filter: Only include these env vars (None = safe subset)

    Returns:
        SystemDiagnostics object
    """
    # Safe environment variables to include by default
    safe_env_vars = env_filter or [
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "DEBUG",
        "LOG_LEVEL",
        "HOME",
        "USER",
    ]

    environment: dict[str, str] = {}
    if include_env:
        for var in safe_env_vars:
            if var in os.environ:
                environment[var] = os.environ[var]

    loaded_modules: list[str] = []
    if include_modules:
        loaded_modules = sorted(sys.modules.keys())[:50]  # Limit to first 50

    return SystemDiagnostics(
        python_version=sys.version,
        platform=sys.platform,
        process_id=os.getpid(),
        thread_count=threading.active_count(),
        memory_mb=get_memory_usage(),
        cpu_count=os.cpu_count(),
        environment=environment,
        loaded_modules=loaded_modules,
    )


def print_diagnostics(file: Any = None) -> None:
    """Print system diagnostics.

    Args:
        file: File to print to (default: stdout)
    """
    if file is None:
        file = sys.stdout

    diag = get_system_diagnostics()

    print("\n=== System Diagnostics ===", file=file)  # noqa: T201
    print(f"Python: {diag.python_version.split()[0]}", file=file)  # noqa: T201
    print(f"Platform: {diag.platform}", file=file)  # noqa: T201
    print(f"PID: {diag.process_id}", file=file)  # noqa: T201
    print(f"Threads: {diag.thread_count}", file=file)  # noqa: T201
    print(f"Memory: {diag.memory_mb:.2f} MB", file=file)  # noqa: T201
    print(f"CPUs: {diag.cpu_count}", file=file)  # noqa: T201
    print("\nEnvironment:", file=file)  # noqa: T201
    for key, value in diag.environment.items():
        print(f"  {key}={value[:50]}...", file=file)  # noqa: T201
    print("==========================\n", file=file)  # noqa: T201


# =============================================================================
# Breakpoint Helpers
# =============================================================================


def conditional_breakpoint(
    condition: bool,
    message: str = "",
) -> None:
    """Set a conditional breakpoint.

    Args:
        condition: Condition to trigger breakpoint
        message: Message to print before breaking
    """
    if condition:
        if message:
            logger.info(f"Breakpoint: {message}")
        breakpoint()


def debug_breakpoint() -> None:
    """Set a breakpoint only in debug mode."""
    if _DEBUG_MODE:
        breakpoint()


# =============================================================================
# Performance Watchdog
# =============================================================================


class PerformanceWatchdog:
    """Monitors performance and warns about slow operations."""

    def __init__(
        self,
        warn_threshold_ms: float = 1000.0,
        error_threshold_ms: float = 5000.0,
    ):
        """Initialize watchdog.

        Args:
            warn_threshold_ms: Threshold for warning (ms)
            error_threshold_ms: Threshold for error (ms)
        """
        self.warn_threshold = warn_threshold_ms
        self.error_threshold = error_threshold_ms
        self._timings: dict[str, list[float]] = {}

    def record(self, name: str, elapsed_ms: float) -> None:
        """Record a timing measurement.

        Args:
            name: Name of the operation
            elapsed_ms: Elapsed time in milliseconds
        """
        if name not in self._timings:
            self._timings[name] = []
        self._timings[name].append(elapsed_ms)

        if elapsed_ms >= self.error_threshold:
            logger.error(
                "SLOW: %s took %.2fms (threshold: %.2fms)",
                name,
                elapsed_ms,
                self.error_threshold,
            )
        elif elapsed_ms >= self.warn_threshold:
            logger.warning(
                "Slow: %s took %.2fms (threshold: %.2fms)",
                name,
                elapsed_ms,
                self.warn_threshold,
            )

    @contextmanager
    def watch(self, name: str) -> Generator[None, None, None]:
        """Context manager to watch operation performance.

        Args:
            name: Name of the operation

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(name, elapsed_ms)

    def get_stats(self, name: str) -> dict[str, float]:
        """Get statistics for an operation.

        Args:
            name: Name of the operation

        Returns:
            Dictionary with min, max, avg, count
        """
        timings = self._timings.get(name, [])
        if not timings:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 0}

        return {
            "min": min(timings),
            "max": max(timings),
            "avg": sum(timings) / len(timings),
            "count": len(timings),
        }

    def report(self) -> str:
        """Generate a performance report.

        Returns:
            Formatted report string
        """
        lines = ["Performance Report", "=" * 50]

        for name in sorted(self._timings.keys()):
            stats = self.get_stats(name)
            lines.append(
                f"{name}: "
                f"avg={stats['avg']:.2f}ms, "
                f"min={stats['min']:.2f}ms, "
                f"max={stats['max']:.2f}ms, "
                f"count={stats['count']}"
            )

        return "\n".join(lines)


# Global watchdog instance
_watchdog = PerformanceWatchdog()


def get_watchdog() -> PerformanceWatchdog:
    """Get the global performance watchdog.

    Returns:
        Global PerformanceWatchdog instance
    """
    return _watchdog


# =============================================================================
# Deprecation Helpers
# =============================================================================


def deprecated(
    reason: str = "",
    removal_version: str | None = None,
) -> Callable[[F], F]:
    """Mark a function as deprecated.

    Args:
        reason: Reason for deprecation
        removal_version: Version when it will be removed

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            message = f"{func.__name__} is deprecated"
            if reason:
                message += f": {reason}"
            if removal_version:
                message += f" (will be removed in {removal_version})"

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Cleanup on Exit
# =============================================================================


def _cleanup() -> None:
    """Cleanup function called on exit."""
    if _DEBUG_MODE:
        report = _watchdog.report()
        if _watchdog._timings:
            logger.debug("Final performance report:\n%s", report)


atexit.register(_cleanup)
