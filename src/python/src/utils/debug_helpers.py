"""Debug helper utilities.

Provides debug logging, exception debugging, system diagnostics,
breakpoint helpers, performance watchdog, and deprecation decorators.
"""

import functools
import linecache
import logging
import os
import sys
import threading
import time
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from utils.debug_memory import get_memory_usage
from utils.debug_tracing import get_caller_info

# Type variables
F = TypeVar("F", bound=Callable[..., Any])

# Module-level logger
logger = logging.getLogger(__name__)


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
    from utils.debug_utils import is_debug_mode

    if not is_debug_mode():
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
    from utils.debug_utils import is_debug_mode

    if not is_debug_mode():
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
    assert exc is not None, "exc must be provided"
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
    assert log_level is not None, "log_level must be provided"
    from utils.debug_utils import is_debug_mode

    if exc is None:
        exc_info = sys.exc_info()
        if exc_info[1] is not None:
            exc = exc_info[1]
        else:
            return

    formatted = format_exception_with_locals(exc, include_locals=is_debug_mode())
    logger.log(log_level, "Exception details:\n%s", formatted)


@contextmanager
def debug_on_error() -> Generator[None, None, None]:
    """Context manager that provides debug info on error.

    Yields:
        None
    """
    try:
        yield
    except Exception as e:  # noqa: BLE001
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
    assert include_env is not None, "include_env must be provided"
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


def format_diagnostics() -> str:
    """Format system diagnostics as a string.

    Returns:
        Formatted diagnostics string.
    """
    diag = get_system_diagnostics()
    lines = [
        "\n=== System Diagnostics ===",
        f"Python: {diag.python_version.split()[0]}",
        f"Platform: {diag.platform}",
        f"PID: {diag.process_id}",
        f"Threads: {diag.thread_count}",
        f"Memory: {diag.memory_mb:.2f} MB",
        f"CPUs: {diag.cpu_count}",
        "\nEnvironment:",
    ]
    for key, value in diag.environment.items():
        lines.append(f"  {key}={value[:50]}...")
    lines.append("==========================\n")
    return "\n".join(lines)


def print_diagnostics(file: Any = None) -> None:
    """Log system diagnostics (or write to *file* for backward compat).

    Args:
        file: File to write to. When *None*, diagnostics are emitted
              via the module logger at INFO level instead of stdout.
    """
    text = format_diagnostics()
    if file is not None:
        file.write(text + "\n")
    else:
        logger.info(text)


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
    from utils.debug_utils import is_debug_mode

    if is_debug_mode():
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
        assert warn_threshold_ms is not None, "warn_threshold_ms must be provided"
        self.warn_threshold = warn_threshold_ms
        self.error_threshold = error_threshold_ms
        self._timings: dict[str, list[float]] = {}

    def record(self, name: str, elapsed_ms: float) -> None:
        """Record a timing measurement.

        Args:
            name: Name of the operation
            elapsed_ms: Elapsed time in milliseconds
        """
        assert name is not None, "name must be provided"
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
        assert name is not None, "name must be provided"
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
        assert name is not None, "name must be provided"
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

    assert reason is not None, "reason must be provided"

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
