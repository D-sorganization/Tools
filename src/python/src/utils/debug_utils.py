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

This module serves as a facade, composing the following submodules:
- debug_profiling: Profiling and timing utilities
- debug_memory: Memory profiling
- debug_tracing: Execution tracing and stack inspection
- debug_helpers: Logging, exceptions, diagnostics, watchdog, deprecation
"""

import atexit
import functools
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar

# Re-export all public symbols for backward compatibility
from utils.debug_helpers import (  # noqa: F401
    PerformanceWatchdog,
    SystemDiagnostics,
    conditional_breakpoint,
    debug_breakpoint,
    debug_exception,
    debug_log,
    debug_on_error,
    debug_vars,
    deprecated,
    format_exception_with_locals,
    get_system_diagnostics,
    print_diagnostics,
)
from utils.debug_memory import (  # noqa: F401
    MemoryStats,
    get_memory_usage,
    memory_profile,
    memory_tracker,
)
from utils.debug_profiling import (  # noqa: F401
    ProfileResult,
    TimingStats,
    profile,
    profile_block,
    timed,
    timer,
)
from utils.debug_tracing import (  # noqa: F401
    ExecutionTracer,
    StackFrame,
    get_call_stack,
    get_caller_info,
    print_call_stack,
    trace_calls,
)

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Module-level logger
logger = logging.getLogger(__name__)

# Debug mode holder (avoids mutable global + global keyword)
class _DebugState:
    enabled: bool = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


# =============================================================================
# Debug Mode Management
# =============================================================================


def is_debug_mode() -> bool:
    """Check if debug mode is enabled.

    Returns:
        True if debug mode is enabled
    """
    return _DebugState.enabled


def set_debug_mode(enabled: bool) -> None:
    """Set the debug mode flag.

    Args:
        enabled: Whether to enable debug mode
    """
    _DebugState.enabled = enabled
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
        if _DebugState.enabled:
            return func(*args, **kwargs)
        return None

    return wrapper  # type: ignore[return-value]


# =============================================================================
# Global watchdog instance
# =============================================================================

_watchdog = PerformanceWatchdog()


def get_watchdog() -> PerformanceWatchdog:
    """Get the global performance watchdog.

    Returns:
        Global PerformanceWatchdog instance
    """
    return _watchdog


# =============================================================================
# Cleanup on Exit
# =============================================================================


def _cleanup() -> None:
    """Cleanup function called on exit."""
    if _DebugState.enabled:
        report = _watchdog.report()
        if _watchdog._timings:
            logger.debug("Final performance report:\n%s", report)


atexit.register(_cleanup)
