"""Memory profiling utilities.

Provides memory usage tracking, decorators, and context managers
for monitoring memory consumption.
"""

import functools
import gc
import logging
import sys
import tracemalloc
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

# Type variables
F = TypeVar("F", bound=Callable[..., Any])

# Module-level logger
logger = logging.getLogger(__name__)


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
    # Import here to avoid circular dependency at module level
    from utils.debug_utils import is_debug_mode

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

        if log_result and is_debug_mode():
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
