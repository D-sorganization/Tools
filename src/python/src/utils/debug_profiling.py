"""Profiling and timing utilities.

Provides decorators and context managers for function profiling
and execution timing.
"""

import cProfile
import functools
import io
import logging
import pstats
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

# Type variables
F = TypeVar("F", bound=Callable[..., Any])

# Module-level logger
logger = logging.getLogger(__name__)


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
    # Import here to avoid circular dependency at module level
    assert sort_by is not None, "sort_by must be provided"
    from utils.debug_utils import is_debug_mode

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
                if is_debug_mode():
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
    # Import here to avoid circular dependency at module level
    assert name is not None, "name must be provided"
    from utils.debug_utils import is_debug_mode

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

        if log_result and is_debug_mode():
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

    assert log_level is not None, "log_level must be provided"

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
    assert name is not None, "name must be provided"
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
