"""Execution tracing and stack inspection utilities.

Provides function call tracing, stack frame inspection,
and caller information utilities.
"""

import functools
import inspect
import logging
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

# Type variables
F = TypeVar("F", bound=Callable[..., Any])

# Module-level logger
logger = logging.getLogger(__name__)


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
    # Import here to avoid circular dependency at module level
    from utils.debug_utils import _DEBUG_MODE

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
