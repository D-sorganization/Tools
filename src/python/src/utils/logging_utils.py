"""
Comprehensive logging utilities for consistent logging across the repository.

This module provides standardized logging configuration, following DRY principles
and consolidating duplicate logging setup code.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

# Try optional imports
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

# Constants
DEFAULT_SEED: int = 42  # Standard reproducibility seed
DEFAULT_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT: str = "%(levelname)s: %(message)s"


class JsonFormatter(logging.Formatter):
    """JSON log formatter for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        assert record is not None, "record must be provided"
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(
    name: str | None = None,
    level: int | str = logging.INFO,
    use_simple_format: bool = False,
) -> logging.Logger:
    """Get a configured logger for a module.

    Args:
        name: Logger name, typically __name__ of the calling module.
              If None, uses calling module's __name__.
        level: Logging level (default: INFO). Can be int or string.
        use_simple_format: Use simple format instead of default format.

    Returns:
        Configured logging.Logger instance.
    """
    assert level is not None, "level must be provided"
    if name is None:
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "root")
        else:
            name = "root"

    logger = logging.getLogger(name)

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        format_string = SIMPLE_FORMAT if use_simple_format else DEFAULT_FORMAT
        console_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(console_handler)

    return logger


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    format_string: str | None = None,
    json_logs: bool = False,
    force: bool = False,
) -> None:
    """Configure root logging for the application.

    Args:
        level: Logging level for root logger. Can be int or string.
        log_file: Optional path to log file.
        format_string: Log message format. Uses default if None.
        json_logs: Use JSON formatting instead of text.
        force: Override any existing configuration.
    """
    # Convert string level to int if needed
    assert level is not None, "level must be provided"
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    # Choose formatter
    if json_logs:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(format_string or DEFAULT_FORMAT)

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=force,
    )


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy (if available),
    and PyTorch (if available).

    Args:
        seed: Random seed value (default: 42)

    Raises:
        ValueError: If seed is negative
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got: {seed}")

    random.seed(seed)

    if NUMPY_AVAILABLE:
        np.random.seed(seed)  # noqa: NPY002

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)

    logger = get_logger(__name__)
    logger.info("Random seeds set to: %d", seed)


def init_default_logging(
    level: str | int = logging.INFO,
    json_logs: bool = False,
) -> logging.Logger:
    """Initialize process-wide logging if no handlers exist.

    Args:
        level: Logging level (default: INFO)
        json_logs: Use JSON formatting

    Returns:
        Root logger instance
    """
    assert level is not None, "level must be provided"
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if json_logs:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        root_logger.addHandler(handler)

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger.setLevel(level)
    return root_logger


class LogExecutionTime:
    """Context manager that logs the wall-clock duration of a block.

    Example::

        with LogExecutionTime("my_operation"):
            do_something()
        # INFO: my_operation completed in 0.1234s
    """

    def __init__(
        self,
        name: str,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        self.name = name
        self._logger = logger_instance or get_logger(name)

    def __enter__(self) -> "LogExecutionTime":
        import time

        self._start = time.perf_counter()
        self._logger.debug("Starting %s…", self.name)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        import time

        duration = time.perf_counter() - self._start
        self._logger.info("%s completed in %.4fs", self.name, duration)


def log_execution_time(name: str) -> LogExecutionTime:
    """Convenience factory for :class:`LogExecutionTime`."""
    return LogExecutionTime(name)


# Module-level logger for convenience
logger = get_logger(__name__)
