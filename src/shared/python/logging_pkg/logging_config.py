"""Logging configuration module for shared Python utilities.

Provides get_logger and setup_logging with a minimal, zero-dependency
implementation that works across the ai/ and data_explorer/ subsystems.
"""

import logging
import sys
from pathlib import Path
from typing import Any

DEFAULT_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT: str = "%(levelname)s: %(message)s"


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
    import json

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if force:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    if not root_logger.handlers:
        if not format_string:
            format_string = SIMPLE_FORMAT if json_logs else DEFAULT_FORMAT

        if json_logs:

            class JsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
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

            formatter: logging.Formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(format_string)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        root_logger.addHandler(file_handler)