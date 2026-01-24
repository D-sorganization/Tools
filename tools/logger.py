"""Unified logging configuration for Tools repository."""

import logging
import sys
from typing import Optional


def setup_logging(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Configure and return a standard logger.

    Args:
        name: Logger name (usually __name__).
        log_file: Optional filename to write logs to.
        level: Logging level (default INFO).
        format_string: Log message format.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(format_string)

    # Stream Handler (Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        try:
            # You might want to ensure a logs directory exists?
            # For now, keeping original behavior of local file or specified path
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Don't crash if file logging fails, just warn to stderr
            sys.stderr.write(f"Failed to setup file logging to {log_file}: {e}\n")

    return logger
