# mypy: ignore-errors
"""Logging configuration for data_processor.

Fixed in issue #530: removed fragile dependency on ``utils.logging_utils``.
Updated in issue #682: removed ``sys.path`` hack; relies on package
installation (``pip install -e .``) or pytest ``pythonpath`` config.
"""

from __future__ import annotations

import logging

# Try to import from shared utils; fall back to local implementations.
# With a proper editable install the ``utils`` package is available
# without any sys.path manipulation.
try:
    from utils.logging_utils import (
        DEFAULT_FORMAT,
        JsonFormatter,
        get_logger,
        init_default_logging,
    )
except ImportError:
    # Local fallback implementations
    DEFAULT_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # type: ignore[no-redef]

    class JsonFormatter(logging.Formatter):  # type: ignore[no-redef]
        """JSON log formatter (inline fallback)."""

        def format(self, record: logging.LogRecord) -> str:
            import json

            return json.dumps(
                {
                    "time": self.formatTime(record),
                    "level": record.levelname,
                    "name": record.name,
                    "message": record.getMessage(),
                }
            )

    def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:  # type: ignore[no-redef]
        """Get a configured logger (inline fallback)."""
        if not (name is not None):
            raise ValueError("name must be provided")
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    def init_default_logging(level: int = logging.INFO) -> None:  # type: ignore[no-redef]
        """Initialize default logging (inline fallback)."""
        logging.basicConfig(level=level, format=DEFAULT_FORMAT)


__all__ = ["DEFAULT_FORMAT", "JsonFormatter", "get_logger", "init_default_logging"]
