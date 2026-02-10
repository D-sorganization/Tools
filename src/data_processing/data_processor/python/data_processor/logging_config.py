"""Logging configuration for data_processor.

Fixed in issue #530: removed fragile dependency on ``utils.logging_utils``
which required ``src/python/src`` to already be on ``sys.path``.  Now
provides a self-contained implementation with an optional import from
the shared utils when available.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Self-contained path bootstrap (see issue #530)
# ---------------------------------------------------------------------------
def _ensure_utils_on_path() -> None:
    """Add the shared utils directory to sys.path if not already present."""
    current = Path(__file__).resolve().parent
    for _ in range(15):
        if any((current / marker).exists() for marker in (".git", "pyproject.toml")):
            utils_path = current / "src" / "python" / "src"
            if utils_path.exists() and str(utils_path) not in sys.path:
                sys.path.insert(0, str(utils_path))
            return
        parent = current.parent
        if parent == current:
            break
        current = parent


_ensure_utils_on_path()

# Try to import from shared utils; fall back to local implementations
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
