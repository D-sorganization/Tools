"""Logging configuration - Uses shared logging utilities.

This module uses the shared logging_utils for consistency.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Add utils to path for import
try:
    from utils.path_helpers import ensure_utils_in_path
    ensure_utils_in_path()
except ImportError:
    # Fallback
    from utils.path_helpers import get_project_root_from_file
    repo_root = get_project_root_from_file(__file__)
    sys.path.insert(0, str(repo_root / "src" / "python" / "src"))

try:
    from utils.logging_utils import (
        DEFAULT_FORMAT,
        JsonFormatter,
        get_logger,
        init_default_logging,
    )

    # Issue deprecation warning for direct use
    warnings.warn(
        "logging_config is deprecated. Use utils.logging_utils directly.",
        DeprecationWarning,
        stacklevel=2,
    )
except ImportError:
    # Fallback if shared utility not available
    import json
    import logging
    from typing import Any

    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class JsonFormatter(logging.Formatter):
        """Simple JSON log formatter for machine readability."""

        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            if record.stack_info:
                payload["stack"] = self.formatStack(record.stack_info)
            extra = getattr(record, "extra", None)
            if isinstance(extra, dict):
                payload.update(extra)
            return json.dumps(payload, default=str)

    def init_default_logging(
        level: str = "INFO", json_logs: bool = False
    ) -> logging.Logger:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                JsonFormatter() if json_logs else logging.Formatter(DEFAULT_FORMAT)
            )
            root_logger.addHandler(handler)
        root_logger.setLevel(level.upper())
        return root_logger

    def get_logger(name: str = __name__) -> logging.Logger:
        init_default_logging()
        return logging.getLogger(name)

__all__ = ["DEFAULT_FORMAT", "JsonFormatter", "get_logger", "init_default_logging"]
