"""Logging configuration - DEPRECATED: Use utils.logging_utils instead.

This module re-exports from the central utils.logging_utils module.
New code should use utils.logging_utils directly.
"""

from __future__ import annotations

import warnings

# Re-export from shared logging utilities
from utils.logging_utils import (
    DEFAULT_FORMAT,
    JsonFormatter,
    get_logger,
    init_default_logging,
)

# Issue deprecation warning
warnings.warn(
    "logging_config is deprecated. Use utils.logging_utils directly.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DEFAULT_FORMAT", "JsonFormatter", "get_logger", "init_default_logging"]
