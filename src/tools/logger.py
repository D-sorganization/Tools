"""Unified logging configuration for Tools repository.

DEPRECATED: Use utils.logging_utils directly for new code.
This module re-exports from the centralized logging_utils module.
"""

import warnings

# Re-export from centralized logging utilities
from utils.logging_utils import (
    DEFAULT_FORMAT,
    get_logger,
    init_default_logging,
    setup_logging,
)

# Issue deprecation warning
warnings.warn(
    "tools.logger is deprecated. Use utils.logging_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DEFAULT_FORMAT", "get_logger", "init_default_logging", "setup_logging"]
