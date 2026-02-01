"""Logging utilities - DEPRECATED: Use utils.logging_utils instead.

This module re-exports from the central utils.logging_utils module.
New code should use utils.logging_utils directly.
"""

import logging
import warnings

# Re-export from shared logging utilities
from utils.logging_utils import (
    DEFAULT_FORMAT,
    DEFAULT_SEED,
    get_logger,
    logger,
    set_seeds,
    setup_logging,
)

# Backward compatibility aliases
LOG_FORMAT = DEFAULT_FORMAT
LOG_LEVEL = logging.INFO

# Issue deprecation warning
warnings.warn(
    "logger_utils is deprecated. Use utils.logging_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DEFAULT_FORMAT",
    "DEFAULT_SEED",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "get_logger",
    "logger",
    "set_seeds",
    "setup_logging",
]
