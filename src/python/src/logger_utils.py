"""Logging utilities - DEPRECATED: Use utils.logging_utils instead.

This module is maintained for backward compatibility.
New code should use src/python/src/utils/logging_utils.py
"""

import warnings

# Import from shared utility. This module is kept importable as top-level
# ``logger_utils`` for legacy callers that still have ``src/python/src`` on
# PYTHONPATH.
from utils.logging_utils import (
    DEFAULT_FORMAT,
    DEFAULT_SEED,
    SIMPLE_FORMAT,
    get_logger,
    logger,
    set_seeds,
    setup_logging,
)

# Issue deprecation warning
warnings.warn(
    "logger_utils is deprecated. Use utils.logging_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DEFAULT_FORMAT",
    "DEFAULT_SEED",
    "SIMPLE_FORMAT",
    "get_logger",
    "logger",
    "set_seeds",
    "setup_logging",
]
