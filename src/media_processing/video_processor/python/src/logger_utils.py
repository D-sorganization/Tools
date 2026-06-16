"""Logging utilities - DEPRECATED: Use utils.logging_utils instead.

This module re-exports from the central utils.logging_utils module.
New code should use utils.logging_utils directly.
"""

import logging
import sys
import warnings
from typing import Any

import utils.logging_utils as _logging_utils

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


def __getattr__(name: str) -> Any:
    if name == "TORCH_AVAILABLE":
        return "torch" in sys.modules or _logging_utils.TORCH_AVAILABLE
    if name == "NUMPY_AVAILABLE":
        return _logging_utils.NUMPY_AVAILABLE
    raise AttributeError(name)
