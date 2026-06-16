"""Deprecated logging utility shim for ``video_processor_src``."""

import logging
import sys
import warnings
from typing import Any

import utils.logging_utils as _logging_utils
from utils.logging_utils import (
    DEFAULT_FORMAT,
    DEFAULT_SEED,
    get_logger,
    logger,
    set_seeds,
    setup_logging,
)

LOG_FORMAT = DEFAULT_FORMAT
LOG_LEVEL = logging.INFO


# Issue deprecation warning for direct imports
warnings.warn(
    "video_processor_src.logger_utils is deprecated. Use utils.logging_utils instead.",
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
