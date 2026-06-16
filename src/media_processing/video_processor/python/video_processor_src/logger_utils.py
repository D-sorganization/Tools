"""Compatibility logging shim for the video processor package."""

import logging
import warnings

from utils import logging_utils as _logging_utils

DEFAULT_FORMAT = _logging_utils.DEFAULT_FORMAT
DEFAULT_SEED = _logging_utils.DEFAULT_SEED
LOG_FORMAT = DEFAULT_FORMAT
LOG_LEVEL = logging.INFO
NUMPY_AVAILABLE = _logging_utils.NUMPY_AVAILABLE
SIMPLE_FORMAT = _logging_utils.SIMPLE_FORMAT
TORCH_AVAILABLE = _logging_utils.TORCH_AVAILABLE
get_logger = _logging_utils.get_logger
logger = _logging_utils.logger
setup_logging = _logging_utils.setup_logging


def _sync_backend_flags() -> None:
    """Mirror canonical optional-backend flags for compatibility callers."""
    global NUMPY_AVAILABLE, TORCH_AVAILABLE
    NUMPY_AVAILABLE = _logging_utils.NUMPY_AVAILABLE
    TORCH_AVAILABLE = _logging_utils.TORCH_AVAILABLE


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds through the canonical logging utility."""
    _logging_utils.set_seeds(seed)
    _sync_backend_flags()


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
    "NUMPY_AVAILABLE",
    "SIMPLE_FORMAT",
    "TORCH_AVAILABLE",
    "get_logger",
    "logger",
    "set_seeds",
    "setup_logging",
]
