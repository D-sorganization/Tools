"""Logging utilities - DEPRECATED: Use utils.logging_utils instead.

This module is maintained for backward compatibility.
New code should use src/python/src/utils/logging_utils.py
"""

import sys
import warnings
from pathlib import Path

# Add utils to path for import
repo_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src" / "python" / "src"))

try:
    from utils.logging_utils import (
        DEFAULT_SEED,
        get_logger,
        logger,
        set_seeds,
        setup_logging,
    )

    # Backward compatibility constants
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = 20  # logging.INFO

    # Issue deprecation warning
    warnings.warn(
        "logger_utils is deprecated. Use utils.logging_utils instead.",
        DeprecationWarning,
        stacklevel=2,
    )
except ImportError:
    # Fallback if shared utility not available
    import logging

    DEFAULT_SEED = 42
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = logging.INFO
    logger = logging.getLogger(__name__)

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

    def setup_logging(level: int = LOG_LEVEL, format_string: str = LOG_FORMAT) -> None:
        logging.basicConfig(
            level=level,
            format=format_string,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def set_seeds(seed: int = DEFAULT_SEED) -> None:
        import random

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)  # noqa: NPY002
        except ImportError:
            pass

__all__ = ["DEFAULT_SEED", "LOG_FORMAT", "LOG_LEVEL", "get_logger", "logger", "set_seeds", "setup_logging"]
