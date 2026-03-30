"""Logging utilities - Uses shared logging utilities.

This module re-exports from the central utils.logging_utils module.
New code should use utils.logging_utils directly.
"""

import logging
import warnings

# Try to import from shared utils
try:
    from utils.logging_utils import (
        DEFAULT_FORMAT,
        DEFAULT_SEED,
        get_logger,
        logger,
        set_seeds,
        setup_logging,
    )

    # Provide backward-compatible aliases
    LOG_FORMAT = DEFAULT_FORMAT
    LOG_LEVEL = logging.INFO

except ImportError:
    # Minimal fallback if shared utils not available
    import sys
    from pathlib import Path

    DEFAULT_SEED = 42
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_FORMAT = LOG_FORMAT
    LOG_LEVEL = logging.INFO
    logger = logging.getLogger(__name__)

    def get_logger(
        name: str | None = None,
        level: int | str = logging.INFO,
        use_simple_format: bool = False,
    ) -> logging.Logger:
        """Get a logger instance."""
        if name is None:
            name = __name__
        return logging.getLogger(name)

    def setup_logging(
        level: int | str = logging.INFO,
        log_file: Path | str | None = None,
        format_string: str | None = None,
        json_logs: bool = False,
        force: bool = False,
    ) -> None:
        """Set up logging configuration."""
        fmt = format_string if format_string is not None else LOG_FORMAT
        logging.basicConfig(
            level=level,
            format=fmt,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def set_seeds(seed: int = DEFAULT_SEED) -> None:
        """Set random seeds for reproducibility."""
        import random

        if seed < 0:
            raise ValueError(f"expected non-negative integer, got: {seed}")
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)  # noqa: NPY002 — legacy compat required here
        except ImportError:
            pass


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
