"""Logging utilities with reproducible random seed management.

This module provides standardized logging configuration for the Tools repository.
Use get_logger() to obtain a properly configured logger for any module.
"""

import logging
import random
import sys
from pathlib import Path

# Constants with clear sources and units
DEFAULT_SEED: int = (
    42  # Standard reproducibility seed per scientific computing best practices
)

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger for a module.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler with simple format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT))
        logger.addHandler(console_handler)

    return logger


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    format_string: str = DEFAULT_FORMAT,
) -> None:
    """Configure root logging for the application.

    Args:
        level: Logging level for root logger.
        log_file: Optional path to log file.
        format_string: Log message format.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


# Module-level logger for backwards compatibility
logger = get_logger(__name__)


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value to use for all random number generators.

    """
    random.seed(seed)

    # Import numpy only when needed to avoid module-level import
    try:
        import numpy as np

        # Use modern numpy random generator
        np.random.seed(seed)
        logger.info("Seeds set: %d", seed)
    except ImportError:
        logger.warning("NumPy not available, skipping numpy seed setting")
