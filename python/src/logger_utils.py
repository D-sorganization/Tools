"""Logging utilities with reproducible random seed management."""

import logging
import random

# Use a local import to avoid module-level numpy import
logger = logging.getLogger(__name__)

# Constants with clear sources and units
DEFAULT_SEED: int = 42  # Standard reproducibility seed per scientific computing best practices


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
        rng = np.random.default_rng(seed)
        logger.info("Seeds set: %d", seed)
    except ImportError:
        logger.warning("NumPy not available, skipping numpy seed setting")
