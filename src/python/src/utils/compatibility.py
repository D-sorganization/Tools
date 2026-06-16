"""Compatibility Shim for Python Versions < 3.11.

The canonical UTC and StrEnum compatibility primitives live in the shared
compatibility module. This legacy ``utils`` module preserves the historical
version check and re-exports those shared primitives so callers do not split
class identity across two backport implementations.
"""

import logging
import sys

from compatibility import UTC, StrEnum

logger = logging.getLogger(__name__)

__all__ = ["UTC", "StrEnum", "check_python_version"]


def check_python_version() -> None:
    """
    Check Python version and provide a friendly error message if incompatible.

    Raises:
        SystemExit: If Python version is < 3.10
    """
    if sys.version_info < (3, 10):  # noqa: UP036
        logger.critical(
            "Critical Error: This application requires Python 3.10 or newer."
        )
        logger.critical(f"Current version: {sys.version}")
        logger.critical("Please upgrade Python or use a Python 3.10+ environment.")
        sys.exit(1)


# Check Python version when module is imported
check_python_version()
