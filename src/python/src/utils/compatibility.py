"""
Compatibility Shim for Python Versions < 3.11

This module provides backports for features introduced in Python 3.11,
allowing the codebase to run on Python 3.10 (Ubuntu 22.04 default).
"""

import logging
import sys
from enum import Enum

logger = logging.getLogger(__name__)


def check_python_version() -> None:
    """
    Check Python version and provide a friendly error message if incompatible.

    Raises:
        SystemExit: If Python version is < 3.10
    """
    if sys.version_info < (3, 10):  # noqa: UP036
        logger.critical("Critical Error: This application requires Python 3.10 or newer.")
        logger.critical(f"Current version: {sys.version}")
        logger.critical("Please upgrade Python or use a Python 3.10+ environment.")
        sys.exit(1)


# Check Python version when module is imported
check_python_version()

if sys.version_info >= (3, 11):  # noqa: UP036
    from datetime import UTC
else:
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

# Backport StrEnum
if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum as _StrEnum

    StrEnum = _StrEnum
else:

    class StrEnum(str, Enum):  # noqa: UP042 - Intentional backport for Python < 3.11
        """
        Enum where members are also (and must be) strings.
        Backport for Python < 3.11.
        """

        def __str__(self) -> str:
            return str(self.value)

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}.{self._name_}"
