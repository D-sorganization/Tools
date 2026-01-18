"""
Compatibility Shim for Python Versions < 3.11

This module provides backports for features introduced in Python 3.11,
allowing the codebase to run on Python 3.10 (Ubuntu 22.04 default).
"""

import sys
from enum import Enum

try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:
    from datetime import timezone
    UTC = timezone.utc


# Backport StrEnum
if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum as _StrEnum

    StrEnum = _StrEnum
else:

    class StrEnum(str, Enum):
        """
        Enum where members are also (and must be) strings.
        Backport for Python < 3.11.
        """

        def __str__(self) -> str:
            return str(self.value)

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}.{self._name_}"
