"""Compatibility shims for supporting older Python versions (3.10+)."""

import sys
from enum import Enum

# Backport datetime.UTC (Added in Python 3.11)
if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    from datetime import timezone
    UTC = timezone.utc

# Backport StrEnum (Added in Python 3.11)
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Enum where members are also (and must be) strings."""

        def __str__(self) -> str:
            return str(self.value)
