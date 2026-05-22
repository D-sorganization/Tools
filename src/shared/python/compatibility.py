"""Shared Python-version compatibility helpers for shared modules."""

from __future__ import annotations

import sys
from datetime import timezone
from enum import Enum

if sys.version_info >= (3, 11):  # noqa: UP036
    from datetime import UTC
    from enum import StrEnum
else:
    UTC = timezone.utc  # noqa: UP017

    class StrEnum(str, Enum):
        """Backport of :class:`enum.StrEnum` for Python 3.10."""

        def __str__(self) -> str:
            return str(self.value)
