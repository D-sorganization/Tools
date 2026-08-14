"""Python-version compatibility for :class:`enum.StrEnum` in this package.

``enum.StrEnum`` is new in Python 3.11, and the CI test matrix still runs 3.10.
An unguarded ``from enum import StrEnum`` therefore raises
``ImportError: cannot import name 'StrEnum' from 'enum'`` on 3.10 — and because
``main`` imports the SCADA modules transitively, a single unguarded import there
aborts collection of every test module that imports the app, not just the one
that owns the enum.

This mirrors ``src/shared/python/compatibility.py``, which already provides the
same backport for the shared tree. It is duplicated here rather than imported
because this package deliberately uses flat intra-package imports (``import
historian``, ``from signal_quality import SignalFrame``) and is run with the
backend directory on ``sys.path``, so ``shared.python.compatibility`` is not
importable from it.

The ``TYPE_CHECKING`` branch keeps type checkers on the real 3.11 symbol, so the
backport never weakens inference.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import TYPE_CHECKING

__all__ = ["StrEnum"]

if TYPE_CHECKING:
    from enum import StrEnum
elif sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:

    class StrEnum(str, Enum):  # noqa: UP042
        """Backport of :class:`enum.StrEnum` for Python 3.10.

        Subclassing ``str`` makes members compare equal to their values, and the
        explicit ``__str__`` keeps ``str(member)`` as the value rather than
        ``"Class.MEMBER"`` — the behaviour 3.11's ``StrEnum`` guarantees and
        which JSON payloads and audit records in this package rely on.
        """

        def __str__(self) -> str:
            return str(self.value)
