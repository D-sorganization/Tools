"""Codemap test package configuration.

This file intentionally defines **no** ``pytest_collection_modifyitems`` hook.

It used to. The hook marked every item it was handed as skipped when the
optional tree-sitter stack was missing, and pytest hands that hook the whole
session's items -- not just the ones under this directory. Co-collecting this
directory with any other tests silenced them all::

    pytest tests/shared/python/theme                    -> 107 passed
    pytest tests/shared/python/theme tests/unit/codemap -> 198 skipped, 0 passed

The nightly full-suite lane reported 8411 collected / 8411 skipped and still
exited 0, because skips are not failures and junit is written either way.
See issue #4497.

The optional-dependency skip now lives on each test module as::

    from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

    pytestmark = CODEMAP_DEPS_SKIP

which cannot affect any module other than the one that declares it.
``tests/architecture/test_no_session_wide_skip_leak_4497.py`` fails if a
session-wide skip hook is reintroduced anywhere under ``tests/``.
"""

from __future__ import annotations
