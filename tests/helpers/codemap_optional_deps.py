"""Optional-dependency detection for the codemap test modules (issue #4497).

The codemap parsers need the optional ``codemap`` extra (a tree-sitter stack
that CI never installs). Those tests must therefore be skipped when the stack
is absent -- but the skip has to be scoped to the codemap test modules.

``tests/unit/codemap/conftest.py`` previously did this with a
``pytest_collection_modifyitems`` hook. That hook is handed **every item in
the session**, not just the ones under its own directory, so co-collecting
the codemap directory with anything else silenced the whole run (8411
collected, 8411 skipped, exit 0).

Exporting a marker that each module applies via ``pytestmark`` keeps the skip
inside the file that opts into it: a module-level marker structurally cannot
reach another module.

This module lives in ``tests/helpers`` (a real package) rather than beside the
codemap tests because CI runs pytest with ``--import-mode=importlib``, under
which a bare sibling import such as ``from _optional_deps import ...`` raises
``ModuleNotFoundError``.

Usage::

    from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

    pytestmark = CODEMAP_DEPS_SKIP
"""

from __future__ import annotations

import pytest

#: Every distribution provided by the ``codemap`` extra that the parser
#: stack imports at runtime.
CODEMAP_DEP_MODULES: tuple[str, ...] = (
    "tree_sitter",
    "tree_sitter_python",
    "tree_sitter_javascript",
    "tree_sitter_typescript",
    "tree_sitter_rust",
    "tree_sitter_markdown",
    "pydantic",
)


def missing_codemap_deps() -> list[str]:
    """Return the codemap optional dependencies that cannot be imported."""
    missing: list[str] = []
    for name in CODEMAP_DEP_MODULES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


MISSING_CODEMAP_DEPS: list[str] = missing_codemap_deps()

CODEMAP_SKIP_REASON = (
    "codemap optional deps missing: "
    + ", ".join(MISSING_CODEMAP_DEPS)
    + " (install with: pip install -e '.[codemap]')"
    if MISSING_CODEMAP_DEPS
    else "codemap optional deps present"
)

#: Apply with ``pytestmark = CODEMAP_DEPS_SKIP`` at module scope.
#:
#: Deliberately a ``skipif`` marker rather than ``pytest.importorskip``: the
#: CI lane asserts that ``tests/unit/codemap`` collects at least one test
#: (the "no vacuous core test" guard from #3324). ``importorskip`` raises at
#: import time and collects zero items, which would trip that guard; a marker
#: collects every test and reports it skipped.
CODEMAP_DEPS_SKIP = pytest.mark.skipif(
    bool(MISSING_CODEMAP_DEPS),
    reason=CODEMAP_SKIP_REASON,
)
