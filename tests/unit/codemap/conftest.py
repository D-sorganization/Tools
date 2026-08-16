"""Skip the codemap unit tests when the optional parser stack isn't installed.

The CI test lane runs on a stock Python that doesn't install the optional
`codemap` extra; these tests would otherwise spuriously fail because the
parsers return empty results.

The skip is applied through an autouse fixture rather than a
``pytest_collection_modifyitems`` hook. A collection hook defined in *any*
loaded conftest receives the whole session's item list, not just the items
below that conftest's directory, so the previous implementation marked every
test in the repository as skipped as soon as ``tests/unit/codemap`` was part
of the collection. Fixture visibility, by contrast, is scoped to this
directory by pytest itself, so the skip cannot leak.

Collection is unaffected, so the mandatory unit-test contract (these tests
must still be discoverable by ``--collect-only``) is preserved.

Regression guard: ``tests/architecture/test_conftest_hook_scoping.py``.
"""

from __future__ import annotations

import pytest

_OPTIONAL_DEPENDENCIES = (
    "tree_sitter",
    "tree_sitter_python",
    "tree_sitter_javascript",
    "tree_sitter_typescript",
    "tree_sitter_rust",
    "tree_sitter_markdown",
    "pydantic",
)


def _missing_deps() -> list[str]:
    missing: list[str] = []
    for name in _OPTIONAL_DEPENDENCIES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


_MISSING = _missing_deps()


@pytest.fixture(autouse=True)
def _require_codemap_dependencies() -> None:
    """Skip the calling codemap test when the optional parser stack is absent."""
    if _MISSING:
        pytest.skip(
            "codemap optional deps missing: "
            + ", ".join(_MISSING)
            + " (install with: pip install -e '.[codemap]')"
        )
