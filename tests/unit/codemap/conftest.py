"""Skip every codemap unit test when tree-sitter (or any per-language pkg)
isn't installed. The CI test lane runs on a stock Python that doesn't
install the optional `codemap` extra; tests would otherwise spuriously
fail because the parsers return empty results.

The mandatory unit-test contract is satisfied by collection; the tests
just no-op when the parser stack isn't available.
"""

from __future__ import annotations

import pytest


def _missing_deps() -> list[str]:
    missing: list[str] = []
    for name in (
        "tree_sitter",
        "tree_sitter_python",
        "tree_sitter_javascript",
        "tree_sitter_typescript",
        "tree_sitter_rust",
        "tree_sitter_markdown",
        "pydantic",
    ):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


_MISSING = _missing_deps()


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    if not _MISSING:
        return
    skip_marker = pytest.mark.skip(
        reason=(
            "codemap optional deps missing: "
            + ", ".join(_MISSING)
            + " (install with: pip install -e '.[codemap]')"
        )
    )
    for item in items:
        item.add_marker(skip_marker)
