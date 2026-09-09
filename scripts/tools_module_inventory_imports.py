"""Conservative Python import signals; parsing never imports or executes code."""

from __future__ import annotations

import ast
from pathlib import Path

_SCIENTIFIC_ROOTS = frozenset(
    {"numpy", "scipy", "math", "nalgebra", "ndarray", "statistics", "sympy", "casadi"}
)


def python_scientific_import(path: Path, text: str) -> bool:
    """Detect absolute scientific imports, including nested/conditional imports.

    Names are case-sensitive; comments, strings and relative imports supply no
    signal. Dynamic imports and arbitrary calculations without these libraries
    require other inventory signals or review. A parse failure raises rather
    than silently classifying unknown source as non-calculation.
    """
    try:
        tree = ast.parse(text, filename=path.as_posix())
    except SyntaxError as error:
        raise ValueError(f"Python import classification failed for {path}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.partition(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots = {node.module.partition(".")[0]}
        else:
            continue
        if roots & _SCIENTIFIC_ROOTS:
            return True
    return False
