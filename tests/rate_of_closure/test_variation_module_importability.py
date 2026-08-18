"""Every `rate_of_closure.variation` module must actually import.

`regional_ground_study_adapter` shipped unreachable: it imported
`to_ground_model_result` from `shared.python.swing_sim.ground`, which had
stopped re-exporting that name one PR earlier because the ground package's own
contract test requires the unqualified compatibility adapter to stay private.
Nothing failed, because no test imported the adapter module — a suite can be
entirely green while a module in it cannot be loaded at all.

Importing each module is the cheap check that catches that whole class: a
module removed from a package's `__all__`, a renamed symbol, a circular import
introduced by a new dependency.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_VARIATION_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "rate_of_closure" / "variation"
)


def _module_names() -> list[str]:
    return sorted(
        f"rate_of_closure.variation.{path.stem}"
        for path in _VARIATION_DIR.glob("*.py")
        if path.stem != "__init__"
    )


def test_variation_package_exposes_modules_to_import() -> None:
    """Guard the guard: an empty sweep would pass without proving anything."""
    assert _module_names(), f"no modules discovered under {_VARIATION_DIR}"


@pytest.mark.parametrize("module_name", _module_names())
def test_variation_module_imports(module_name: str) -> None:
    """Each module must load; an unreachable module is a broken module."""
    importlib.import_module(module_name)
