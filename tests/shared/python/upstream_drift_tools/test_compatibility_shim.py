"""Provider-contract tests for the legacy upstream_drift_tools shim."""

from __future__ import annotations

import importlib
import sys
import warnings


def test_legacy_package_reexports_sidekick_contracts() -> None:
    """Legacy imports keep resolving to the canonical sidekick API."""
    sys.modules.pop("upstream_drift_tools", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = importlib.import_module("upstream_drift_tools")

    sidekick = importlib.import_module("sidekick")

    assert legacy.__version__ == sidekick.__version__
    assert legacy.Calculator is sidekick.Calculator
    assert legacy.ValidationResult is sidekick.ValidationResult
    assert any(item.category is DeprecationWarning for item in caught)


def test_legacy_submodule_aliases_canonical_sidekick_modules() -> None:
    """Submodule aliases point at the same module objects as sidekick."""
    legacy_theme = importlib.import_module("upstream_drift_tools.theme")
    sidekick_theme = importlib.import_module("sidekick.theme")

    assert legacy_theme is sidekick_theme
