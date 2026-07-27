"""Focused public-API stability gate for canonical standalone Sidekick."""

from __future__ import annotations

import json

import pytest

from tests import test_sidekick_public_api_stability as api_stability

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_CANONICAL_MODULES = (
    "__main__.py",
    "persistence/__init__.py",
    "persistence/schema.py",
    "persistence/state_profile.py",
    "standalone/__init__.py",
    "standalone/onboarding.py",
    "standalone/preferences.py",
    "standalone/runner.py",
    "standalone/session_store.py",
    "standalone/window.py",
)


@pytest.mark.parametrize("relative_path", _CANONICAL_MODULES)
def test_canonical_standalone_public_api_matches_baseline(
    relative_path: str,
) -> None:
    """Each newly canonical module must match its reviewed API baseline."""
    baseline = json.loads(api_stability.BASELINE_PATH.read_text(encoding="utf-8"))
    module_path = api_stability.SIDEKICK_ROOT / relative_path

    assert relative_path in baseline
    assert api_stability.extract_module_api(module_path) == baseline[relative_path]
