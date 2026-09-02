"""Regression tests for tile launcher catalog discoverability.

The tile launcher must read its catalog from the repository's single
canonical registry (``tools.json``) rather than a hand-maintained duplicate.
A stale duplicate (``app_catalog.json``) previously drifted until 5 of its
11 entries pointed at files that no longer existed (#3982).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tile_launcher.manager import AppManager, load_tools_registry

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture()
def isolated_layout_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep `AppManager.from_default_paths` from touching the real user home."""
    layout_path = tmp_path / "layout.json"
    monkeypatch.setattr("tile_launcher.manager.DEFAULT_LAYOUT_PATH", layout_path)
    return layout_path


def test_duplicate_app_catalog_json_was_removed() -> None:
    """The stale, hand-maintained registry copy must not come back."""
    stale_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tile_launcher"
        / "app_catalog.json"
    )
    assert not stale_path.exists()


def test_tile_launcher_reads_the_canonical_tools_json_registry(
    isolated_layout_store: Path,
) -> None:
    """`AppManager.from_default_paths` must load `tools.json`, not a copy.

    The layout (which apps are pinned, and in what order) is a
    per-machine user preference, so this compares catalog *membership*
    rather than the persisted tile order.
    """
    manager = AppManager.from_default_paths()

    registry = load_tools_registry(REPO_ROOT / "tools.json")
    loaded_apps = manager.apps_in_layout() + manager.available_to_add()
    assert {app.id for app in loaded_apps} == {app.id for app in registry}
    assert len(loaded_apps) == len(registry)


def test_every_catalog_entry_resolves_to_a_file_that_exists(
    isolated_layout_store: Path,
) -> None:
    """Every tile in the canonical registry must point at a real file.

    This is the direct regression check for #3982: the drifted duplicate
    catalog had 5 of 11 tiles pointing at files that had moved or been
    deleted.
    """
    manager = AppManager.from_default_paths()
    catalog = manager.apps_in_layout()
    assert catalog, "expected the tools.json registry to contain apps"

    broken = [
        app.id
        for app in catalog
        if not app.resolved_path(manager.repository_root).exists()
    ]
    assert broken == [], f"tile launcher entries point at missing files: {broken}"
