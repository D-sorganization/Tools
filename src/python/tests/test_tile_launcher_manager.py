"""Unit tests for the tile launcher catalog and layout management."""

from __future__ import annotations

from pathlib import Path

import pytest
from tile_launcher.manager import AppCatalogError, AppManager, load_catalog
from tile_launcher.models import AppDefinition, InMemoryLayoutStore, LaunchType


@pytest.fixture()
def sample_catalog() -> list[AppDefinition]:
    """Provide a sample catalog for testing."""
    return [
        AppDefinition(
            id="alpha",
            name="Alpha",
            relative_path="alpha.py",
            launch_type=LaunchType.PYTHON,
        ),
        AppDefinition(
            id="beta",
            name="Beta",
            relative_path="beta.bat",
            launch_type=LaunchType.BAT,
        ),
        AppDefinition(
            id="gamma",
            name="Gamma",
            relative_path="gamma.html",
            launch_type=LaunchType.HTML,
        ),
    ]


def test_defaults_when_layout_missing(
    sample_catalog: list[AppDefinition], tmp_path: Path
) -> None:
    """Ensure that the default layout includes all apps if no layout is saved."""
    manager = AppManager(
        catalog=sample_catalog,
        repository_root=tmp_path,
        layout_store=InMemoryLayoutStore(),
    )

    assert [app.id for app in manager.apps_in_layout()] == ["alpha", "beta", "gamma"]


def test_add_remove_and_reorder(
    sample_catalog: list[AppDefinition], tmp_path: Path
) -> None:
    """Test adding, removing, and reordering apps in the layout."""
    store = InMemoryLayoutStore(["alpha", "beta"])
    manager = AppManager(
        catalog=sample_catalog, repository_root=tmp_path, layout_store=store
    )

    manager.add_app("gamma")
    assert [app.id for app in manager.apps_in_layout()] == ["alpha", "beta", "gamma"]

    manager.reorder(["gamma", "alpha", "beta"])
    assert [app.id for app in manager.apps_in_layout()] == ["gamma", "alpha", "beta"]

    manager.remove_app("alpha")
    assert [app.id for app in manager.apps_in_layout()] == ["gamma", "beta"]

    manager.reset_layout()
    assert [app.id for app in manager.apps_in_layout()] == ["alpha", "beta", "gamma"]


def test_available_to_add_sorted(
    sample_catalog: list[AppDefinition], tmp_path: Path
) -> None:
    """Ensure that apps not in the layout are available to add."""
    manager = AppManager(
        catalog=sample_catalog,
        repository_root=tmp_path,
        layout_store=InMemoryLayoutStore(["alpha"]),
    )

    available = manager.available_to_add()
    assert [app.id for app in available] == ["beta", "gamma"]


def test_load_catalog_rejects_duplicates(tmp_path: Path) -> None:
    """Ensure that duplicate app IDs in the catalog raise an error."""
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        """
        [
            {"id": "alpha", "name": "A", "relative_path": "a", "launch_type": "python"},
            {"id": "alpha", "name": "B", "relative_path": "b", "launch_type": "python"}
        ]
        """,
        encoding="utf-8",
    )

    with pytest.raises(AppCatalogError):
        load_catalog(catalog_path)
