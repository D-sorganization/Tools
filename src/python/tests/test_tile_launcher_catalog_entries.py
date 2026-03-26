"""Regression tests for science launcher discoverability."""

from __future__ import annotations

from pathlib import Path

from tile_launcher.manager import load_catalog


def test_science_apps_are_listed_in_tile_launcher() -> None:
    """Solar system and RRT tools should be easy to launch from the app catalog."""
    catalog_path = (
        Path(__file__).resolve().parents[1] / "src" / "tile_launcher" / "app_catalog.json"
    )

    catalog = load_catalog(catalog_path)
    app_ids = {app.id for app in catalog}

    assert "solar_system_model" in app_ids
    assert "rrt_asteroid_navigator" in app_ids
