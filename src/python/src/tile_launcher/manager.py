"""Catalog loading and layout management for the tile launcher."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from tile_launcher.models import AppDefinition, LaunchType, LayoutStore

from . import DEFAULT_LAYOUT_PATH

logger = logging.getLogger(__name__)

#: Canonical launcher registry, shared with every other Tools entry point.
#: This is the single source of truth for launchable apps -- do not
#: reintroduce a hand-maintained duplicate (see #3982).
TOOLS_REGISTRY_FILENAME = "tools.json"

_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    """Derive a stable, catalog-safe id from a human-readable app name."""

    slug = _SLUG_NON_ALNUM.sub("_", name.strip().lower()).strip("_")
    return slug or "app"


def _finalize_catalog(entries: list[AppDefinition]) -> list[AppDefinition]:
    """Validate a catalog's id uniqueness before handing it back."""

    ids = [app.id for app in entries]
    if len(set(ids)) != len(ids):
        raise AppCatalogError("Catalog contains duplicate app ids")

    return entries


class AppCatalogError(Exception):
    """Raised when the catalog cannot be loaded correctly."""


def load_catalog(catalog_path: Path) -> list[AppDefinition]:
    """Load the application catalog from a flat JSON list.

    Each entry must provide ``id``, ``name``, ``relative_path`` and
    ``launch_type``; ``logo`` and ``description`` are optional.
    """

    if not catalog_path.exists():
        message = f"Catalog file not found: {catalog_path}"
        logger.error(message)
        raise AppCatalogError(message)

    catalog_data = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog: list[AppDefinition] = []
    for entry in catalog_data:
        app = AppDefinition(
            id=entry["id"],
            name=entry["name"],
            relative_path=entry["relative_path"],
            launch_type=LaunchType(entry["launch_type"]),
            logo=entry.get("logo"),
            description=entry.get("description"),
        )
        catalog.append(app)

    return _finalize_catalog(catalog)


def load_tools_registry(registry_path: Path) -> list[AppDefinition]:
    """Load the canonical launcher catalog from the root ``tools.json``.

    ``tools.json`` groups apps by category::

        {"Category": [{"name": ..., "path": ..., "type": ..., "desc": ...}]}

    This is the single registry every launch surface in the repository
    (the CLI, the unified launcher window, and this tile launcher) should
    read from -- see #3982.
    """

    if not registry_path.exists():
        message = f"Tools registry file not found: {registry_path}"
        logger.error(message)
        raise AppCatalogError(message)

    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
    catalog: list[AppDefinition] = []
    seen_ids: set[str] = set()
    for entries in registry_data.values():
        for entry in entries:
            app_id = _slugify(entry["name"])
            suffix = 2
            unique_id = app_id
            while unique_id in seen_ids:
                unique_id = f"{app_id}_{suffix}"
                suffix += 1
            seen_ids.add(unique_id)

            catalog.append(
                AppDefinition(
                    id=unique_id,
                    name=entry["name"],
                    relative_path=entry["path"],
                    launch_type=LaunchType(entry["type"]),
                    logo=entry.get("logo"),
                    description=entry.get("desc"),
                )
            )

    return _finalize_catalog(catalog)


class AppManager:
    """Manage available apps and the current tile layout."""

    def __init__(
        self,
        catalog: Sequence[AppDefinition],
        repository_root: Path,
        layout_store: LayoutStore | None = None,
    ) -> None:
        """Initialize the manager with a catalog and layout store."""
        if catalog is None:
            raise ValueError("catalog must be provided")
        self._catalog = {app.id: app for app in catalog}
        self.repository_root = repository_root
        self.layout_store = layout_store
        self.layout: list[str] = []
        self._load_layout()

    @classmethod
    def from_default_paths(cls) -> AppManager:
        """Create an AppManager using default paths for catalog and layout."""
        # manager.py lives at <repo_root>/src/python/src/tile_launcher/manager.py
        base_path = Path(__file__).resolve().parents[4]
        registry_path = base_path / TOOLS_REGISTRY_FILENAME
        catalog = load_tools_registry(registry_path)
        layout_store = cls._default_store()
        return cls(
            catalog=catalog, repository_root=base_path, layout_store=layout_store
        )

    @staticmethod
    def _default_store() -> LayoutStore:
        """Return the default layout store strategy."""
        from tile_launcher.models import FileLayoutStore

        return FileLayoutStore(path=DEFAULT_LAYOUT_PATH)

    def _load_layout(self) -> None:
        """Load the layout from the store or initialize default."""
        saved_layout = self.layout_store.load() if self.layout_store else []
        seen: set[str] = set()
        self.layout = []
        for app_id in saved_layout:
            if app_id in self._catalog and app_id not in seen:
                self.layout.append(app_id)
                seen.add(app_id)
        if not self.layout:
            self.layout = list(self._catalog.keys())
            self._save_layout()

    def _save_layout(self) -> None:
        """Save the current layout to the store."""
        if self.layout_store:
            self.layout_store.save(self.layout)

    def apps_in_layout(self) -> list[AppDefinition]:
        """Return the list of apps currently in the layout."""
        return [
            self._catalog[app_id] for app_id in self.layout if app_id in self._catalog
        ]

    def available_to_add(self) -> list[AppDefinition]:
        """Return list of apps available to be added to the layout."""
        available = [
            app for app_id, app in self._catalog.items() if app_id not in self.layout
        ]
        return sorted(available, key=lambda app: app.name.lower())

    def add_app(self, app_id: str) -> None:
        """Add an app to the layout."""
        if app_id not in self._catalog:
            raise KeyError(f"App id '{app_id}' is not present in the catalog")

        if app_id in self.layout:
            logger.info("App %s already in layout; skipping add", app_id)
            return

        self.layout.append(app_id)
        self._save_layout()

    def remove_app(self, app_id: str) -> None:
        """Remove an app from the layout."""
        if app_id is None:
            raise ValueError("app_id must be provided")
        if app_id not in self.layout:
            logger.info("App %s not in layout; skipping remove", app_id)
            return

        self.layout = [existing for existing in self.layout if existing != app_id]
        self._save_layout()

    def reorder(self, new_order: Iterable[str]) -> None:
        """Reorder the layout based on a new sequence of app IDs."""
        validated: list[str] = []
        for app_id in new_order:
            if app_id in self._catalog and app_id not in validated:
                validated.append(app_id)
        if not validated:
            raise ValueError("Cannot set an empty layout")

        self.layout = validated
        self._save_layout()

    def reset_layout(self) -> None:
        """Reset the layout to include all apps from the catalog."""
        self.layout = list(self._catalog.keys())
        self._save_layout()

    def get_app(self, app_id: str) -> AppDefinition:
        """Get the definition for a specific app."""
        return self._catalog[app_id]
