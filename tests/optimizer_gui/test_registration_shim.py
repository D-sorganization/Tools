"""Contract tests for the legacy optimizer_gui registration shim.

The optimizer_gui package no longer ships its own GUI: it is a thin shim that
redirects to the canonical movement_optimizer application (#3983). These tests
pin that contract so the shim can never silently resurrect drifted physics or
diverge from the canonical registration.
"""

from __future__ import annotations

import importlib

import pytest


def test_shim_registration_resolves_canonical_app() -> None:
    """The shim registration must resolve to the canonical movement_optimizer GUI."""
    importlib.import_module("optimizer_gui")
    registration = importlib.import_module("optimizer_gui.gui_registration")

    assert registration.get_gui_info is not None


def test_shim_registers_catalog_hidden_canonical_app() -> None:
    """The shim must be catalog-hidden and resolve to the canonical app."""
    registration = importlib.import_module("optimizer_gui.gui_registration")

    info = registration.get_gui_info()
    assert info["catalog_visible"] is False
    assert info["tool_name"] == "movement_optimizer"
    assert info["pyqt6"] == {
        "module": "movement_optimizer.gui.main_window",
        "class": "MainWindow",
        "dependencies": ["PyQt6", "matplotlib", "numpy", "scipy"],
        "settings_app": "MovementOptimizer",
    }


def test_dead_vendored_models_are_not_importable() -> None:
    """The deleted drifted physics copy must stay deleted (#3983)."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("optimizer_gui.models.chain_model")


def test_dead_vendored_main_window_is_not_importable() -> None:
    """The deleted drifted PyQt main window must stay deleted (#3983)."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("optimizer_gui.ui.pyqt6.main_window")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
