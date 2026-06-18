"""Regression tests for the shared Movement Optimizer provider manifest."""

from __future__ import annotations

from scripts.movement_optimizer_provider_manifest import (
    validate_movement_optimizer_provider_manifest,
)


def test_manifest_validates_against_repo_layout() -> None:
    """The published Movement Optimizer pack resolves cleanly from the repo root."""
    manifest = validate_movement_optimizer_provider_manifest()

    assert manifest["pack_id"] == "tools-movement-optimizer"
    assert manifest["provider"] == "tools"
    assert len(manifest["models"]) == 1


def test_manifest_declares_shared_launcher_metadata() -> None:
    """The pack exposes shared-launch metadata for UpstreamDrift discovery."""
    manifest = validate_movement_optimizer_provider_manifest()
    entry = manifest["models"][0]

    assert entry["id"] == "tools_movement_optimizer"
    assert entry["capabilities"] == [
        "optimization",
        "biomechanics",
        "trajectory",
        "cli",
        "pyqt6",
        "swingset",
        "chain_dynamics",
    ]
    assert entry["supported_exercises"] == [
        "squat",
        "full_squat",
        "deadlift",
        "bench_press",
        "snatch",
        "clean",
        "jerk",
    ]
    assert entry["launcher"]["category"] == "tool"
    assert entry["launcher"]["status"] == "provider_ready"
    assert entry["launcher"]["web_route"] == "/tools/movement-optimizer"


def test_manifest_points_at_vendored_entry_module() -> None:
    """The provider path tracks the canonical PyQt launcher entry point."""
    manifest = validate_movement_optimizer_provider_manifest()
    entry = manifest["models"][0]

    assert entry["source_root"] == "src/movement_optimizer"
    assert entry["path"] == "launch_pyqt6.py"


def test_gui_registration_names_canonical_app() -> None:
    """The launcher-backed directory exposes one manifest registration."""
    from movement_optimizer.gui_registration import get_gui_info

    info = get_gui_info()

    assert info["tool_name"] == "movement_optimizer"
    assert info["pyqt6"]["module"] == "movement_optimizer.gui.main_window"


def test_legacy_optimizer_gui_registration_points_to_canonical_app() -> None:
    """Old Tools launch paths remain a shim over the migrated implementation."""
    from optimizer_gui.gui_registration import get_gui_info

    info = get_gui_info()

    assert info["catalog_visible"] is False
    assert info["tool_name"] == "movement_optimizer"
    assert info["pyqt6"]["module"] == "movement_optimizer.gui.main_window"
    assert info["pyqt6"]["class"] == "MainWindow"
