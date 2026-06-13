"""Regression tests for the shared Movement Optimizer provider manifest."""

from __future__ import annotations

from scripts.movement_optimizer_provider_manifest import (
    validate_movement_optimizer_provider_manifest,
)


def test_manifest_validates_against_repo_layout() -> None:
    """The published Movement Optimizer pack resolves cleanly from the repo root."""
    manifest = validate_movement_optimizer_provider_manifest()

    assert manifest["pack_id"] == "tools-movement-optimizer-biomech"
    assert manifest["provider"] == "tools"
    assert len(manifest["models"]) == 1


def test_manifest_declares_shared_launcher_metadata() -> None:
    """The pack exposes shared-launch metadata for UpstreamDrift discovery."""
    manifest = validate_movement_optimizer_provider_manifest()
    entry = manifest["models"][0]

    assert entry["id"] == "movement_optimizer"
    assert entry["capabilities"] == [
        "optimization",
        "biomechanics",
        "trajectory",
        "cli",
    ]
    assert entry["launcher"]["category"] == "tool"
    assert entry["launcher"]["status"] == "provider_ready"
    assert entry["launcher"]["web_route"] == "/tools/movement-optimizer-biomech"


def test_manifest_points_at_vendored_entry_module() -> None:
    """The provider path tracks the vendored package ``__main__`` entry point."""
    manifest = validate_movement_optimizer_provider_manifest()
    entry = manifest["models"][0]

    assert entry["source_root"] == "src/movement_optimizer"
    assert entry["path"] == "__main__.py"
