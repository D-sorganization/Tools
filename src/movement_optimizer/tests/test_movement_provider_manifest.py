# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Regression tests for the shared Movement Optimizer provider manifest."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]  # Python 3.10 fallback

from scripts.movement_provider_manifest import (
    MOVEMENT_PROVIDER_MANIFEST,
    REPO_ROOT,
    validate_movement_provider_manifest,
)


def test_movement_provider_manifest_validates_against_repo_layout() -> None:
    """The published optimizer pack should resolve cleanly from the repo root."""
    manifest = validate_movement_provider_manifest()

    assert manifest["pack_id"] == "movement-optimizer-utilities"
    assert manifest["provider"] == "movement_optimizer"
    assert len(manifest["models"]) == 1


def test_movement_provider_manifest_declares_shared_launcher_metadata() -> None:
    """The optimizer pack should expose launcher metadata for shared consumers."""
    manifest = validate_movement_provider_manifest()
    entry = manifest["models"][0]

    assert entry["capabilities"] == [
        "optimization",
        "biomechanics",
        "trajectory",
        "cli",
    ]
    assert entry["launcher"] == {
        "category": "tool",
        "logo": "assets/movement_optimizer_icon.svg",
        "status": "provider_ready",
        "web_route": "/tools/movement-optimizer",
    }


def test_movement_provider_manifest_points_at_console_entry_module() -> None:
    """The provider path should track the installed console entry module."""
    manifest = validate_movement_provider_manifest()
    entry = manifest["models"][0]
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["movement-optimizer"] == (
        "movement_optimizer.__main__:main"
    )
    assert entry["path"] == "src/movement_optimizer/__main__.py"


def test_movement_provider_manifest_stays_in_expected_location() -> None:
    """The shared optimizer manifest should remain a top-level provider artifact."""
    assert MOVEMENT_PROVIDER_MANIFEST == REPO_ROOT / "model_pack.yaml"
    assert Path(MOVEMENT_PROVIDER_MANIFEST).is_file()
