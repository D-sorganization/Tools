"""Regression tests for the shared pendulum provider manifest."""

from __future__ import annotations

import tomllib
from pathlib import Path

from scripts.pendulum_provider_manifest import (
    PENDULUM_PROVIDER_MANIFEST,
    REPO_ROOT,
    validate_pendulum_provider_manifest,
)


def test_pendulum_provider_manifest_validates_against_repo_layout() -> None:
    """The published pendulum pack should resolve cleanly from the repo root."""
    manifest = validate_pendulum_provider_manifest()

    assert manifest["pack_id"] == "tools-pendulum-simulator"
    assert manifest["provider"] == "tools"
    assert len(manifest["models"]) == 1


def test_pendulum_provider_manifest_declares_shared_launcher_metadata() -> None:
    """The pendulum pack should expose shared-launch metadata for UpstreamDrift."""
    manifest = validate_pendulum_provider_manifest()
    entry = manifest["models"][0]

    assert entry["capabilities"] == [
        "pendulum",
        "simulation",
        "optimization",
        "biomechanics",
        "proximal-distal-companion",
    ]
    assert entry["launcher"] == {
        "category": "tool",
        "logo": "src/double_pendulum_golf/resources/pendulum_icon.png",
        "status": "provider_ready",
        "web_route": "/tools/pendulum-simulator",
    }


def test_pendulum_provider_manifest_points_at_console_entry_module() -> None:
    """The provider path should track the installed console entry module."""
    manifest = validate_pendulum_provider_manifest()
    entry = manifest["models"][0]
    pyproject_path = REPO_ROOT / "src" / "pendulum_simulator" / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["pendulum-golf"] == (
        "double_pendulum_golf.__main__:main"
    )
    assert entry["path"] == "src/double_pendulum_golf/__main__.py"
    assert entry["embed_adapter"] == (
        "src/double_pendulum_golf/__main__.py::get_dockable_ui"
    )


def test_pendulum_provider_manifest_stays_in_expected_location() -> None:
    """The shared pendulum manifest should live alongside the pendulum source tree."""
    assert PENDULUM_PROVIDER_MANIFEST == (
        REPO_ROOT / "src" / "pendulum_simulator" / "model_pack.yaml"
    )
    assert Path(PENDULUM_PROVIDER_MANIFEST).is_file()
