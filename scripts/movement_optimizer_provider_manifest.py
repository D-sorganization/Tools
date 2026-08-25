"""Validation helpers for the shared Movement Optimizer provider manifest.

The Movement Optimizer biomechanics app is vendored under
``src/movement_optimizer`` (migrated from the standalone
``D-sorganization/Movement_Optimizer`` repository, which is archived in favour
of this canonical Tools-resident copy — see Tools#3407).  This module loads and
validates the published ``model_pack.yaml`` so the UpstreamDrift launcher can
discover the tool from the Tools provider surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MOVEMENT_OPTIMIZER_PROVIDER_MANIFEST = (
    REPO_ROOT / "src" / "movement_optimizer" / "model_pack.yaml"
)

_REQUIRED_TOP_LEVEL_FIELDS = (
    "manifest_version",
    "force_attribution_schema",
    "pack_id",
    "pack_name",
    "provider",
    "models",
)
_REQUIRED_MODEL_FIELDS = (
    "id",
    "name",
    "description",
    "type",
    "path",
    "source_root",
    "working_dir",
    "python_paths",
    "capabilities",
    "supported_exercises",
    "launcher",
)
_REQUIRED_LAUNCHER_FIELDS = ("category", "logo", "status")
_REQUIRED_CAPABILITIES = {
    "optimization",
    "biomechanics",
    "trajectory",
    "cli",
    "pyqt6",
    "swingset",
    "chain_dynamics",
    "coordinate_force_attribution",
    "component_impulse_optimization",
}


def _expected_exercises() -> list[str]:
    """Return the exercises exported by the canonical tool-pack surface."""
    from movement_optimizer.tool_pack import list_exercises

    return list(list_exercises())


def load_movement_optimizer_provider_manifest(
    path: Path = MOVEMENT_OPTIMIZER_PROVIDER_MANIFEST,
) -> dict[str, Any]:
    """Load the Movement Optimizer provider manifest from disk."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("movement optimizer provider manifest must be a mapping")
    return data


def _require_fields(
    payload: dict[str, Any], required_fields: tuple[str, ...], *, label: str
) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")


def _resolve_repo_relative_path(
    repo_root: Path, relative_path: str, *, label: str
) -> Path:
    path = (repo_root / relative_path).resolve(strict=False)
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def validate_movement_optimizer_provider_manifest(
    repo_root: Path = REPO_ROOT,
    path: Path = MOVEMENT_OPTIMIZER_PROVIDER_MANIFEST,
) -> dict[str, Any]:
    """Validate the Movement Optimizer provider manifest against the repo layout."""
    manifest = load_movement_optimizer_provider_manifest(path)
    _require_fields(manifest, _REQUIRED_TOP_LEVEL_FIELDS, label="manifest")

    models = manifest["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("manifest must contain at least one model entry")

    model_ids: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("model entries must be mappings")
        _require_fields(
            model, _REQUIRED_MODEL_FIELDS, label=f"model[{model.get('id', '?')}]"
        )

        model_id = model["id"]
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model id must be a non-empty string")
        if model_id in model_ids:
            raise ValueError(f"duplicate model id: {model_id}")
        model_ids.add(model_id)

        source_root = _resolve_repo_relative_path(
            repo_root,
            str(model["source_root"]),
            label=f"{model_id}.source_root",
        )
        artifact_path = (source_root / str(model["path"])).resolve(strict=False)
        if not artifact_path.exists():
            raise ValueError(f"{model_id}.path does not exist: {artifact_path}")

        _resolve_repo_relative_path(
            repo_root,
            str(model["working_dir"]),
            label=f"{model_id}.working_dir",
        )

        python_paths = model["python_paths"]
        if not isinstance(python_paths, list) or not python_paths:
            raise ValueError(f"{model_id}.python_paths must be a non-empty list")
        for index, python_path in enumerate(python_paths):
            _resolve_repo_relative_path(
                repo_root,
                str(python_path),
                label=f"{model_id}.python_paths[{index}]",
            )

        capabilities = model["capabilities"]
        if not isinstance(capabilities, list) or not capabilities:
            raise ValueError(f"{model_id}.capabilities must be a non-empty list")
        missing_capabilities = _REQUIRED_CAPABILITIES.difference(capabilities)
        if missing_capabilities:
            missing = sorted(missing_capabilities)
            raise ValueError(
                f"{model_id}.capabilities missing required items: {missing}"
            )

        supported_exercises = model["supported_exercises"]
        if supported_exercises != _expected_exercises():
            raise ValueError(
                f"{model_id}.supported_exercises must match movement_optimizer.tool_pack"
            )

        launcher = model["launcher"]
        if not isinstance(launcher, dict):
            raise ValueError(f"{model_id}.launcher must be a mapping")
        _require_fields(
            launcher, _REQUIRED_LAUNCHER_FIELDS, label=f"{model_id}.launcher"
        )
        if launcher["category"] != "tool":
            raise ValueError(f"{model_id}.launcher.category must be 'tool'")

        logo_path = (source_root / str(launcher["logo"])).resolve(strict=False)
        if not logo_path.exists():
            raise ValueError(f"{model_id}.launcher.logo does not exist: {logo_path}")

    return manifest
