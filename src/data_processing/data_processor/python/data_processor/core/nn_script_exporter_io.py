"""IO helpers for neural network script export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .nn_architecture import NetworkConfig


def validate_script_output_path(output_path: Path | str) -> Path:
    """Validate script output path preconditions."""
    output = Path(output_path)
    if not str(output).strip() or str(output) == ".":
        raise ValueError("output_path must not be empty")
    if output.suffix.lower() != ".py":
        raise ValueError("output_path must end with .py")
    return output


def validate_config_output_path(output_path: Path | str) -> Path:
    """Validate config output path preconditions."""
    output = Path(output_path)
    if not str(output).strip() or str(output) == ".":
        raise ValueError("output_path must not be empty")
    return output


def validate_data_path(data_path: str | None) -> str | None:
    """Validate optional data path argument."""
    if data_path is None:
        return None
    if not data_path.strip():
        raise ValueError("data_path must not be empty")
    return data_path


def export_config_file(
    config: NetworkConfig,
    output_path: Path,
    normalization_params: dict[str, Any],
) -> Path:
    """Serialize a network config plus normalization metadata to JSON."""
    config_dict = config.to_dict()
    config_dict["normalization_params"] = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in normalization_params.items()
    }
    output_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    return output_path


def import_config_file(config_path: Path | str) -> tuple[NetworkConfig, dict[str, Any]]:
    """Load a network config plus normalization metadata from JSON."""
    path = Path(config_path)
    if not path.exists():
        msg = f"Configuration file does not exist: {path}"
        raise FileNotFoundError(msg)

    data = json.loads(path.read_text(encoding="utf-8"))
    normalization_params = {
        key: np.array(value) if isinstance(value, list) else value
        for key, value in data.pop("normalization_params", {}).items()
    }
    return NetworkConfig.from_dict(data), normalization_params
