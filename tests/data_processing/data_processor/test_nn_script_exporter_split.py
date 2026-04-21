"""Regression tests for the nn_script_exporter decomposition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from data_processor.core.nn_architecture import (
    ActivationFunction,
    Framework,
    LayerConfig,
    NetworkConfig,
    NetworkType,
)
from data_processor.core.nn_script_exporter import NeuralNetworkScriptExporter


def test_nn_script_exporter_facade_uses_extracted_modules(repo_root: Path) -> None:
    exporter_path = (
        repo_root
        / "src"
        / "data_processing"
        / "data_processor"
        / "python"
        / "data_processor"
        / "core"
        / "nn_script_exporter.py"
    )
    content = exporter_path.read_text(encoding="utf-8")

    assert "# mypy: ignore-errors" not in content
    assert "nn_script_exporter_io" in content
    assert "nn_script_exporter_renderers" in content


def _sample_config() -> NetworkConfig:
    return NetworkConfig(
        network_type=NetworkType.MLP,
        layers=[
            LayerConfig(layer_type="dense", units=32),
            LayerConfig(layer_type="dropout", dropout_rate=0.25),
            LayerConfig(
                layer_type="dense",
                units=1,
                activation=ActivationFunction.LINEAR,
            ),
        ],
        input_features=4,
        output_features=1,
        epochs=10,
    )


def test_export_script_still_generates_framework_specific_content(
    tmp_path: Path,
) -> None:
    exporter = NeuralNetworkScriptExporter()

    pytorch_path = tmp_path / "pytorch_train.py"
    tensorflow_path = tmp_path / "tensorflow_train.py"

    exporter.export_script(_sample_config(), pytorch_path, framework=Framework.PYTORCH)
    exporter.export_script(
        _sample_config(),
        tensorflow_path,
        framework=Framework.TENSORFLOW,
    )

    assert "import torch" in pytorch_path.read_text(encoding="utf-8")
    assert "keras.Sequential" in tensorflow_path.read_text(encoding="utf-8")


def test_import_and_export_config_round_trip_normalization_params(
    tmp_path: Path,
) -> None:
    exporter = NeuralNetworkScriptExporter()
    exporter.normalization_params = {"means": np.array([1.0, 2.0])}
    config_path = tmp_path / "network.json"

    exporter.export_config(_sample_config(), config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["normalization_params"]["means"] == [1.0, 2.0]

    restored = exporter.import_config(config_path)
    assert restored.input_features == 4
    assert exporter.normalization_params["means"].tolist() == [1.0, 2.0]
