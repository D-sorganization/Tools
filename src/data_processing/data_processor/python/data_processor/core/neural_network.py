# mypy: ignore-errors
"""Neural Network Training Interface with Script Export.

Provides a flexible interface for:
- Configuring neural network architectures
- Training models on data
- Evaluating model performance
- Exporting training scripts for different environments
- Importing pre-trained models for prediction

Designed to be framework-agnostic, generating scripts for
PyTorch, TensorFlow/Keras, or scikit-learn as needed.

Supports various use cases including gasification data modeling
and robotics learning applications.

This module re-exports all public symbols from the decomposed
sub-modules and provides the original NeuralNetworkInterface
class for backward compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .nn_architecture import (
    ActivationFunction,
    DataSplitConfig,
    Framework,
    LayerConfig,
    LossFunction,
    NetworkConfig,
    NetworkType,
    Optimizer,
    TrainingResult,
)
from .nn_script_exporter import NeuralNetworkScriptExporter
from .nn_trainer import NeuralNetworkTrainer

logger = logging.getLogger(__name__)


class NeuralNetworkInterface:
    """Interface for neural network training and export.

    Provides a unified API for configuring, training, and
    exporting neural networks without requiring specific
    framework installations.

    Delegates to NeuralNetworkTrainer for training and
    NeuralNetworkScriptExporter for script generation.
    """

    def __init__(self) -> None:
        """Initialize the neural network interface."""
        self._trainer = NeuralNetworkTrainer()
        self._exporter = NeuralNetworkScriptExporter()

    @property
    def _config(self) -> NetworkConfig | None:
        """Get the current configuration."""
        return self._trainer.config

    @_config.setter
    def _config(self, value: NetworkConfig | None) -> None:
        """Set the current configuration."""
        self._trainer.config = value

    @property
    def _normalization_params(self) -> dict[str, Any]:
        """Get normalization parameters."""
        return self._trainer.normalization_params

    def create_config(
        self,
        input_features: int,
        output_features: int = 1,
        network_type: NetworkType = NetworkType.MLP,
        hidden_layers: list[int] | None = None,
        activation: ActivationFunction = ActivationFunction.RELU,
        dropout_rate: float = 0.2,
        **kwargs: Any,
    ) -> NetworkConfig:
        """Create a network configuration with sensible defaults."""
        return self._trainer.create_config(
            input_features=input_features,
            output_features=output_features,
            network_type=network_type,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            **kwargs,
        )

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        feature_columns: list[str] | None = None,
        split_config: DataSplitConfig | None = None,
    ) -> dict[str, Any]:
        """Prepare data for training."""
        return self._trainer.prepare_data(
            df=df,
            target_columns=target_columns,
            feature_columns=feature_columns,
            split_config=split_config,
        )

    def train_simple(
        self,
        data: dict[str, np.ndarray],
        config: NetworkConfig | None = None,
    ) -> TrainingResult:
        """Train a simple neural network using NumPy."""
        return self._trainer.train_simple(data=data, config=config)

    def export_script(
        self,
        config: NetworkConfig,
        output_path: Path | str,
        framework: Framework = Framework.PYTORCH,
        data_path: str | None = None,
        include_data_loading: bool = True,
        include_training: bool = True,
        include_evaluation: bool = True,
    ) -> Path:
        """Export training script for a specific framework."""
        assert config is not None, "config must be provided"
        self._exporter.normalization_params = self._trainer.normalization_params
        return self._exporter.export_script(
            config=config,
            output_path=output_path,
            framework=framework,
            data_path=data_path,
            include_data_loading=include_data_loading,
            include_training=include_training,
            include_evaluation=include_evaluation,
        )

    def export_config(
        self,
        config: NetworkConfig,
        output_path: Path | str,
    ) -> Path:
        """Export network configuration to JSON."""
        assert config is not None, "config must be provided"
        self._exporter.normalization_params = self._trainer.normalization_params
        return self._exporter.export_config(config=config, output_path=output_path)

    def import_config(self, config_path: Path | str) -> NetworkConfig:
        """Import network configuration from JSON."""
        assert config_path is not None, "config_path must be provided"
        config = self._exporter.import_config(config_path)
        self._trainer.config = config
        self._trainer._normalization_params = self._exporter.normalization_params
        return config


__all__ = [
    "ActivationFunction",
    "DataSplitConfig",
    "Framework",
    "LayerConfig",
    "LossFunction",
    "NetworkConfig",
    "NetworkType",
    "NeuralNetworkInterface",
    "NeuralNetworkScriptExporter",
    "NeuralNetworkTrainer",
    "Optimizer",
    "TrainingResult",
]
