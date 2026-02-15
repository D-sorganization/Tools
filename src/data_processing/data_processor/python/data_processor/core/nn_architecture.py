# mypy: ignore-errors
"""Neural network architecture definitions.

Contains all enums, dataclasses, and type definitions for
configuring neural network architectures, training parameters,
and result storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Framework(Enum):
    """Supported ML frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"


class NetworkType(Enum):
    """Types of neural network architectures."""

    MLP = "mlp"  # Multi-Layer Perceptron
    LSTM = "lstm"  # Long Short-Term Memory
    GRU = "gru"  # Gated Recurrent Unit
    CNN_1D = "cnn_1d"  # 1D Convolutional
    TRANSFORMER = "transformer"
    AUTOENCODER = "autoencoder"


class ActivationFunction(Enum):
    """Available activation functions."""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    LINEAR = "linear"
    GELU = "gelu"
    SWISH = "swish"


class LossFunction(Enum):
    """Available loss functions."""

    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    CUSTOM = "custom"


class Optimizer(Enum):
    """Available optimizers."""

    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


@dataclass
class LayerConfig:
    """Configuration for a single network layer."""

    layer_type: str  # "dense", "lstm", "conv1d", "dropout", "batchnorm"
    units: int = 64
    activation: ActivationFunction = ActivationFunction.RELU
    dropout_rate: float = 0.0
    kernel_size: int = 3  # For conv layers
    return_sequences: bool = True  # For recurrent layers
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkConfig:
    """Complete neural network configuration."""

    # Architecture
    network_type: NetworkType = NetworkType.MLP
    layers: list[LayerConfig] = field(default_factory=list)
    input_features: int = 0
    output_features: int = 1

    # Training
    optimizer: Optimizer = Optimizer.ADAM
    learning_rate: float = 0.001
    loss_function: LossFunction = LossFunction.MSE
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2

    # Regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5

    # Data preprocessing
    normalize_inputs: bool = True
    normalize_outputs: bool = False
    sequence_length: int = 10  # For sequential data

    # Output
    task_type: str = "regression"  # "regression", "classification", "multi_output"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "network_type": self.network_type.value,
            "layers": [
                {
                    "layer_type": layer_cfg.layer_type,
                    "units": layer_cfg.units,
                    "activation": layer_cfg.activation.value,
                    "dropout_rate": layer_cfg.dropout_rate,
                    "kernel_size": layer_cfg.kernel_size,
                    "return_sequences": layer_cfg.return_sequences,
                    "parameters": layer_cfg.parameters,
                }
                for layer_cfg in self.layers
            ],
            "input_features": self.input_features,
            "output_features": self.output_features,
            "optimizer": self.optimizer.value,
            "learning_rate": self.learning_rate,
            "loss_function": self.loss_function.value,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "l1_regularization": self.l1_regularization,
            "l2_regularization": self.l2_regularization,
            "early_stopping_patience": self.early_stopping_patience,
            "reduce_lr_patience": self.reduce_lr_patience,
            "normalize_inputs": self.normalize_inputs,
            "normalize_outputs": self.normalize_outputs,
            "sequence_length": self.sequence_length,
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkConfig:
        """Create from dictionary."""
        layers = [
            LayerConfig(
                layer_type=layer_data["layer_type"],
                units=layer_data.get("units", 64),
                activation=ActivationFunction(layer_data.get("activation", "relu")),
                dropout_rate=layer_data.get("dropout_rate", 0.0),
                kernel_size=layer_data.get("kernel_size", 3),
                return_sequences=layer_data.get("return_sequences", True),
                parameters=layer_data.get("parameters", {}),
            )
            for layer_data in data.get("layers", [])
        ]

        return cls(
            network_type=NetworkType(data.get("network_type", "mlp")),
            layers=layers,
            input_features=data.get("input_features", 0),
            output_features=data.get("output_features", 1),
            optimizer=Optimizer(data.get("optimizer", "adam")),
            learning_rate=data.get("learning_rate", 0.001),
            loss_function=LossFunction(data.get("loss_function", "mse")),
            batch_size=data.get("batch_size", 32),
            epochs=data.get("epochs", 100),
            validation_split=data.get("validation_split", 0.2),
            l1_regularization=data.get("l1_regularization", 0.0),
            l2_regularization=data.get("l2_regularization", 0.0),
            early_stopping_patience=data.get("early_stopping_patience", 10),
            reduce_lr_patience=data.get("reduce_lr_patience", 5),
            normalize_inputs=data.get("normalize_inputs", True),
            normalize_outputs=data.get("normalize_outputs", False),
            sequence_length=data.get("sequence_length", 10),
            task_type=data.get("task_type", "regression"),
        )


@dataclass
class TrainingResult:
    """Results from neural network training."""

    # Training metrics
    train_loss_history: list[float]
    val_loss_history: list[float]
    best_epoch: int
    best_val_loss: float

    # Final metrics
    final_train_loss: float
    final_val_loss: float
    test_loss: float | None = None
    test_metrics: dict[str, float] = field(default_factory=dict)

    # Model info
    total_parameters: int = 0
    trainable_parameters: int = 0

    # Training info
    training_time_seconds: float = 0.0
    stopped_early: bool = False

    # Predictions (for evaluation)
    predictions: np.ndarray | None = None
    actual_values: np.ndarray | None = None


@dataclass
class DataSplitConfig:
    """Configuration for train/validation/test splits."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle: bool = True
    random_state: int | None = 42
    stratify_column: str | None = None  # For classification


__all__ = [
    "ActivationFunction",
    "DataSplitConfig",
    "Framework",
    "LayerConfig",
    "LossFunction",
    "NetworkConfig",
    "NetworkType",
    "Optimizer",
    "TrainingResult",
]
