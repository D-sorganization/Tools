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
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
                    "layer_type": l.layer_type,
                    "units": l.units,
                    "activation": l.activation.value,
                    "dropout_rate": l.dropout_rate,
                    "kernel_size": l.kernel_size,
                    "return_sequences": l.return_sequences,
                    "parameters": l.parameters,
                }
                for l in self.layers
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
                layer_type=l["layer_type"],
                units=l.get("units", 64),
                activation=ActivationFunction(l.get("activation", "relu")),
                dropout_rate=l.get("dropout_rate", 0.0),
                kernel_size=l.get("kernel_size", 3),
                return_sequences=l.get("return_sequences", True),
                parameters=l.get("parameters", {}),
            )
            for l in data.get("layers", [])
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


class NeuralNetworkInterface:
    """Interface for neural network training and export.

    Provides a unified API for configuring, training, and
    exporting neural networks without requiring specific
    framework installations.
    """

    def __init__(self) -> None:
        """Initialize the neural network interface."""
        self._config: NetworkConfig | None = None
        self._normalization_params: dict[str, Any] = {}

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
        """Create a network configuration with sensible defaults.

        Args:
            input_features: Number of input features
            output_features: Number of output features
            network_type: Type of network architecture
            hidden_layers: List of hidden layer sizes (e.g., [128, 64, 32])
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate between layers
            **kwargs: Additional configuration options

        Returns:
            Complete NetworkConfig
        """
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        # Build layers based on network type
        layers = []

        if network_type == NetworkType.MLP:
            for units in hidden_layers:
                layers.append(LayerConfig(
                    layer_type="dense",
                    units=units,
                    activation=activation,
                ))
                if dropout_rate > 0:
                    layers.append(LayerConfig(
                        layer_type="dropout",
                        dropout_rate=dropout_rate,
                    ))

        elif network_type in (NetworkType.LSTM, NetworkType.GRU):
            layer_type = "lstm" if network_type == NetworkType.LSTM else "gru"
            for i, units in enumerate(hidden_layers):
                is_last = i == len(hidden_layers) - 1
                layers.append(LayerConfig(
                    layer_type=layer_type,
                    units=units,
                    return_sequences=not is_last,
                ))
                if dropout_rate > 0:
                    layers.append(LayerConfig(
                        layer_type="dropout",
                        dropout_rate=dropout_rate,
                    ))

        elif network_type == NetworkType.CNN_1D:
            filters = [32, 64, 128]
            for f in filters:
                layers.append(LayerConfig(
                    layer_type="conv1d",
                    units=f,
                    kernel_size=3,
                    activation=activation,
                ))
            layers.append(LayerConfig(layer_type="flatten"))
            for units in hidden_layers:
                layers.append(LayerConfig(
                    layer_type="dense",
                    units=units,
                    activation=activation,
                ))

        # Output layer
        output_activation = (
            ActivationFunction.LINEAR if kwargs.get("task_type", "regression") == "regression"
            else ActivationFunction.SOFTMAX
        )
        layers.append(LayerConfig(
            layer_type="dense",
            units=output_features,
            activation=output_activation,
        ))

        config = NetworkConfig(
            network_type=network_type,
            layers=layers,
            input_features=input_features,
            output_features=output_features,
            **{k: v for k, v in kwargs.items() if hasattr(NetworkConfig, k)},
        )

        self._config = config
        return config

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        feature_columns: list[str] | None = None,
        split_config: DataSplitConfig | None = None,
    ) -> dict[str, np.ndarray]:
        """Prepare data for training.

        Args:
            df: DataFrame with features and targets
            target_columns: Names of target columns
            feature_columns: Names of feature columns (None = all except targets)
            split_config: Train/val/test split configuration

        Returns:
            Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
        """
        split_config = split_config or DataSplitConfig()

        # Select features
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c not in target_columns]

        # Extract data
        data = df[feature_columns + target_columns].dropna()
        X = data[feature_columns].values.astype(np.float32)
        y = data[target_columns].values.astype(np.float32)

        if y.shape[1] == 1:
            y = y.ravel()

        n = len(X)

        # Shuffle if requested
        if split_config.shuffle:
            rng = np.random.default_rng(split_config.random_state)
            indices = rng.permutation(n)
            X = X[indices]
            y = y[indices] if y.ndim == 1 else y[indices]

        # Split data
        n_train = int(n * split_config.train_ratio)
        n_val = int(n * split_config.val_ratio)

        X_train = X[:n_train]
        y_train = y[:n_train] if y.ndim == 1 else y[:n_train]

        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val] if y.ndim == 1 else y[n_train:n_train + n_val]

        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:] if y.ndim == 1 else y[n_train + n_val:]

        # Normalize
        if self._config and self._config.normalize_inputs:
            X_train, X_val, X_test = self._normalize_features(X_train, X_val, X_test)

        if self._config and self._config.normalize_outputs and y.ndim > 0:
            y_train, y_val, y_test = self._normalize_targets(y_train, y_val, y_test)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_columns,
            "target_names": target_columns,
        }

    def _normalize_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features using training statistics."""
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1

        self._normalization_params["X_mean"] = mean
        self._normalization_params["X_std"] = std

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        return X_train, X_val, X_test

    def _normalize_targets(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize targets using training statistics."""
        mean = np.mean(y_train, axis=0) if y_train.ndim > 1 else np.mean(y_train)
        std = np.std(y_train, axis=0) if y_train.ndim > 1 else np.std(y_train)
        if isinstance(std, np.ndarray):
            std[std == 0] = 1
        elif std == 0:
            std = 1

        self._normalization_params["y_mean"] = mean
        self._normalization_params["y_std"] = std

        y_train = (y_train - mean) / std
        y_val = (y_val - mean) / std
        y_test = (y_test - mean) / std

        return y_train, y_val, y_test

    def train_simple(
        self,
        data: dict[str, np.ndarray],
        config: NetworkConfig | None = None,
    ) -> TrainingResult:
        """Train a simple neural network using NumPy (no external frameworks).

        This provides basic training capability without requiring
        PyTorch or TensorFlow. For production use, export scripts
        to the appropriate framework.

        Args:
            data: Data dictionary from prepare_data
            config: Network configuration (uses stored config if None)

        Returns:
            Training results
        """
        config = config or self._config
        if not config:
            raise ValueError("No network configuration provided")

        # Simple MLP implementation with NumPy
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]

        # Initialize weights
        weights, biases = self._initialize_weights(config, X_train.shape[1])

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        import time
        start_time = time.time()

        for epoch in range(config.epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            batch_losses = []

            for i in range(0, len(X_train), config.batch_size):
                batch_idx = indices[i:i + config.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward pass
                activations = self._forward_pass(X_batch, weights, biases, config)

                # Backward pass
                gradients = self._backward_pass(
                    activations, y_batch, weights, config
                )

                # Update weights
                weights, biases = self._update_weights(
                    weights, biases, gradients, config
                )

                # Compute batch loss
                y_pred = activations[-1]
                batch_loss = np.mean((y_pred - y_batch.reshape(y_pred.shape)) ** 2)
                batch_losses.append(batch_loss)

            # Epoch metrics
            train_loss = np.mean(batch_losses)
            train_losses.append(train_loss)

            # Validation loss
            val_activations = self._forward_pass(X_val, weights, biases, config)
            val_pred = val_activations[-1]
            val_loss = np.mean((val_pred - y_val.reshape(val_pred.shape)) ** 2)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        training_time = time.time() - start_time

        # Test evaluation
        test_loss = None
        predictions = None
        actual_values = None

        if "X_test" in data and len(data["X_test"]) > 0:
            test_activations = self._forward_pass(data["X_test"], weights, biases, config)
            predictions = test_activations[-1]
            actual_values = data["y_test"]
            test_loss = np.mean((predictions - actual_values.reshape(predictions.shape)) ** 2)

        return TrainingResult(
            train_loss_history=train_losses,
            val_loss_history=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            final_train_loss=train_losses[-1] if train_losses else 0,
            final_val_loss=val_losses[-1] if val_losses else 0,
            test_loss=test_loss,
            training_time_seconds=training_time,
            stopped_early=patience_counter >= config.early_stopping_patience,
            predictions=predictions,
            actual_values=actual_values,
        )

    def _initialize_weights(
        self,
        config: NetworkConfig,
        input_dim: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Initialize network weights using Xavier/He initialization."""
        weights = []
        biases = []

        prev_dim = input_dim

        for layer in config.layers:
            if layer.layer_type == "dense":
                # He initialization for ReLU variants
                std = np.sqrt(2.0 / prev_dim)
                W = np.random.randn(prev_dim, layer.units) * std
                b = np.zeros(layer.units)
                weights.append(W)
                biases.append(b)
                prev_dim = layer.units
            elif layer.layer_type == "dropout":
                # Dropout doesn't have weights
                weights.append(None)
                biases.append(None)

        return weights, biases

    def _forward_pass(
        self,
        X: np.ndarray,
        weights: list[np.ndarray | None],
        biases: list[np.ndarray | None],
        config: NetworkConfig,
    ) -> list[np.ndarray]:
        """Forward pass through the network."""
        activations = [X]
        current = X

        layer_idx = 0
        for layer in config.layers:
            if layer.layer_type == "dense":
                W = weights[layer_idx]
                b = biases[layer_idx]
                if W is not None and b is not None:
                    z = current @ W + b
                    current = self._apply_activation(z, layer.activation)
                layer_idx += 1
            elif layer.layer_type == "dropout":
                # No dropout during inference
                layer_idx += 1

            activations.append(current)

        return activations

    def _backward_pass(
        self,
        activations: list[np.ndarray],
        y_true: np.ndarray,
        weights: list[np.ndarray | None],
        config: NetworkConfig,
    ) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
        """Backward pass to compute gradients."""
        gradients = []
        m = len(y_true)

        # Output layer gradient
        y_pred = activations[-1]
        delta = (y_pred - y_true.reshape(y_pred.shape)) / m

        # Backward through layers
        layer_idx = len(weights) - 1
        for i, layer in enumerate(reversed(config.layers)):
            if layer.layer_type == "dense":
                # Gradient for weights and bias
                prev_activation = activations[-(i + 2)]
                dW = prev_activation.T @ delta

                # Add regularization
                W = weights[layer_idx]
                if W is not None:
                    if config.l2_regularization > 0:
                        dW += config.l2_regularization * W
                    if config.l1_regularization > 0:
                        dW += config.l1_regularization * np.sign(W)

                db = np.sum(delta, axis=0)
                gradients.insert(0, (dW, db))

                # Propagate delta
                if layer_idx > 0 and W is not None:
                    delta = delta @ W.T
                    # Apply activation derivative
                    if i < len(config.layers) - 1:
                        delta = delta * self._activation_derivative(
                            prev_activation, config.layers[-(i + 2)].activation
                        )

                layer_idx -= 1
            elif layer.layer_type == "dropout":
                gradients.insert(0, (None, None))
                layer_idx -= 1

        return gradients

    def _update_weights(
        self,
        weights: list[np.ndarray | None],
        biases: list[np.ndarray | None],
        gradients: list[tuple[np.ndarray | None, np.ndarray | None]],
        config: NetworkConfig,
    ) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
        """Update weights using gradient descent."""
        lr = config.learning_rate

        for i, (dW, db) in enumerate(gradients):
            if dW is not None and weights[i] is not None:
                weights[i] = weights[i] - lr * dW
            if db is not None and biases[i] is not None:
                biases[i] = biases[i] - lr * db

        return weights, biases

    def _apply_activation(self, z: np.ndarray, activation: ActivationFunction) -> np.ndarray:
        """Apply activation function."""
        if activation == ActivationFunction.RELU:
            return np.maximum(0, z)
        elif activation == ActivationFunction.LEAKY_RELU:
            return np.where(z > 0, z, 0.01 * z)
        elif activation == ActivationFunction.TANH:
            return np.tanh(z)
        elif activation == ActivationFunction.SIGMOID:
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif activation == ActivationFunction.LINEAR:
            return z
        elif activation == ActivationFunction.SOFTMAX:
            exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        else:
            return z

    def _activation_derivative(self, a: np.ndarray, activation: ActivationFunction) -> np.ndarray:
        """Compute activation derivative."""
        if activation == ActivationFunction.RELU:
            return (a > 0).astype(float)
        elif activation == ActivationFunction.LEAKY_RELU:
            return np.where(a > 0, 1, 0.01)
        elif activation == ActivationFunction.TANH:
            return 1 - a ** 2
        elif activation == ActivationFunction.SIGMOID:
            return a * (1 - a)
        elif activation == ActivationFunction.LINEAR:
            return np.ones_like(a)
        else:
            return np.ones_like(a)

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
        """Export training script for a specific framework.

        Args:
            config: Network configuration
            output_path: Path for the output script
            framework: Target ML framework
            data_path: Path to data file (for data loading code)
            include_data_loading: Include data loading code
            include_training: Include training loop
            include_evaluation: Include evaluation code

        Returns:
            Path to exported script
        """
        output_path = Path(output_path)

        if framework == Framework.PYTORCH:
            script = self._generate_pytorch_script(
                config, data_path, include_data_loading, include_training, include_evaluation
            )
        elif framework == Framework.TENSORFLOW:
            script = self._generate_tensorflow_script(
                config, data_path, include_data_loading, include_training, include_evaluation
            )
        else:  # sklearn
            script = self._generate_sklearn_script(
                config, data_path, include_data_loading, include_training, include_evaluation
            )

        output_path.write_text(script)
        logger.info(f"Exported {framework.value} script to {output_path}")
        return output_path

    def _generate_pytorch_script(
        self,
        config: NetworkConfig,
        data_path: str | None,
        include_data_loading: bool,
        include_training: bool,
        include_evaluation: bool,
    ) -> str:
        """Generate PyTorch training script."""
        lines = [
            '"""',
            f"Neural Network Training Script (PyTorch)",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "Configuration:",
            f"  Network Type: {config.network_type.value}",
            f"  Optimizer: {config.optimizer.value}",
            f"  Learning Rate: {config.learning_rate}",
            f"  Epochs: {config.epochs}",
            '"""',
            "",
            "import torch",
            "import torch.nn as nn",
            "import torch.optim as optim",
            "from torch.utils.data import DataLoader, TensorDataset",
            "import numpy as np",
            "import pandas as pd",
            "",
        ]

        # Model definition
        lines.extend([
            "# Model Definition",
            "class NeuralNetwork(nn.Module):",
            "    def __init__(self, input_size, output_size):",
            "        super(NeuralNetwork, self).__init__()",
            "        layers = []",
            f"        prev_size = input_size",
        ])

        for layer in config.layers:
            if layer.layer_type == "dense":
                act = self._pytorch_activation(layer.activation)
                lines.append(f"        layers.append(nn.Linear(prev_size, {layer.units}))")
                if act:
                    lines.append(f"        layers.append({act})")
                lines.append(f"        prev_size = {layer.units}")
            elif layer.layer_type == "dropout":
                lines.append(f"        layers.append(nn.Dropout({layer.dropout_rate}))")

        lines.extend([
            "        self.network = nn.Sequential(*layers)",
            "",
            "    def forward(self, x):",
            "        return self.network(x)",
            "",
        ])

        # Data loading
        if include_data_loading:
            data_path_str = data_path or "data.csv"
            lines.extend([
                "# Data Loading",
                f'data = pd.read_csv("{data_path_str}")',
                "# Specify your feature and target columns",
                "feature_cols = []  # Fill in feature column names",
                "target_cols = []   # Fill in target column names",
                "",
                "X = data[feature_cols].values.astype(np.float32)",
                "y = data[target_cols].values.astype(np.float32)",
                "",
                "# Train/val/test split",
                f"train_size = int(len(X) * {1 - config.validation_split - 0.15})",
                f"val_size = int(len(X) * {config.validation_split})",
                "",
                "X_train, y_train = X[:train_size], y[:train_size]",
                "X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]",
                "X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]",
                "",
            ])

            if config.normalize_inputs:
                lines.extend([
                    "# Normalization",
                    "X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)",
                    "X_std[X_std == 0] = 1",
                    "X_train = (X_train - X_mean) / X_std",
                    "X_val = (X_val - X_mean) / X_std",
                    "X_test = (X_test - X_mean) / X_std",
                    "",
                ])

            lines.extend([
                "# Create DataLoaders",
                "train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))",
                "val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))",
                f"train_loader = DataLoader(train_dataset, batch_size={config.batch_size}, shuffle=True)",
                f"val_loader = DataLoader(val_dataset, batch_size={config.batch_size})",
                "",
            ])

        # Training
        if include_training:
            opt = self._pytorch_optimizer(config.optimizer)
            loss = self._pytorch_loss(config.loss_function)

            lines.extend([
                "# Training Setup",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
                f"model = NeuralNetwork(input_size={config.input_features}, output_size={config.output_features}).to(device)",
                f"optimizer = {opt}(model.parameters(), lr={config.learning_rate})",
                f"criterion = {loss}()",
                "",
                "# Early stopping",
                "best_val_loss = float('inf')",
                "patience_counter = 0",
                f"patience = {config.early_stopping_patience}",
                "",
                "# Training Loop",
                f"for epoch in range({config.epochs}):",
                "    model.train()",
                "    train_loss = 0",
                "    for X_batch, y_batch in train_loader:",
                "        X_batch, y_batch = X_batch.to(device), y_batch.to(device)",
                "        optimizer.zero_grad()",
                "        outputs = model(X_batch)",
                "        loss = criterion(outputs, y_batch)",
                "        loss.backward()",
                "        optimizer.step()",
                "        train_loss += loss.item()",
                "",
                "    # Validation",
                "    model.eval()",
                "    val_loss = 0",
                "    with torch.no_grad():",
                "        for X_batch, y_batch in val_loader:",
                "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)",
                "            outputs = model(X_batch)",
                "            val_loss += criterion(outputs, y_batch).item()",
                "",
                "    val_loss /= len(val_loader)",
                "    print(f'Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}')",
                "",
                "    # Early stopping check",
                "    if val_loss < best_val_loss:",
                "        best_val_loss = val_loss",
                "        patience_counter = 0",
                "        torch.save(model.state_dict(), 'best_model.pth')",
                "    else:",
                "        patience_counter += 1",
                "        if patience_counter >= patience:",
                "            print(f'Early stopping at epoch {epoch+1}')",
                "            break",
                "",
            ])

        # Evaluation
        if include_evaluation:
            lines.extend([
                "# Evaluation",
                "model.load_state_dict(torch.load('best_model.pth'))",
                "model.eval()",
                "with torch.no_grad():",
                "    X_test_tensor = torch.FloatTensor(X_test).to(device)",
                "    predictions = model(X_test_tensor).cpu().numpy()",
                "",
                "# Metrics",
                "from sklearn.metrics import mean_squared_error, r2_score",
                "mse = mean_squared_error(y_test, predictions)",
                "r2 = r2_score(y_test, predictions)",
                "print(f'Test MSE: {mse:.4f}')",
                "print(f'Test R2: {r2:.4f}')",
            ])

        return "\n".join(lines)

    def _generate_tensorflow_script(
        self,
        config: NetworkConfig,
        data_path: str | None,
        include_data_loading: bool,
        include_training: bool,
        include_evaluation: bool,
    ) -> str:
        """Generate TensorFlow/Keras training script."""
        lines = [
            '"""',
            f"Neural Network Training Script (TensorFlow/Keras)",
            f"Generated: {datetime.now().isoformat()}",
            '"""',
            "",
            "import tensorflow as tf",
            "from tensorflow import keras",
            "from tensorflow.keras import layers, callbacks",
            "import numpy as np",
            "import pandas as pd",
            "",
        ]

        # Model definition
        lines.extend([
            "# Model Definition",
            "def create_model(input_size, output_size):",
            "    model = keras.Sequential([",
            f"        layers.Input(shape=(input_size,)),",
        ])

        for layer in config.layers[:-1]:  # Skip last layer
            if layer.layer_type == "dense":
                act = layer.activation.value
                lines.append(f"        layers.Dense({layer.units}, activation='{act}'),")
            elif layer.layer_type == "dropout":
                lines.append(f"        layers.Dropout({layer.dropout_rate}),")

        # Output layer
        output_layer = config.layers[-1]
        output_act = output_layer.activation.value
        lines.extend([
            f"        layers.Dense(output_size, activation='{output_act}'),",
            "    ])",
            "    return model",
            "",
        ])

        # Data loading (similar to PyTorch)
        if include_data_loading:
            data_path_str = data_path or "data.csv"
            lines.extend([
                "# Data Loading",
                f'data = pd.read_csv("{data_path_str}")',
                "feature_cols = []  # Fill in feature column names",
                "target_cols = []   # Fill in target column names",
                "",
                "X = data[feature_cols].values.astype(np.float32)",
                "y = data[target_cols].values.astype(np.float32)",
                "",
                f"train_size = int(len(X) * {1 - config.validation_split - 0.15})",
                f"val_size = int(len(X) * {config.validation_split})",
                "",
                "X_train, y_train = X[:train_size], y[:train_size]",
                "X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]",
                "X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]",
                "",
            ])

        # Training
        if include_training:
            opt_name = config.optimizer.value
            loss_name = self._keras_loss(config.loss_function)

            lines.extend([
                "# Create and compile model",
                f"model = create_model({config.input_features}, {config.output_features})",
                f"model.compile(",
                f"    optimizer='{opt_name}',",
                f"    loss='{loss_name}',",
                "    metrics=['mae']",
                ")",
                "",
                "# Callbacks",
                "early_stop = callbacks.EarlyStopping(",
                f"    patience={config.early_stopping_patience},",
                "    restore_best_weights=True",
                ")",
                "reduce_lr = callbacks.ReduceLROnPlateau(",
                f"    patience={config.reduce_lr_patience},",
                "    factor=0.5",
                ")",
                "",
                "# Training",
                "history = model.fit(",
                "    X_train, y_train,",
                f"    epochs={config.epochs},",
                f"    batch_size={config.batch_size},",
                "    validation_data=(X_val, y_val),",
                "    callbacks=[early_stop, reduce_lr],",
                "    verbose=1",
                ")",
                "",
                "# Save model",
                "model.save('trained_model.keras')",
            ])

        # Evaluation
        if include_evaluation:
            lines.extend([
                "",
                "# Evaluation",
                "results = model.evaluate(X_test, y_test)",
                "print(f'Test Loss: {results[0]:.4f}')",
                "print(f'Test MAE: {results[1]:.4f}')",
                "",
                "# Predictions",
                "predictions = model.predict(X_test)",
            ])

        return "\n".join(lines)

    def _generate_sklearn_script(
        self,
        config: NetworkConfig,
        data_path: str | None,
        include_data_loading: bool,
        include_training: bool,
        include_evaluation: bool,
    ) -> str:
        """Generate scikit-learn training script."""
        lines = [
            '"""',
            f"Neural Network Training Script (scikit-learn)",
            f"Generated: {datetime.now().isoformat()}",
            '"""',
            "",
            "from sklearn.neural_network import MLPRegressor, MLPClassifier",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.metrics import mean_squared_error, r2_score",
            "import numpy as np",
            "import pandas as pd",
            "import joblib",
            "",
        ]

        # Extract layer sizes
        hidden_sizes = [l.units for l in config.layers if l.layer_type == "dense"][:-1]

        if include_data_loading:
            data_path_str = data_path or "data.csv"
            lines.extend([
                "# Data Loading",
                f'data = pd.read_csv("{data_path_str}")',
                "feature_cols = []  # Fill in feature column names",
                "target_cols = []   # Fill in target column names",
                "",
                "X = data[feature_cols].values",
                "y = data[target_cols].values.ravel()",
                "",
                "# Split data",
                "X_train, X_temp, y_train, y_temp = train_test_split(",
                f"    X, y, test_size={config.validation_split + 0.15}, random_state=42",
                ")",
                "X_val, X_test, y_val, y_test = train_test_split(",
                "    X_temp, y_temp, test_size=0.5, random_state=42",
                ")",
                "",
            ])

            if config.normalize_inputs:
                lines.extend([
                    "# Normalization",
                    "scaler = StandardScaler()",
                    "X_train = scaler.fit_transform(X_train)",
                    "X_val = scaler.transform(X_val)",
                    "X_test = scaler.transform(X_test)",
                    "joblib.dump(scaler, 'scaler.joblib')",
                    "",
                ])

        if include_training:
            model_class = "MLPRegressor" if config.task_type == "regression" else "MLPClassifier"
            lines.extend([
                "# Model",
                f"model = {model_class}(",
                f"    hidden_layer_sizes={tuple(hidden_sizes)},",
                f"    activation='{config.layers[0].activation.value}',",
                f"    solver='{config.optimizer.value}',",
                f"    learning_rate_init={config.learning_rate},",
                f"    max_iter={config.epochs},",
                f"    batch_size={config.batch_size},",
                f"    early_stopping=True,",
                f"    validation_fraction={config.validation_split},",
                f"    n_iter_no_change={config.early_stopping_patience},",
                "    verbose=True,",
                "    random_state=42",
                ")",
                "",
                "# Training",
                "model.fit(X_train, y_train)",
                "",
                "# Save model",
                "joblib.dump(model, 'trained_model.joblib')",
            ])

        if include_evaluation:
            lines.extend([
                "",
                "# Evaluation",
                "predictions = model.predict(X_test)",
                "mse = mean_squared_error(y_test, predictions)",
                "r2 = r2_score(y_test, predictions)",
                "print(f'Test MSE: {mse:.4f}')",
                "print(f'Test R2: {r2:.4f}')",
            ])

        return "\n".join(lines)

    def _pytorch_activation(self, activation: ActivationFunction) -> str:
        """Convert activation to PyTorch module string."""
        mapping = {
            ActivationFunction.RELU: "nn.ReLU()",
            ActivationFunction.LEAKY_RELU: "nn.LeakyReLU(0.01)",
            ActivationFunction.ELU: "nn.ELU()",
            ActivationFunction.SELU: "nn.SELU()",
            ActivationFunction.TANH: "nn.Tanh()",
            ActivationFunction.SIGMOID: "nn.Sigmoid()",
            ActivationFunction.GELU: "nn.GELU()",
            ActivationFunction.LINEAR: "",
        }
        return mapping.get(activation, "nn.ReLU()")

    def _pytorch_optimizer(self, optimizer: Optimizer) -> str:
        """Convert optimizer to PyTorch optimizer string."""
        mapping = {
            Optimizer.SGD: "optim.SGD",
            Optimizer.ADAM: "optim.Adam",
            Optimizer.ADAMW: "optim.AdamW",
            Optimizer.RMSPROP: "optim.RMSprop",
            Optimizer.ADAGRAD: "optim.Adagrad",
        }
        return mapping.get(optimizer, "optim.Adam")

    def _pytorch_loss(self, loss: LossFunction) -> str:
        """Convert loss to PyTorch loss string."""
        mapping = {
            LossFunction.MSE: "nn.MSELoss",
            LossFunction.MAE: "nn.L1Loss",
            LossFunction.HUBER: "nn.SmoothL1Loss",
            LossFunction.CROSS_ENTROPY: "nn.CrossEntropyLoss",
            LossFunction.BINARY_CROSS_ENTROPY: "nn.BCELoss",
        }
        return mapping.get(loss, "nn.MSELoss")

    def _keras_loss(self, loss: LossFunction) -> str:
        """Convert loss to Keras loss string."""
        mapping = {
            LossFunction.MSE: "mse",
            LossFunction.MAE: "mae",
            LossFunction.HUBER: "huber",
            LossFunction.CROSS_ENTROPY: "categorical_crossentropy",
            LossFunction.BINARY_CROSS_ENTROPY: "binary_crossentropy",
        }
        return mapping.get(loss, "mse")

    def export_config(self, config: NetworkConfig, output_path: Path | str) -> Path:
        """Export network configuration to JSON."""
        output_path = Path(output_path)
        config_dict = config.to_dict()
        config_dict["normalization_params"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self._normalization_params.items()
        }

        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        return output_path

    def import_config(self, config_path: Path | str) -> NetworkConfig:
        """Import network configuration from JSON."""
        with open(config_path) as f:
            data = json.load(f)

        if "normalization_params" in data:
            self._normalization_params = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in data.pop("normalization_params").items()
            }

        config = NetworkConfig.from_dict(data)
        self._config = config
        return config


__all__ = [
    "Framework",
    "NetworkType",
    "ActivationFunction",
    "LossFunction",
    "Optimizer",
    "LayerConfig",
    "NetworkConfig",
    "TrainingResult",
    "DataSplitConfig",
    "NeuralNetworkInterface",
]
