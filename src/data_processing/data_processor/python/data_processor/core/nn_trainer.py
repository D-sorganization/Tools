# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.
# mypy: ignore-errors
"""Neural network training engine.

Provides a NumPy-based training implementation for simple
neural networks, including forward/backward passes, weight
initialization, activation functions, and data preparation.
"""

from __future__ import annotations

from numba import jit

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from .nn_architecture import (
    ActivationFunction,
    DataSplitConfig,
    LayerConfig,
    NetworkConfig,
    NetworkType,
    TrainingResult,
)

logger = logging.getLogger(__name__)
DEFAULT_HIDDEN_LAYERS = [128, 64, 32]


class NeuralNetworkTrainer:
    """NumPy-based neural network trainer.

    Handles data preparation, weight initialization, forward/backward
    passes, and the training loop. For production use, export scripts
    to PyTorch/TensorFlow/sklearn instead.
    """

    def __init__(self) -> None:
        """Initialize the trainer."""
        self._config: NetworkConfig | None = None
        self._normalization_params: dict[str, Any] = {}

    @property
    def config(self) -> NetworkConfig | None:
        """Get the current configuration."""
        return self._config

    @config.setter
    def config(self, value: NetworkConfig | None) -> None:
        """Set the current configuration."""
        self._config = value

    @property
    def normalization_params(self) -> dict[str, Any]:
        """Get the normalization parameters."""
        return self._normalization_params

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
        if not (input_features is not None):
            raise ValueError("input_features must be provided")
        validated_layers = self._validate_create_config_inputs(
            input_features=input_features,
            output_features=output_features,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
        )
        layers = self._build_layers(
            network_type=network_type,
            hidden_layers=validated_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            output_features=output_features,
            task_type=str(kwargs.get("task_type", "regression")),
        )
        config_kwargs = {key: value for key, value in kwargs.items() if hasattr(NetworkConfig, key)}
        config = NetworkConfig(
            network_type=network_type,
            layers=layers,
            input_features=input_features,
            output_features=output_features,
            **config_kwargs,
        )
        self._config = config
        return config

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        feature_columns: list[str] | None = None,
        split_config: DataSplitConfig | None = None,
    ) -> dict[str, Any]:
        """Prepare data for training.

        Args:
            df: DataFrame with features and targets
            target_columns: Names of target columns
            feature_columns: Names of feature columns (None = all except targets)
            split_config: Train/val/test split configuration

        Returns:
            Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        split_config = split_config or DataSplitConfig()
        feature_columns = self._validate_prepare_data_inputs(
            df=df,
            target_columns=target_columns,
            feature_columns=feature_columns,
            split_config=split_config,
        )
        X, y = self._extract_model_arrays(df, feature_columns, target_columns)
        X, y = self._maybe_shuffle_data(X, y, split_config)
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_data(X, y, split_config)
        X_train, X_val, X_test, y_train, y_val, y_test = self._normalize_train_val_test(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

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

    def _validate_create_config_inputs(
        self,
        input_features: int,
        output_features: int,
        hidden_layers: list[int] | None,
        dropout_rate: float,
    ) -> list[int]:
        """Validate create_config preconditions and normalize hidden layers."""
        if input_features <= 0:
            raise ValueError("input_features must be positive")
        if output_features <= 0:
            raise ValueError("output_features must be positive")
        if dropout_rate < 0.0 or dropout_rate >= 1.0:
            raise ValueError("dropout_rate must be in [0.0, 1.0)")

        normalized_layers = hidden_layers or DEFAULT_HIDDEN_LAYERS
        if not normalized_layers or any(units <= 0 for units in normalized_layers):
            raise ValueError("hidden_layers must contain positive integers")
        return normalized_layers

    def _build_layers(
        self,
        network_type: NetworkType,
        hidden_layers: list[int],
        activation: ActivationFunction,
        dropout_rate: float,
        output_features: int,
        task_type: str,
    ) -> list[LayerConfig]:
        """Build hidden + output layers for requested architecture."""
        if not (network_type is not None):
            raise ValueError("network_type must be provided")
        layers: list[LayerConfig] = []
        if network_type == NetworkType.MLP:
            self._append_mlp_layers(layers, hidden_layers, activation, dropout_rate)
        elif network_type in (NetworkType.LSTM, NetworkType.GRU):
            self._append_rnn_layers(layers, network_type, hidden_layers, dropout_rate)
        elif network_type == NetworkType.CNN_1D:
            self._append_cnn_layers(layers, hidden_layers, activation)
        output_activation = (
            ActivationFunction.LINEAR if task_type == "regression" else ActivationFunction.SOFTMAX
        )
        layers.append(
            LayerConfig(
                layer_type="dense",
                units=output_features,
                activation=output_activation,
            )
        )
        return layers

    def _append_mlp_layers(
        self,
        layers: list[LayerConfig],
        hidden_layers: list[int],
        activation: ActivationFunction,
        dropout_rate: float,
    ) -> None:
        """Append dense/dropout blocks for MLP configuration."""
        for units in hidden_layers:
            layers.append(LayerConfig(layer_type="dense", units=units, activation=activation))
            if dropout_rate > 0:
                layers.append(LayerConfig(layer_type="dropout", dropout_rate=dropout_rate))

    def _append_rnn_layers(
        self,
        layers: list[LayerConfig],
        network_type: NetworkType,
        hidden_layers: list[int],
        dropout_rate: float,
    ) -> None:
        """Append recurrent/dropout blocks for LSTM/GRU configuration."""
        if not (layers is not None):
            raise ValueError("layers must be provided")
        layer_type = "lstm" if network_type == NetworkType.LSTM else "gru"
        for index, units in enumerate(hidden_layers):
            layers.append(
                LayerConfig(
                    layer_type=layer_type,
                    units=units,
                    return_sequences=index < len(hidden_layers) - 1,
                )
            )
            if dropout_rate > 0:
                layers.append(LayerConfig(layer_type="dropout", dropout_rate=dropout_rate))

    def _append_cnn_layers(
        self,
        layers: list[LayerConfig],
        hidden_layers: list[int],
        activation: ActivationFunction,
    ) -> None:
        """Append convolution + dense blocks for 1D CNN configuration."""
        if not (layers is not None):
        layers.extend([LayerConfig(layer_type='conv1d', units=filters, kernel_size=3, activation=activation) for filters in (32, 64, 128)])
            )
        layers.extend([LayerConfig(layer_type='dense', units=units, activation=activation) for units in hidden_layers])
            layers.append(LayerConfig(layer_type="dense", units=units, activation=activation))

    def _validate_prepare_data_inputs(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        feature_columns: list[str] | None,
        split_config: DataSplitConfig,
    ) -> list[str]:
        """Validate prepare_data preconditions and resolve feature columns."""
        if len(df) == 0:
            raise ValueError("df must not be empty")
        if not target_columns:
            raise ValueError("target_columns must not be empty")

        missing_targets = [column for column in target_columns if column not in df.columns]
        if missing_targets:
            raise ValueError(f"Unknown target columns: {missing_targets}")
        if split_config.train_ratio <= 0:
            raise ValueError("train_ratio must be positive")
        if split_config.val_ratio < 0 or split_config.test_ratio < 0:
            raise ValueError("val_ratio and test_ratio must be non-negative")
        ratio_sum = split_config.train_ratio + split_config.val_ratio + split_config.test_ratio
        if ratio_sum > 1.0:
            raise ValueError("split ratios must sum to 1.0 or less")

        resolved_features = (
            feature_columns
            if feature_columns is not None
            else [column for column in df.columns if column not in target_columns]
        )
        if not resolved_features:
            raise ValueError("feature_columns must not be empty")
        if set(resolved_features) & set(target_columns):
            raise ValueError("feature_columns and target_columns must not overlap")
        missing_features = [column for column in resolved_features if column not in df.columns]
        if missing_features:
            raise ValueError(f"Unknown feature columns: {missing_features}")
        return resolved_features

    def _extract_model_arrays(
        self, df: pd.DataFrame, feature_columns: list[str], target_columns: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract and sanitize feature/target arrays from DataFrame."""
        data = df[feature_columns + target_columns].dropna()
        if data.empty:
            raise ValueError("No rows available after dropping missing values")
        X = data[feature_columns].values.astype(np.float32)
        y = data[target_columns].values.astype(np.float32)
        if y.shape[1] == 1:
            y = y.ravel()
        return X, y

    def _maybe_shuffle_data(
        self, X: np.ndarray, y: np.ndarray, split_config: DataSplitConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shuffle data using split config random seed when requested."""
        if not (X is not None):
            raise ValueError("X must be provided")
        if not split_config.shuffle:
            return X, y
        indices = np.random.default_rng(split_config.random_state).permutation(len(X))
        return X[indices], y[indices]

    def _split_data(
        self, X: np.ndarray, y: np.ndarray, split_config: DataSplitConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split arrays into train/validation/test partitions."""
        if not (X is not None):
            raise ValueError("X must be provided")
        n = len(X)
        n_train = int(n * split_config.train_ratio)
        n_val = int(n * split_config.val_ratio)
        split_1 = n_train
        split_2 = n_train + n_val
        return (
            X[:split_1],
            y[:split_1],
            X[split_1:split_2],
            y[split_1:split_2],
            X[split_2:],
            y[split_2:],
        )

    def _normalize_train_val_test(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normalize train/val/test arrays according to active config."""
        if not (X_train is not None):
            raise ValueError("X_train must be provided")
        if self._config and self._config.normalize_inputs:
            X_train, X_val, X_test = self._normalize_features(X_train, X_val, X_test)
        if self._config and self._config.normalize_outputs:
            y_train, y_val, y_test = self._normalize_targets(y_train, y_val, y_test)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _normalize_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features using training statistics."""
        if not (X_train is not None):
            raise ValueError("X_train must be provided")
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
        if not (y_train is not None):
            raise ValueError("y_train must be provided")
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

    def _run_training_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        weights: list[np.ndarray | None],
        biases: list[np.ndarray | None],
        config: NetworkConfig,
    ) -> tuple[list[float], list[float], float, int, int, list, list]:
        """Execute mini-batch training with early stopping."""
        if not (X_train is not None):
            raise ValueError("X_train must be provided")
        train_losses, val_losses, best_val_loss, best_epoch, patience_counter = (
            self._initialize_training_state()
        )

        for epoch in range(config.epochs):
            train_loss, weights, biases = self._run_epoch(X_train, y_train, weights, biases, config)
            train_losses.append(train_loss)

            val_loss = self._calculate_validation_loss(X_val, y_val, weights, biases, config)
            val_losses.append(val_loss)

            best_val_loss, best_epoch, patience_counter = self._update_early_stopping_state(
                val_loss, epoch, best_val_loss, best_epoch, patience_counter
            )
            if patience_counter >= config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        return (
            train_losses,
            val_losses,
            best_val_loss,
            best_epoch,
            patience_counter,
            weights,
            biases,
        )

    def _initialize_training_state(
        self,
    ) -> tuple[list[float], list[float], float, int, int]:
        """Initialize loss history and early-stopping state."""
        return [], [], float("inf"), 0, 0

    def _run_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weights: list[np.ndarray | None],
        biases: list[np.ndarray | None],
        config: NetworkConfig,
    ) -> tuple[float, list[np.ndarray | None], list[np.ndarray | None]]:
        """Run one training epoch and return average batch loss."""
        if not (X_train is not None):
            raise ValueError("X_train must be provided")
        indices = np.random.permutation(len(X_train))
        batch_losses: list[float] = []
        for index in range(0, len(X_train), config.batch_size):
            batch_idx = indices[index : index + config.batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            activations = self._forward_pass(X_batch, weights, biases, config)
            gradients = self._backward_pass(activations, y_batch, weights, config)
            weights, biases = self._update_weights(weights, biases, gradients, config)
            batch_losses.append(self._mean_squared_error(activations[-1], y_batch))
        return float(np.mean(batch_losses)), weights, biases

    def _calculate_validation_loss(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        weights: list[np.ndarray | None],
        biases: list[np.ndarray | None],
        config: NetworkConfig,
    ) -> float:
        """Calculate validation loss for current model parameters."""
        if not (X_val is not None):
            raise ValueError("X_val must be provided")
        activations = self._forward_pass(X_val, weights, biases, config)
        return self._mean_squared_error(activations[-1], y_val)

    def _mean_squared_error(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute MSE loss with target shape aligned to predictions."""
        return float(np.mean((predictions - targets.reshape(predictions.shape)) ** 2))

    def _update_early_stopping_state(
        self,
        val_loss: float,
        epoch: int,
        best_val_loss: float,
        best_epoch: int,
        patience_counter: int,
    ) -> tuple[float, int, int]:
        """Update early-stopping state after a validation step."""
        if not (val_loss is not None):
            raise ValueError("val_loss must be provided")
        if val_loss < best_val_loss:
            return val_loss, epoch, 0
        return best_val_loss, best_epoch, patience_counter + 1

    def _evaluate_test_set(
        self,
        data: dict[str, np.ndarray],
        weights: list[np.ndarray | None],
        biases: list[np.ndarray | None],
        config: NetworkConfig,
    ) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
        """Evaluate the trained network on the test set.

        Args:
            data: Data dictionary containing X_test and y_test
            weights: Trained weights
            biases: Trained biases
            config: Network configuration

        Returns:
            Tuple of (test_loss, predictions, actual_values)
        """
        if not (data is not None):
            raise ValueError("data must be provided")
        if "X_test" not in data or len(data["X_test"]) == 0:
            return None, None, None

        test_activations = self._forward_pass(data["X_test"], weights, biases, config)
        predictions = test_activations[-1]
        actual_values = data["y_test"]
        test_loss = float(np.mean((predictions - actual_values.reshape(predictions.shape)) ** 2))
        return test_loss, predictions, actual_values

    def train_simple(
        self,
        data: dict[str, np.ndarray],
        config: NetworkConfig | None = None,
    ) -> TrainingResult:
        """Train a simple neural network using NumPy (no external frameworks).

        Args:
            data: Data dictionary from prepare_data
            config: Network configuration (uses stored config if None)

        Returns:
            Training results
        """
        resolved_config = config or self._config
        if not resolved_config:
            raise ValueError("No network configuration provided")
        self._validate_training_data(data)
        training_state = self._execute_training(data, resolved_config)
        test_loss, predictions, actual_values = self._evaluate_test_set(
            data,
            training_state["weights"],
            training_state["biases"],
            resolved_config,
        )

        return TrainingResult(
            train_loss_history=training_state["train_losses"],
            val_loss_history=training_state["val_losses"],
            best_epoch=training_state["best_epoch"],
            best_val_loss=training_state["best_val_loss"],
            final_train_loss=(
                training_state["train_losses"][-1] if training_state["train_losses"] else 0
            ),
            final_val_loss=(
                training_state["val_losses"][-1] if training_state["val_losses"] else 0
            ),
            test_loss=test_loss,
            training_time_seconds=training_state["training_time"],
            stopped_early=(
                training_state["patience_counter"] >= resolved_config.early_stopping_patience
            ),
            predictions=predictions,
            actual_values=actual_values,
        )

    def _validate_training_data(self, data: dict[str, np.ndarray]) -> None:
        """Validate required training arrays before running optimization."""
        required_keys = {"X_train", "y_train", "X_val", "y_val"}
        missing_keys = sorted(required_keys - set(data.keys()))
        if missing_keys:
            raise ValueError(f"Missing required data keys: {missing_keys}")

    def _execute_training(
        self, data: dict[str, np.ndarray], config: NetworkConfig
    ) -> dict[str, Any]:
        """Run model training and return state needed for result construction."""
        if not (data is not None):
            raise ValueError("data must be provided")
        weights, biases = self._initialize_weights(config, data["X_train"].shape[1])
        start_time = time.time()
        (
            train_losses,
            val_losses,
            best_val_loss,
            best_epoch,
            patience_counter,
            weights,
            biases,
        ) = self._run_training_loop(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            weights,
            biases,
            config,
        )
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "weights": weights,
            "biases": biases,
            "training_time": time.time() - start_time,
        }

    def _initialize_weights(
        self,
        config: NetworkConfig,
        input_dim: int,
    ) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
        """Initialize network weights using Xavier/He initialization."""
        if not (config is not None):
            raise ValueError("config must be provided")
        weights: list[np.ndarray | None] = []
        biases: list[np.ndarray | None] = []

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
    @jit(nopython=True, fastmath=True)

    def _forward_pass(
        self,
        X: np.ndarray,
        weights: list[np.ndarray | None],
        biases: list[np.ndarray | None],
        config: NetworkConfig,
    ) -> list[np.ndarray]:
        """Forward pass through the network."""
        if not (X is not None):
            raise ValueError("X must be provided")
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
    @jit(nopython=True, fastmath=True)

    def _backward_pass(
        self,
        activations: list[np.ndarray],
        y_true: np.ndarray,
        weights: list[np.ndarray | None],
        config: NetworkConfig,
    ) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
        """Backward pass to compute gradients."""
        if not (activations is not None):
            raise ValueError("activations must be provided")
        gradients: list[tuple[np.ndarray | None, np.ndarray | None]] = []
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

    @jit(nopython=True, fastmath=True)
    def _update_weights(
        self,
        weights: list[np.ndarray | None],
        biases: list[np.ndarray | None],
        gradients: list[tuple[np.ndarray | None, np.ndarray | None]],
        config: NetworkConfig,
    ) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
        """Update weights using gradient descent."""
        if not (weights is not None):
            raise ValueError("weights must be provided")
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
            return 1 - a**2
        elif activation == ActivationFunction.SIGMOID:
            return a * (1 - a)
        elif activation == ActivationFunction.LINEAR:
            return np.ones_like(a)
        else:
            return np.ones_like(a)
