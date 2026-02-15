# mypy: ignore-errors
"""Neural network training engine.

Provides a NumPy-based training implementation for simple
neural networks, including forward/backward passes, weight
initialization, activation functions, and data preparation.
"""

from __future__ import annotations

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
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        # Build layers based on network type
        layers: list[LayerConfig] = []

        if network_type == NetworkType.MLP:
            for units in hidden_layers:
                layers.append(
                    LayerConfig(
                        layer_type="dense",
                        units=units,
                        activation=activation,
                    )
                )
                if dropout_rate > 0:
                    layers.append(
                        LayerConfig(
                            layer_type="dropout",
                            dropout_rate=dropout_rate,
                        )
                    )

        elif network_type in (NetworkType.LSTM, NetworkType.GRU):
            layer_type = "lstm" if network_type == NetworkType.LSTM else "gru"
            for i, units in enumerate(hidden_layers):
                is_last = i == len(hidden_layers) - 1
                layers.append(
                    LayerConfig(
                        layer_type=layer_type,
                        units=units,
                        return_sequences=not is_last,
                    )
                )
                if dropout_rate > 0:
                    layers.append(
                        LayerConfig(
                            layer_type="dropout",
                            dropout_rate=dropout_rate,
                        )
                    )

        elif network_type == NetworkType.CNN_1D:
            filters = [32, 64, 128]
            for f in filters:
                layers.append(
                    LayerConfig(
                        layer_type="conv1d",
                        units=f,
                        kernel_size=3,
                        activation=activation,
                    )
                )
            layers.append(LayerConfig(layer_type="flatten"))
            for units in hidden_layers:
                layers.append(
                    LayerConfig(
                        layer_type="dense",
                        units=units,
                        activation=activation,
                    )
                )

        # Output layer
        output_activation = (
            ActivationFunction.LINEAR
            if kwargs.get("task_type", "regression") == "regression"
            else ActivationFunction.SOFTMAX
        )
        layers.append(
            LayerConfig(
                layer_type="dense",
                units=output_features,
                activation=output_activation,
            )
        )

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

        X_val = X[n_train : n_train + n_val]
        y_val = (
            y[n_train : n_train + n_val]
            if y.ndim == 1
            else y[n_train : n_train + n_val]
        )

        X_test = X[n_train + n_val :]
        y_test = y[n_train + n_val :] if y.ndim == 1 else y[n_train + n_val :]

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

        Args:
            data: Data dictionary from prepare_data
            config: Network configuration (uses stored config if None)

        Returns:
            Training results
        """
        config = config or self._config
        if not config:
            raise ValueError("No network configuration provided")

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]

        # Initialize weights
        weights, biases = self._initialize_weights(config, X_train.shape[1])

        # Training loop
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        start_time = time.time()

        for epoch in range(config.epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            batch_losses: list[float] = []

            for i in range(0, len(X_train), config.batch_size):
                batch_idx = indices[i : i + config.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward pass
                activations = self._forward_pass(X_batch, weights, biases, config)

                # Backward pass
                gradients = self._backward_pass(activations, y_batch, weights, config)

                # Update weights
                weights, biases = self._update_weights(
                    weights, biases, gradients, config
                )

                # Compute batch loss
                y_pred = activations[-1]
                batch_loss = float(
                    np.mean((y_pred - y_batch.reshape(y_pred.shape)) ** 2)
                )
                batch_losses.append(batch_loss)

            # Epoch metrics
            train_loss = float(np.mean(batch_losses))
            train_losses.append(train_loss)

            # Validation loss
            val_activations = self._forward_pass(X_val, weights, biases, config)
            val_pred = val_activations[-1]
            val_loss = float(np.mean((val_pred - y_val.reshape(val_pred.shape)) ** 2))
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        training_time = time.time() - start_time

        # Test evaluation
        test_loss = None
        predictions = None
        actual_values = None

        if "X_test" in data and len(data["X_test"]) > 0:
            test_activations = self._forward_pass(
                data["X_test"], weights, biases, config
            )
            predictions = test_activations[-1]
            actual_values = data["y_test"]
            test_loss = float(
                np.mean((predictions - actual_values.reshape(predictions.shape)) ** 2)
            )

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
    ) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
        """Initialize network weights using Xavier/He initialization."""
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

    def _apply_activation(
        self, z: np.ndarray, activation: ActivationFunction
    ) -> np.ndarray:
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

    def _activation_derivative(
        self, a: np.ndarray, activation: ActivationFunction
    ) -> np.ndarray:
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
