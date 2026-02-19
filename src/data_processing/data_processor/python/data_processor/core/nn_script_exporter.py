# mypy: ignore-errors
"""Neural network script exporter.

Generates standalone training scripts for PyTorch, TensorFlow/Keras,
and scikit-learn from a NetworkConfig. Also handles config
import/export to JSON.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .nn_architecture import (
    ActivationFunction,
    Framework,
    LossFunction,
    NetworkConfig,
    Optimizer,
)

logger = logging.getLogger(__name__)


class NeuralNetworkScriptExporter:
    """Generates framework-specific training scripts.

    Converts a NetworkConfig into a standalone Python training
    script for PyTorch, TensorFlow/Keras, or scikit-learn.
    """

    def __init__(self) -> None:
        """Initialize the exporter."""
        self._normalization_params: dict[str, Any] = {}

    @property
    def normalization_params(self) -> dict[str, Any]:
        """Get the normalization parameters."""
        return self._normalization_params

    @normalization_params.setter
    def normalization_params(self, value: dict[str, Any]) -> None:
        """Set the normalization parameters."""
        self._normalization_params = value

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
                config,
                data_path,
                include_data_loading,
                include_training,
                include_evaluation,
            )
        elif framework == Framework.TENSORFLOW:
            script = self._generate_tensorflow_script(
                config,
                data_path,
                include_data_loading,
                include_training,
                include_evaluation,
            )
        else:  # sklearn
            script = self._generate_sklearn_script(
                config,
                data_path,
                include_data_loading,
                include_training,
                include_evaluation,
            )

        output_path.write_text(script)
        logger.info("Exported %s script to %s", framework.value, output_path)
        return output_path

    def export_config(
        self,
        config: NetworkConfig,
        output_path: Path | str,
    ) -> Path:
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

        return NetworkConfig.from_dict(data)

    # ------------------------------------------------------------------ #
    #  PyTorch script generation
    # ------------------------------------------------------------------ #

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
            "Neural Network Training Script (PyTorch)",
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
        lines.extend(
            [
                "# Model Definition",
                "class NeuralNetwork(nn.Module):",
                "    def __init__(self, input_size, output_size):",
                "        super(NeuralNetwork, self).__init__()",
                "        layers = []",
                "        prev_size = input_size",
            ]
        )

        for layer in config.layers:
            if layer.layer_type == "dense":
                act = self._pytorch_activation(layer.activation)
                lines.append(
                    f"        layers.append(nn.Linear(prev_size, {layer.units}))"
                )
                if act:
                    lines.append(f"        layers.append({act})")
                lines.append(f"        prev_size = {layer.units}")
            elif layer.layer_type == "dropout":
                lines.append(f"        layers.append(nn.Dropout({layer.dropout_rate}))")

        lines.extend(
            [
                "        self.network = nn.Sequential(*layers)",
                "",
                "    def forward(self, x):",
                "        return self.network(x)",
                "",
            ]
        )

        # Data loading
        if include_data_loading:
            data_path_str = data_path or "data.csv"
            lines.extend(
                [
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
                    "X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]",  # noqa: E501
                    "X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]",
                    "",
                ]
            )

            if config.normalize_inputs:
                lines.extend(
                    [
                        "# Normalization",
                        "X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)",
                        "X_std[X_std == 0] = 1",
                        "X_train = (X_train - X_mean) / X_std",
                        "X_val = (X_val - X_mean) / X_std",
                        "X_test = (X_test - X_mean) / X_std",
                        "",
                    ]
                )

            lines.extend(
                [
                    "# Create DataLoaders",
                    "train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))",  # noqa: E501
                    "val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))",  # noqa: E501
                    f"train_loader = DataLoader(train_dataset, batch_size={config.batch_size}, shuffle=True)",  # noqa: E501
                    f"val_loader = DataLoader(val_dataset, batch_size={config.batch_size})",  # noqa: E501
                    "",
                ]
            )

        # Training
        if include_training:
            opt = self._pytorch_optimizer(config.optimizer)
            loss = self._pytorch_loss(config.loss_function)

            lines.extend(
                [
                    "# Training Setup",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",  # noqa: E501
                    f"model = NeuralNetwork(input_size={config.input_features}, output_size={config.output_features}).to(device)",  # noqa: E501
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
                    "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)",  # noqa: E501
                    "            outputs = model(X_batch)",
                    "            val_loss += criterion(outputs, y_batch).item()",
                    "",
                    "    val_loss /= len(val_loader)",
                    "    print(f'Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}')",  # noqa: E501
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
                ]
            )

        # Evaluation
        if include_evaluation:
            lines.extend(
                [
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
                ]
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  TensorFlow script generation
    # ------------------------------------------------------------------ #

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
            "Neural Network Training Script (TensorFlow/Keras)",
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
        lines.extend(
            [
                "# Model Definition",
                "def create_model(input_size, output_size):",
                "    model = keras.Sequential([",
                "        layers.Input(shape=(input_size,)),",
            ]
        )

        for layer in config.layers[:-1]:  # Skip last layer
            if layer.layer_type == "dense":
                act = layer.activation.value
                lines.append(
                    f"        layers.Dense({layer.units}, activation='{act}'),"
                )
            elif layer.layer_type == "dropout":
                lines.append(f"        layers.Dropout({layer.dropout_rate}),")

        # Output layer
        output_layer = config.layers[-1]
        output_act = output_layer.activation.value
        lines.extend(
            [
                f"        layers.Dense(output_size, activation='{output_act}'),",
                "    ])",
                "    return model",
                "",
            ]
        )

        # Data loading
        if include_data_loading:
            data_path_str = data_path or "data.csv"
            lines.extend(
                [
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
                    "X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]",  # noqa: E501
                    "X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]",
                    "",
                ]
            )

        # Training
        if include_training:
            opt_name = config.optimizer.value
            loss_name = self._keras_loss(config.loss_function)

            lines.extend(
                [
                    "# Create and compile model",
                    f"model = create_model({config.input_features}, {config.output_features})",  # noqa: E501
                    "model.compile(",
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
                ]
            )

        # Evaluation
        if include_evaluation:
            lines.extend(
                [
                    "",
                    "# Evaluation",
                    "results = model.evaluate(X_test, y_test)",
                    "print(f'Test Loss: {results[0]:.4f}')",
                    "print(f'Test MAE: {results[1]:.4f}')",
                    "",
                    "# Predictions",
                    "predictions = model.predict(X_test)",
                ]
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  scikit-learn script generation
    # ------------------------------------------------------------------ #

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
            "Neural Network Training Script (scikit-learn)",
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
        hidden_sizes = [
            layer_cfg.units
            for layer_cfg in config.layers
            if layer_cfg.layer_type == "dense"
        ][:-1]

        if include_data_loading:
            data_path_str = data_path or "data.csv"
            lines.extend(
                [
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
                    f"    X, y, test_size={config.validation_split + 0.15}, random_state=42",  # noqa: E501
                    ")",
                    "X_val, X_test, y_val, y_test = train_test_split(",
                    "    X_temp, y_temp, test_size=0.5, random_state=42",
                    ")",
                    "",
                ]
            )

            if config.normalize_inputs:
                lines.extend(
                    [
                        "# Normalization",
                        "scaler = StandardScaler()",
                        "X_train = scaler.fit_transform(X_train)",
                        "X_val = scaler.transform(X_val)",
                        "X_test = scaler.transform(X_test)",
                        "joblib.dump(scaler, 'scaler.joblib')",
                        "",
                    ]
                )

        if include_training:
            model_class = (
                "MLPRegressor" if config.task_type == "regression" else "MLPClassifier"
            )
            lines.extend(
                [
                    "# Model",
                    f"model = {model_class}(",
                    f"    hidden_layer_sizes={tuple(hidden_sizes)},",
                    f"    activation='{config.layers[0].activation.value}',",
                    f"    solver='{config.optimizer.value}',",
                    f"    learning_rate_init={config.learning_rate},",
                    f"    max_iter={config.epochs},",
                    f"    batch_size={config.batch_size},",
                    "    early_stopping=True,",
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
                ]
            )

        if include_evaluation:
            lines.extend(
                [
                    "",
                    "# Evaluation",
                    "predictions = model.predict(X_test)",
                    "mse = mean_squared_error(y_test, predictions)",
                    "r2 = r2_score(y_test, predictions)",
                    "print(f'Test MSE: {mse:.4f}')",
                    "print(f'Test R2: {r2:.4f}')",
                ]
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Framework-specific conversion helpers
    # ------------------------------------------------------------------ #

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
