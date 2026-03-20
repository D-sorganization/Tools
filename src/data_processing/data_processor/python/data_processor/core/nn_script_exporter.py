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
DEFAULT_DATA_PATH = "data.csv"


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
        """Export training script for a specific framework."""
        assert config is not None, "config must be provided"
        output_path = self._validate_script_output_path(output_path)
        validated_data_path = self._validate_data_path(data_path)
        script = self._build_framework_script(
            config=config,
            framework=framework,
            data_path=validated_data_path,
            include_data_loading=include_data_loading,
            include_training=include_training,
            include_evaluation=include_evaluation,
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
        assert config is not None, "config must be provided"
        output_path = self._validate_config_output_path(output_path)
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
        path = Path(config_path)
        if not path.exists():
            msg = f"Configuration file does not exist: {path}"
            raise FileNotFoundError(msg)
        with open(path) as f:
            data = json.load(f)

        if "normalization_params" in data:
            self._normalization_params = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in data.pop("normalization_params").items()
            }

        return NetworkConfig.from_dict(data)

    def _validate_script_output_path(self, output_path: Path | str) -> Path:
        """Validate script output path preconditions."""
        output = Path(output_path)
        if not str(output).strip() or str(output) == ".":
            raise ValueError("output_path must not be empty")
        if output.suffix.lower() != ".py":
            raise ValueError("output_path must end with .py")
        return output

    def _validate_config_output_path(self, output_path: Path | str) -> Path:
        """Validate config output path preconditions."""
        output = Path(output_path)
        if not str(output).strip() or str(output) == ".":
            raise ValueError("output_path must not be empty")
        return output

    def _validate_data_path(self, data_path: str | None) -> str | None:
        """Validate optional data path argument."""
        if data_path is None:
            return None
        if not data_path.strip():
            raise ValueError("data_path must not be empty")
        return data_path

    def _build_framework_script(
        self,
        config: NetworkConfig,
        framework: Framework,
        data_path: str | None,
        include_data_loading: bool,
        include_training: bool,
        include_evaluation: bool,
    ) -> str:
        """Dispatch framework-specific script generation."""
        assert config is not None, "config must be provided"
        if framework == Framework.PYTORCH:
            return self._generate_pytorch_script(
                config,
                data_path,
                include_data_loading,
                include_training,
                include_evaluation,
            )
        if framework == Framework.TENSORFLOW:
            return self._generate_tensorflow_script(
                config,
                data_path,
                include_data_loading,
                include_training,
                include_evaluation,
            )
        return self._generate_sklearn_script(
            config,
            data_path,
            include_data_loading,
            include_training,
            include_evaluation,
        )

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
        assert config is not None, "config must be provided"
        lines = self._pytorch_header(config)
        lines.extend(self._pytorch_model_definition(config))

        if include_data_loading:
            lines.extend(self._pytorch_data_loading(config, data_path))
        if include_training:
            lines.extend(self._pytorch_training(config))
        if include_evaluation:
            lines.extend(self._pytorch_evaluation())

        return "\n".join(lines)

    def _pytorch_header(self, config: NetworkConfig) -> list[str]:
        """Generate PyTorch script header with imports."""
        return [
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

    def _pytorch_model_definition(self, config: NetworkConfig) -> list[str]:
        """Generate PyTorch nn.Module class definition."""
        assert config is not None, "config must be provided"
        lines = [
            "# Model Definition",
            "class NeuralNetwork(nn.Module):",
            "    def __init__(self, input_size, output_size):",
            "        super(NeuralNetwork, self).__init__()",
            "        layers = []",
            "        prev_size = input_size",
        ]

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
        return lines

    def _pytorch_data_loading(
        self, config: NetworkConfig, data_path: str | None
    ) -> list[str]:
        """Generate PyTorch data loading and DataLoader creation."""
        assert config is not None, "config must be provided"
        data_path_str = data_path or DEFAULT_DATA_PATH
        train_fraction = 1 - config.validation_split - 0.15
        lines = [
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
            f"train_size = int(len(X) * {train_fraction})",
            f"val_size = int(len(X) * {config.validation_split})",
            "",
            "X_train, y_train = X[:train_size], y[:train_size]",
            "X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]",  # noqa: E501
            "X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]",
            "",
        ]

        if config.normalize_inputs:
            lines.extend(self._pytorch_normalization_block())

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
        return lines

    def _pytorch_normalization_block(self) -> list[str]:
        """Generate PyTorch input normalization block."""
        return [
            "# Normalization",
            "X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)",
            "X_std[X_std == 0] = 1",
            "X_train = (X_train - X_mean) / X_std",
            "X_val = (X_val - X_mean) / X_std",
            "X_test = (X_test - X_mean) / X_std",
            "",
        ]

    def _pytorch_training(self, config: NetworkConfig) -> list[str]:
        """Generate PyTorch training loop with early stopping."""
        assert config is not None, "config must be provided"
        optimizer_name = self._pytorch_optimizer(config.optimizer)
        loss_name = self._pytorch_loss(config.loss_function)
        lines = self._pytorch_training_setup(config, optimizer_name, loss_name)
        lines.extend(self._pytorch_training_loop(config))
        return lines

    def _pytorch_training_setup(
        self, config: NetworkConfig, optimizer_name: str, loss_name: str
    ) -> list[str]:
        """Generate PyTorch setup block."""
        return [
            "# Training Setup",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",  # noqa: E501
            f"model = NeuralNetwork(input_size={config.input_features}, output_size={config.output_features}).to(device)",  # noqa: E501
            (
                "optimizer = "
                f"{optimizer_name}(model.parameters(), lr={config.learning_rate})"
            ),
            f"criterion = {loss_name}()",
            "",
            "# Early stopping",
            "best_val_loss = float('inf')",
            "patience_counter = 0",
            f"patience = {config.early_stopping_patience}",
            "",
        ]

    def _pytorch_training_loop(self, config: NetworkConfig) -> list[str]:
        """Generate PyTorch epoch loop block."""
        return [
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

    def _pytorch_evaluation(self) -> list[str]:
        """Generate PyTorch model evaluation code."""
        return [
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
        assert config is not None, "config must be provided"
        lines = self._tensorflow_header()
        lines.extend(self._tensorflow_model_definition(config))

        if include_data_loading:
            lines.extend(self._tensorflow_data_loading(config, data_path))
        if include_training:
            lines.extend(self._tensorflow_training(config))
        if include_evaluation:
            lines.extend(self._tensorflow_evaluation())

        return "\n".join(lines)

    def _tensorflow_header(self) -> list[str]:
        """Generate TensorFlow script header with imports."""
        return [
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

    def _tensorflow_model_definition(self, config: NetworkConfig) -> list[str]:
        """Generate Keras Sequential model definition."""
        assert config is not None, "config must be provided"
        lines = [
            "# Model Definition",
            "def create_model(input_size, output_size):",
            "    model = keras.Sequential([",
            "        layers.Input(shape=(input_size,)),",
        ]

        for layer in config.layers[:-1]:  # Skip last layer
            if layer.layer_type == "dense":
                act = layer.activation.value
                lines.append(
                    f"        layers.Dense({layer.units}, activation='{act}'),"
                )
            elif layer.layer_type == "dropout":
                lines.append(f"        layers.Dropout({layer.dropout_rate}),")

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
        return lines

    def _tensorflow_data_loading(
        self, config: NetworkConfig, data_path: str | None
    ) -> list[str]:
        """Generate TensorFlow data loading code."""
        assert config is not None, "config must be provided"
        data_path_str = data_path or DEFAULT_DATA_PATH
        train_fraction = 1 - config.validation_split - 0.15
        return [
            "# Data Loading",
            f'data = pd.read_csv("{data_path_str}")',
            "feature_cols = []  # Fill in feature column names",
            "target_cols = []   # Fill in target column names",
            "",
            "X = data[feature_cols].values.astype(np.float32)",
            "y = data[target_cols].values.astype(np.float32)",
            "",
            f"train_size = int(len(X) * {train_fraction})",
            f"val_size = int(len(X) * {config.validation_split})",
            "",
            "X_train, y_train = X[:train_size], y[:train_size]",
            "X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]",  # noqa: E501
            "X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]",
            "",
        ]

    def _tensorflow_training(self, config: NetworkConfig) -> list[str]:
        """Generate TensorFlow model compilation, callbacks, and training."""
        assert config is not None, "config must be provided"
        opt_name = config.optimizer.value
        loss_name = self._keras_loss(config.loss_function)

        return [
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

    def _tensorflow_evaluation(self) -> list[str]:
        """Generate TensorFlow model evaluation code."""
        return [
            "",
            "# Evaluation",
            "results = model.evaluate(X_test, y_test)",
            "print(f'Test Loss: {results[0]:.4f}')",
            "print(f'Test MAE: {results[1]:.4f}')",
            "",
            "# Predictions",
            "predictions = model.predict(X_test)",
        ]

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
        assert config is not None, "config must be provided"
        lines = self._sklearn_header()

        hidden_sizes = [
            layer_cfg.units
            for layer_cfg in config.layers
            if layer_cfg.layer_type == "dense"
        ][:-1]

        if include_data_loading:
            lines.extend(self._sklearn_data_loading(config, data_path))
        if include_training:
            lines.extend(self._sklearn_training(config, hidden_sizes))
        if include_evaluation:
            lines.extend(self._sklearn_evaluation())

        return "\n".join(lines)

    def _sklearn_header(self) -> list[str]:
        """Generate scikit-learn script header with imports."""
        return [
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

    def _sklearn_data_loading(
        self, config: NetworkConfig, data_path: str | None
    ) -> list[str]:
        """Generate scikit-learn data loading and split code."""
        assert config is not None, "config must be provided"
        data_path_str = data_path or DEFAULT_DATA_PATH
        lines = [
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
        return lines

    def _sklearn_training(
        self, config: NetworkConfig, hidden_sizes: list[int]
    ) -> list[str]:
        """Generate scikit-learn MLP model creation and training."""
        assert config is not None, "config must be provided"
        model_class = (
            "MLPRegressor" if config.task_type == "regression" else "MLPClassifier"
        )
        return [
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

    def _sklearn_evaluation(self) -> list[str]:
        """Generate scikit-learn model evaluation code."""
        return [
            "",
            "# Evaluation",
            "predictions = model.predict(X_test)",
            "mse = mean_squared_error(y_test, predictions)",
            "r2 = r2_score(y_test, predictions)",
            "print(f'Test MSE: {mse:.4f}')",
            "print(f'Test R2: {r2:.4f}')",
        ]

    # ------------------------------------------------------------------ #
    #  Framework-specific conversion helpers
    # ------------------------------------------------------------------ #

    def _pytorch_activation(self, activation: ActivationFunction) -> str:
        """Convert activation to PyTorch module string."""
        assert activation is not None, "activation must be provided"
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
        assert optimizer is not None, "optimizer must be provided"
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
        assert loss is not None, "loss must be provided"
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
        assert loss is not None, "loss must be provided"
        mapping = {
            LossFunction.MSE: "mse",
            LossFunction.MAE: "mae",
            LossFunction.HUBER: "huber",
            LossFunction.CROSS_ENTROPY: "categorical_crossentropy",
            LossFunction.BINARY_CROSS_ENTROPY: "binary_crossentropy",
        }
        return mapping.get(loss, "mse")
