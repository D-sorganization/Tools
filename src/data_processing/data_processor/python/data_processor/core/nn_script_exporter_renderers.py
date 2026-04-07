"""Framework-specific renderers for neural network training scripts."""

from __future__ import annotations

from datetime import datetime

from .nn_architecture import (
    ActivationFunction,
    Framework,
    LossFunction,
    NetworkConfig,
    Optimizer,
)

DEFAULT_DATA_PATH = "data.csv"


def build_framework_script(
    config: NetworkConfig,
    framework: Framework,
    data_path: str | None,
    include_data_loading: bool,
    include_training: bool,
    include_evaluation: bool,
) -> str:
    """Dispatch framework-specific script generation."""
    if framework == Framework.PYTORCH:
        return generate_pytorch_script(
            config,
            data_path,
            include_data_loading,
            include_training,
            include_evaluation,
        )
    if framework == Framework.TENSORFLOW:
        return generate_tensorflow_script(
            config,
            data_path,
            include_data_loading,
            include_training,
            include_evaluation,
        )
    return generate_sklearn_script(
        config,
        data_path,
        include_data_loading,
        include_training,
        include_evaluation,
    )


def generate_pytorch_script(
    config: NetworkConfig,
    data_path: str | None,
    include_data_loading: bool,
    include_training: bool,
    include_evaluation: bool,
) -> str:
    """Generate a PyTorch training script."""
    lines = pytorch_header(config)
    lines.extend(pytorch_model_definition(config))

    if include_data_loading:
        lines.extend(pytorch_data_loading(config, data_path))
    if include_training:
        lines.extend(pytorch_training(config))
    if include_evaluation:
        lines.extend(pytorch_evaluation())

    return "\n".join(lines)


def pytorch_header(config: NetworkConfig) -> list[str]:
    """Generate a PyTorch header block."""
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


def pytorch_model_definition(config: NetworkConfig) -> list[str]:
    """Generate a PyTorch model definition block."""
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
            activation = pytorch_activation(layer.activation)
            lines.append(f"        layers.append(nn.Linear(prev_size, {layer.units}))")
            if activation:
                lines.append(f"        layers.append({activation})")
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


def pytorch_data_loading(config: NetworkConfig, data_path: str | None) -> list[str]:
    """Generate PyTorch data loading and normalization code."""
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
        "X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]",
        "X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]",
        "",
    ]

    if config.normalize_inputs:
        lines.extend(pytorch_normalization_block())

    lines.extend(
        [
            "# Create DataLoaders",
            "train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))",
            "val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))",
            f"train_loader = DataLoader(train_dataset, batch_size={config.batch_size}, shuffle=True)",
            f"val_loader = DataLoader(val_dataset, batch_size={config.batch_size})",
            "",
        ]
    )
    return lines


def pytorch_normalization_block() -> list[str]:
    """Generate the PyTorch normalization block."""
    return [
        "# Normalization",
        "X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)",
        "X_std[X_std == 0] = 1",
        "X_train = (X_train - X_mean) / X_std",
        "X_val = (X_val - X_mean) / X_std",
        "X_test = (X_test - X_mean) / X_std",
        "",
    ]


def pytorch_training(config: NetworkConfig) -> list[str]:
    """Generate PyTorch training setup and loop blocks."""
    optimizer_name = pytorch_optimizer(config.optimizer)
    loss_name = pytorch_loss(config.loss_function)
    lines = pytorch_training_setup(config, optimizer_name, loss_name)
    lines.extend(pytorch_training_loop(config))
    return lines


def pytorch_training_setup(
    config: NetworkConfig, optimizer_name: str, loss_name: str
) -> list[str]:
    """Generate PyTorch training setup code."""
    return [
        "# Training Setup",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        f"model = NeuralNetwork(input_size={config.input_features}, output_size={config.output_features}).to(device)",
        f"optimizer = {optimizer_name}(model.parameters(), lr={config.learning_rate})",
        f"criterion = {loss_name}()",
        "",
        "# Early stopping",
        "best_val_loss = float('inf')",
        "patience_counter = 0",
        f"patience = {config.early_stopping_patience}",
        "",
    ]


def pytorch_training_loop(config: NetworkConfig) -> list[str]:
    """Generate the PyTorch epoch loop."""
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
    ]


def pytorch_evaluation() -> list[str]:
    """Generate PyTorch evaluation code."""
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


def generate_tensorflow_script(
    config: NetworkConfig,
    data_path: str | None,
    include_data_loading: bool,
    include_training: bool,
    include_evaluation: bool,
) -> str:
    """Generate a TensorFlow / Keras training script."""
    lines = tensorflow_header()
    lines.extend(tensorflow_model_definition(config))

    if include_data_loading:
        lines.extend(tensorflow_data_loading(config, data_path))
    if include_training:
        lines.extend(tensorflow_training(config))
    if include_evaluation:
        lines.extend(tensorflow_evaluation())

    return "\n".join(lines)


def tensorflow_header() -> list[str]:
    """Generate a TensorFlow header block."""
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


def tensorflow_model_definition(config: NetworkConfig) -> list[str]:
    """Generate a Keras sequential model definition block."""
    lines = [
        "# Model Definition",
        "def create_model(input_size, output_size):",
        "    model = keras.Sequential([",
        "        layers.Input(shape=(input_size,)),",
    ]

    for layer in config.layers[:-1]:
        if layer.layer_type == "dense":
            lines.append(
                f"        layers.Dense({layer.units}, activation='{layer.activation.value}'),"
            )
        elif layer.layer_type == "dropout":
            lines.append(f"        layers.Dropout({layer.dropout_rate}),")

    output_layer = config.layers[-1]
    lines.extend(
        [
            f"        layers.Dense(output_size, activation='{output_layer.activation.value}'),",
            "    ])",
            "    return model",
            "",
        ]
    )
    return lines


def tensorflow_data_loading(config: NetworkConfig, data_path: str | None) -> list[str]:
    """Generate TensorFlow data loading code."""
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
        "X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]",
        "X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]",
        "",
    ]


def tensorflow_training(config: NetworkConfig) -> list[str]:
    """Generate TensorFlow compile, callback, and fit code."""
    return [
        "# Create and compile model",
        f"model = create_model({config.input_features}, {config.output_features})",
        "model.compile(",
        f"    optimizer='{config.optimizer.value}',",
        f"    loss='{keras_loss(config.loss_function)}',",
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


def tensorflow_evaluation() -> list[str]:
    """Generate TensorFlow evaluation code."""
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


def generate_sklearn_script(
    config: NetworkConfig,
    data_path: str | None,
    include_data_loading: bool,
    include_training: bool,
    include_evaluation: bool,
) -> str:
    """Generate a scikit-learn training script."""
    lines = sklearn_header()
    hidden_sizes = [
        layer_config.units
        for layer_config in config.layers
        if layer_config.layer_type == "dense"
    ][:-1]

    if include_data_loading:
        lines.extend(sklearn_data_loading(config, data_path))
    if include_training:
        lines.extend(sklearn_training(config, hidden_sizes))
    if include_evaluation:
        lines.extend(sklearn_evaluation())

    return "\n".join(lines)


def sklearn_header() -> list[str]:
    """Generate a scikit-learn header block."""
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


def sklearn_data_loading(config: NetworkConfig, data_path: str | None) -> list[str]:
    """Generate scikit-learn data loading and split code."""
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
        f"    X, y, test_size={config.validation_split + 0.15}, random_state=42",
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


def sklearn_training(config: NetworkConfig, hidden_sizes: list[int]) -> list[str]:
    """Generate scikit-learn model creation and fit code."""
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


def sklearn_evaluation() -> list[str]:
    """Generate scikit-learn evaluation code."""
    return [
        "",
        "# Evaluation",
        "predictions = model.predict(X_test)",
        "mse = mean_squared_error(y_test, predictions)",
        "r2 = r2_score(y_test, predictions)",
        "print(f'Test MSE: {mse:.4f}')",
        "print(f'Test R2: {r2:.4f}')",
    ]


def pytorch_activation(activation: ActivationFunction) -> str:
    """Convert an activation to a PyTorch module expression."""
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


def pytorch_optimizer(optimizer: Optimizer) -> str:
    """Convert an optimizer to a PyTorch optimizer class name."""
    mapping = {
        Optimizer.SGD: "optim.SGD",
        Optimizer.ADAM: "optim.Adam",
        Optimizer.ADAMW: "optim.AdamW",
        Optimizer.RMSPROP: "optim.RMSprop",
        Optimizer.ADAGRAD: "optim.Adagrad",
    }
    return mapping.get(optimizer, "optim.Adam")


def pytorch_loss(loss: LossFunction) -> str:
    """Convert a loss function to a PyTorch criterion."""
    mapping = {
        LossFunction.MSE: "nn.MSELoss",
        LossFunction.MAE: "nn.L1Loss",
        LossFunction.HUBER: "nn.SmoothL1Loss",
        LossFunction.CROSS_ENTROPY: "nn.CrossEntropyLoss",
        LossFunction.BINARY_CROSS_ENTROPY: "nn.BCELoss",
    }
    return mapping.get(loss, "nn.MSELoss")


def keras_loss(loss: LossFunction) -> str:
    """Convert a loss function to a Keras loss name."""
    mapping = {
        LossFunction.MSE: "mse",
        LossFunction.MAE: "mae",
        LossFunction.HUBER: "huber",
        LossFunction.CROSS_ENTROPY: "categorical_crossentropy",
        LossFunction.BINARY_CROSS_ENTROPY: "binary_crossentropy",
    }
    return mapping.get(loss, "mse")
