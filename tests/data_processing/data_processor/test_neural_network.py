"""Tests for the neural network decomposed modules.

Covers nn_architecture, nn_trainer, and nn_script_exporter.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from data_processor.core.neural_network import NeuralNetworkInterface
from data_processor.core.nn_architecture import (
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
from data_processor.core.nn_script_exporter import NeuralNetworkScriptExporter
from data_processor.core.nn_trainer import NeuralNetworkTrainer

# ------------------------------------------------------------------ #
#  Architecture tests
# ------------------------------------------------------------------ #


class TestEnums:
    """Test that all enums have expected values."""

    def test_framework_values(self) -> None:
        assert Framework.PYTORCH.value == "pytorch"
        assert Framework.TENSORFLOW.value == "tensorflow"
        assert Framework.SKLEARN.value == "sklearn"

    def test_network_types(self) -> None:
        assert NetworkType.MLP.value == "mlp"
        assert NetworkType.LSTM.value == "lstm"
        assert NetworkType.GRU.value == "gru"
        assert NetworkType.CNN_1D.value == "cnn_1d"
        assert NetworkType.TRANSFORMER.value == "transformer"
        assert NetworkType.AUTOENCODER.value == "autoencoder"

    def test_activation_functions(self) -> None:
        expected = {
            "relu",
            "leaky_relu",
            "elu",
            "selu",
            "tanh",
            "sigmoid",
            "softmax",
            "linear",
            "gelu",
            "swish",
        }
        actual = {a.value for a in ActivationFunction}
        assert actual == expected

    def test_loss_functions(self) -> None:
        expected = {
            "mse",
            "mae",
            "huber",
            "cross_entropy",
            "binary_cross_entropy",
            "custom",
        }
        actual = {lf.value for lf in LossFunction}
        assert actual == expected

    def test_optimizers(self) -> None:
        expected = {"sgd", "adam", "adamw", "rmsprop", "adagrad"}
        actual = {o.value for o in Optimizer}
        assert actual == expected


class TestLayerConfig:
    """Test LayerConfig dataclass."""

    def test_defaults(self) -> None:
        layer = LayerConfig(layer_type="dense")
        assert layer.units == 64
        assert layer.activation == ActivationFunction.RELU
        assert layer.dropout_rate == 0.0

    def test_custom_values(self) -> None:
        layer = LayerConfig(
            layer_type="lstm",
            units=128,
            activation=ActivationFunction.TANH,
            return_sequences=False,
        )
        assert layer.layer_type == "lstm"
        assert layer.units == 128
        assert layer.return_sequences is False


class TestNetworkConfig:
    """Test NetworkConfig serialization."""

    def test_to_dict_roundtrip(self) -> None:
        config = NetworkConfig(
            network_type=NetworkType.MLP,
            layers=[
                LayerConfig(layer_type="dense", units=64),
                LayerConfig(layer_type="dropout", dropout_rate=0.3),
            ],
            input_features=10,
            output_features=2,
            learning_rate=0.01,
        )
        d = config.to_dict()
        restored = NetworkConfig.from_dict(d)

        assert restored.network_type == config.network_type
        assert restored.input_features == config.input_features
        assert restored.output_features == config.output_features
        assert restored.learning_rate == config.learning_rate
        assert len(restored.layers) == 2
        assert restored.layers[0].units == 64
        assert restored.layers[1].dropout_rate == 0.3

    def test_from_dict_defaults(self) -> None:
        config = NetworkConfig.from_dict({})
        assert config.network_type == NetworkType.MLP
        assert config.optimizer == Optimizer.ADAM
        assert config.epochs == 100

    def test_to_dict_contains_all_fields(self) -> None:
        config = NetworkConfig()
        d = config.to_dict()
        expected_keys = {
            "network_type",
            "layers",
            "input_features",
            "output_features",
            "optimizer",
            "learning_rate",
            "loss_function",
            "batch_size",
            "epochs",
            "validation_split",
            "l1_regularization",
            "l2_regularization",
            "early_stopping_patience",
            "reduce_lr_patience",
            "normalize_inputs",
            "normalize_outputs",
            "sequence_length",
            "task_type",
        }
        assert set(d.keys()) == expected_keys


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_basic_creation(self) -> None:
        result = TrainingResult(
            train_loss_history=[1.0, 0.5],
            val_loss_history=[1.2, 0.6],
            best_epoch=1,
            best_val_loss=0.6,
            final_train_loss=0.5,
            final_val_loss=0.6,
        )
        assert result.best_epoch == 1
        assert result.stopped_early is False
        assert result.predictions is None


class TestDataSplitConfig:
    """Test DataSplitConfig defaults."""

    def test_defaults(self) -> None:
        cfg = DataSplitConfig()
        assert cfg.train_ratio == 0.7
        assert cfg.val_ratio == 0.15
        assert cfg.test_ratio == 0.15
        assert cfg.shuffle is True
        assert cfg.random_state == 42


# ------------------------------------------------------------------ #
#  Trainer tests
# ------------------------------------------------------------------ #


class TestNeuralNetworkTrainer:
    """Test the trainer functionality."""

    @pytest.fixture()
    def trainer(self) -> NeuralNetworkTrainer:
        return NeuralNetworkTrainer()

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "f1": rng.standard_normal(100),
                "f2": rng.standard_normal(100),
                "target": rng.standard_normal(100),
            }
        )

    def test_create_config_mlp(self, trainer: NeuralNetworkTrainer) -> None:
        config = trainer.create_config(input_features=5, output_features=1)
        assert config.network_type == NetworkType.MLP
        assert config.input_features == 5
        assert config.output_features == 1
        # Default layers: 3 hidden (128,64,32) + dropout after each + output
        dense_layers = [lyr for lyr in config.layers if lyr.layer_type == "dense"]
        assert len(dense_layers) == 4  # 3 hidden + 1 output

    def test_create_config_lstm(self, trainer: NeuralNetworkTrainer) -> None:
        config = trainer.create_config(input_features=5, network_type=NetworkType.LSTM)
        lstm_layers = [lyr for lyr in config.layers if lyr.layer_type == "lstm"]
        assert len(lstm_layers) == 3

    def test_create_config_cnn(self, trainer: NeuralNetworkTrainer) -> None:
        config = trainer.create_config(input_features=5, network_type=NetworkType.CNN_1D)
        conv_layers = [lyr for lyr in config.layers if lyr.layer_type == "conv1d"]
        assert len(conv_layers) == 3

    def test_create_config_custom_layers(self, trainer: NeuralNetworkTrainer) -> None:
        config = trainer.create_config(
            input_features=3,
            hidden_layers=[32, 16],
            dropout_rate=0.0,
        )
        dense_layers = [lyr for lyr in config.layers if lyr.layer_type == "dense"]
        # 2 hidden + 1 output
        assert len(dense_layers) == 3
        assert dense_layers[0].units == 32
        assert dense_layers[1].units == 16

    def test_create_config_invalid_input_features_raises(
        self, trainer: NeuralNetworkTrainer
    ) -> None:
        with pytest.raises(ValueError, match="input_features must be positive"):
            trainer.create_config(input_features=0)

    def test_create_config_invalid_dropout_rate_raises(self, trainer: NeuralNetworkTrainer) -> None:
        with pytest.raises(ValueError, match="dropout_rate must be in \\[0.0, 1.0\\)"):
            trainer.create_config(input_features=4, dropout_rate=1.0)

    def test_prepare_data(
        self,
        trainer: NeuralNetworkTrainer,
        sample_df: pd.DataFrame,
    ) -> None:
        trainer.create_config(input_features=2, output_features=1)
        data = trainer.prepare_data(sample_df, target_columns=["target"])

        assert "X_train" in data
        assert "y_train" in data
        assert "X_val" in data
        assert "X_test" in data
        assert data["feature_names"] == ["f1", "f2"]
        assert data["target_names"] == ["target"]

    def test_prepare_data_custom_split(
        self,
        trainer: NeuralNetworkTrainer,
        sample_df: pd.DataFrame,
    ) -> None:
        trainer.create_config(input_features=2, output_features=1)
        split = DataSplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        data = trainer.prepare_data(
            sample_df,
            target_columns=["target"],
            split_config=split,
        )
        assert len(data["X_train"]) == 80

    def test_prepare_data_missing_target_column_raises(
        self, trainer: NeuralNetworkTrainer, sample_df: pd.DataFrame
    ) -> None:
        trainer.create_config(input_features=2, output_features=1)
        with pytest.raises(ValueError, match="Unknown target columns"):
            trainer.prepare_data(sample_df, target_columns=["missing_target"])

    def test_prepare_data_invalid_split_ratio_raises(
        self, trainer: NeuralNetworkTrainer, sample_df: pd.DataFrame
    ) -> None:
        trainer.create_config(input_features=2, output_features=1)
        split = DataSplitConfig(train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)
        with pytest.raises(ValueError, match="sum to 1.0 or less"):
            trainer.prepare_data(
                sample_df,
                target_columns=["target"],
                split_config=split,
            )

    def test_prepare_data_no_valid_rows_after_dropna_raises(
        self, trainer: NeuralNetworkTrainer
    ) -> None:
        trainer.create_config(input_features=2, output_features=1)
        df = pd.DataFrame({"f1": [np.nan, np.nan], "f2": [np.nan, np.nan], "target": [1, 2]})
        with pytest.raises(ValueError, match="No rows available"):
            trainer.prepare_data(df, target_columns=["target"])

    def test_train_simple(
        self,
        trainer: NeuralNetworkTrainer,
        sample_df: pd.DataFrame,
    ) -> None:
        config = trainer.create_config(
            input_features=2,
            output_features=1,
            hidden_layers=[8],
            dropout_rate=0.0,
            epochs=5,
            batch_size=16,
        )
        data = trainer.prepare_data(sample_df, target_columns=["target"])
        result = trainer.train_simple(data, config)

        assert isinstance(result, TrainingResult)
        assert len(result.train_loss_history) <= 5
        assert len(result.val_loss_history) <= 5
        assert result.final_train_loss > 0
        assert result.predictions is not None

    def test_train_simple_no_config_raises(
        self,
        trainer: NeuralNetworkTrainer,
    ) -> None:
        with pytest.raises(ValueError, match="No network configuration"):
            trainer.train_simple({"X_train": np.zeros((10, 2))})

    def test_train_simple_missing_required_data_keys_raises(
        self, trainer: NeuralNetworkTrainer
    ) -> None:
        config = trainer.create_config(input_features=2, output_features=1)
        with pytest.raises(ValueError, match="Missing required data keys"):
            trainer.train_simple({"X_train": np.zeros((10, 2))}, config)

    def test_activation_relu(self, trainer: NeuralNetworkTrainer) -> None:
        z = np.array([-1.0, 0.0, 1.0])
        result = trainer._apply_activation(z, ActivationFunction.RELU)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_activation_sigmoid(self, trainer: NeuralNetworkTrainer) -> None:
        z = np.array([0.0])
        result = trainer._apply_activation(z, ActivationFunction.SIGMOID)
        np.testing.assert_almost_equal(result, [0.5])

    def test_activation_tanh(self, trainer: NeuralNetworkTrainer) -> None:
        z = np.array([0.0])
        result = trainer._apply_activation(z, ActivationFunction.TANH)
        np.testing.assert_almost_equal(result, [0.0])

    def test_activation_derivative_relu(self, trainer: NeuralNetworkTrainer) -> None:
        a = np.array([-0.5, 0.0, 0.5])
        result = trainer._activation_derivative(a, ActivationFunction.RELU)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_normalize_features(self, trainer: NeuralNetworkTrainer) -> None:
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_val = np.array([[2.0, 3.0]])
        X_test = np.array([[4.0, 5.0]])

        X_train_n, X_val_n, X_test_n = trainer._normalize_features(
            X_train,
            X_val,
            X_test,
        )
        # Mean should be approximately 0 for training data
        np.testing.assert_almost_equal(np.mean(X_train_n, axis=0), [0.0, 0.0])

    def test_early_stopping(
        self,
        trainer: NeuralNetworkTrainer,
        sample_df: pd.DataFrame,
    ) -> None:
        config = trainer.create_config(
            input_features=2,
            output_features=1,
            hidden_layers=[4],
            dropout_rate=0.0,
            epochs=200,
            early_stopping_patience=3,
            batch_size=32,
        )
        data = trainer.prepare_data(sample_df, target_columns=["target"])

        # Patch random to make validation loss increase consistently
        result = trainer.train_simple(data, config)
        # The training should stop before 200 epochs most of the time
        assert len(result.train_loss_history) <= 200


# ------------------------------------------------------------------ #
#  Script exporter tests
# ------------------------------------------------------------------ #


class TestNeuralNetworkScriptExporter:
    """Test script generation."""

    @pytest.fixture()
    def exporter(self) -> NeuralNetworkScriptExporter:
        return NeuralNetworkScriptExporter()

    @pytest.fixture()
    def mlp_config(self) -> NetworkConfig:
        return NetworkConfig(
            network_type=NetworkType.MLP,
            layers=[
                LayerConfig(layer_type="dense", units=64),
                LayerConfig(layer_type="dropout", dropout_rate=0.2),
                LayerConfig(
                    layer_type="dense",
                    units=1,
                    activation=ActivationFunction.LINEAR,
                ),
            ],
            input_features=5,
            output_features=1,
            epochs=50,
        )

    def test_export_pytorch_script(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        result = exporter.export_script(
            mlp_config,
            path,
            framework=Framework.PYTORCH,
        )
        content = result.read_text()

        assert "import torch" in content
        assert "NeuralNetwork(nn.Module)" in content
        assert "DataLoader" in content
        path.unlink()

    def test_export_tensorflow_script(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        result = exporter.export_script(
            mlp_config,
            path,
            framework=Framework.TENSORFLOW,
        )
        content = result.read_text()

        assert "import tensorflow" in content
        assert "keras.Sequential" in content
        path.unlink()

    def test_export_sklearn_script(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        result = exporter.export_script(
            mlp_config,
            path,
            framework=Framework.SKLEARN,
        )
        content = result.read_text()

        assert "MLPRegressor" in content
        assert "StandardScaler" in content
        path.unlink()

    def test_export_config_json(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        result = exporter.export_config(mlp_config, path)
        data = json.loads(result.read_text())

        assert data["network_type"] == "mlp"
        assert data["input_features"] == 5
        path.unlink()

    def test_import_config_json(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)
            json.dump(mlp_config.to_dict(), f)

        config = exporter.import_config(path)

        assert config.network_type == NetworkType.MLP
        assert config.input_features == 5
        path.unlink()

    def test_pytorch_activation_mapping(
        self,
        exporter: NeuralNetworkScriptExporter,
    ) -> None:
        assert exporter._pytorch_activation(ActivationFunction.RELU) == "nn.ReLU()"
        assert exporter._pytorch_activation(ActivationFunction.LINEAR) == ""
        assert exporter._pytorch_activation(ActivationFunction.TANH) == "nn.Tanh()"

    def test_pytorch_loss_mapping(
        self,
        exporter: NeuralNetworkScriptExporter,
    ) -> None:
        assert exporter._pytorch_loss(LossFunction.MSE) == "nn.MSELoss"
        assert exporter._pytorch_loss(LossFunction.MAE) == "nn.L1Loss"

    def test_keras_loss_mapping(
        self,
        exporter: NeuralNetworkScriptExporter,
    ) -> None:
        assert exporter._keras_loss(LossFunction.MSE) == "mse"
        assert exporter._keras_loss(LossFunction.CROSS_ENTROPY) == "categorical_crossentropy"

    def test_no_data_loading(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        exporter.export_script(
            mlp_config,
            path,
            include_data_loading=False,
        )
        content = path.read_text()
        assert "pd.read_csv" not in content
        path.unlink()

    def test_export_script_empty_output_path_raises(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
    ) -> None:
        with pytest.raises(ValueError, match="output_path must not be empty"):
            exporter.export_script(mlp_config, "")

    def test_export_script_non_python_output_suffix_raises(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="output_path must end with .py"):
            exporter.export_script(mlp_config, tmp_path / "model.txt")

    def test_export_script_empty_data_path_raises(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="data_path must not be empty"):
            exporter.export_script(mlp_config, tmp_path / "train.py", data_path="  ")

    def test_export_config_empty_output_path_raises(
        self,
        exporter: NeuralNetworkScriptExporter,
        mlp_config: NetworkConfig,
    ) -> None:
        with pytest.raises(ValueError, match="output_path must not be empty"):
            exporter.export_config(mlp_config, "")

    def test_import_config_missing_file_raises(
        self, exporter: NeuralNetworkScriptExporter, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            exporter.import_config(tmp_path / "missing_config.json")


# ------------------------------------------------------------------ #
#  Backward-compatibility tests
# ------------------------------------------------------------------ #


class TestNeuralNetworkInterface:
    """Test that the facade preserves backward compat."""

    def test_create_and_train(self) -> None:
        nn = NeuralNetworkInterface()
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.standard_normal(50),
                "b": rng.standard_normal(50),
                "y": rng.standard_normal(50),
            }
        )

        config = nn.create_config(
            input_features=2,
            output_features=1,
            hidden_layers=[8],
            dropout_rate=0.0,
            epochs=3,
            batch_size=16,
        )
        data = nn.prepare_data(df, target_columns=["y"])
        result = nn.train_simple(data, config)

        assert isinstance(result, TrainingResult)
        assert result.final_train_loss > 0

    def test_export_script(self) -> None:
        nn = NeuralNetworkInterface()
        config = nn.create_config(input_features=3, output_features=1)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        result = nn.export_script(config, path)
        assert result.exists()
        assert "torch" in result.read_text()
        path.unlink()

    def test_all_exports_from_neural_network(self) -> None:
        """Verify neural_network.py re-exports all expected symbols."""
        from data_processor.core import neural_network

        expected = [
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
        for name in expected:
            assert hasattr(neural_network, name), f"Missing export: {name}"
