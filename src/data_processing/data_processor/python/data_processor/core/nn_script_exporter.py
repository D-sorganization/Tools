"""Facade for neural network script export and configuration IO."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .nn_architecture import (
    ActivationFunction,
    Framework,
    LossFunction,
    NetworkConfig,
    Optimizer,
)
from .nn_script_exporter_io import (
    export_config_file,
    import_config_file,
    validate_config_output_path,
    validate_data_path,
    validate_script_output_path,
)
from .nn_script_exporter_renderers import (
    build_framework_script,
    generate_pytorch_script,
    generate_sklearn_script,
    generate_tensorflow_script,
    keras_loss,
    pytorch_activation,
    pytorch_data_loading,
    pytorch_evaluation,
    pytorch_header,
    pytorch_loss,
    pytorch_model_definition,
    pytorch_normalization_block,
    pytorch_optimizer,
    pytorch_training,
    pytorch_training_loop,
    pytorch_training_setup,
    sklearn_data_loading,
    sklearn_evaluation,
    sklearn_header,
    sklearn_training,
    tensorflow_data_loading,
    tensorflow_evaluation,
    tensorflow_header,
    tensorflow_model_definition,
    tensorflow_training,
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
        """Export training script for a specific framework."""
        if not (config is not None):
            raise ValueError("config must be provided")
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

        output_path.write_text(script, encoding="utf-8")
        logger.info("Exported %s script to %s", framework.value, output_path)
        return output_path

    def export_config(
        self,
        config: NetworkConfig,
        output_path: Path | str,
    ) -> Path:
        """Export network configuration to JSON."""
        if not (config is not None):
            raise ValueError("config must be provided")
        output_path = self._validate_config_output_path(output_path)
        return export_config_file(config, output_path, self._normalization_params)

    def import_config(self, config_path: Path | str) -> NetworkConfig:
        """Import network configuration from JSON."""
        config, normalization_params = import_config_file(config_path)
        self._normalization_params = normalization_params
        return config

    def _validate_script_output_path(self, output_path: Path | str) -> Path:
        """Validate script output path preconditions."""
        return validate_script_output_path(output_path)

    def _validate_config_output_path(self, output_path: Path | str) -> Path:
        """Validate config output path preconditions."""
        return validate_config_output_path(output_path)

    def _validate_data_path(self, data_path: str | None) -> str | None:
        """Validate optional data path argument."""
        return validate_data_path(data_path)

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
        return build_framework_script(
            config=config,
            framework=framework,
            data_path=data_path,
            include_data_loading=include_data_loading,
            include_training=include_training,
            include_evaluation=include_evaluation,
        )

    def _generate_pytorch_script(
        self,
        config: NetworkConfig,
        data_path: str | None,
        include_data_loading: bool,
        include_training: bool,
        include_evaluation: bool,
    ) -> str:
        """Generate PyTorch training script."""
        return generate_pytorch_script(
            config,
            data_path,
            include_data_loading,
            include_training,
            include_evaluation,
        )

    def _pytorch_header(self, config: NetworkConfig) -> list[str]:
        """Generate PyTorch script header with imports."""
        return pytorch_header(config)

    def _pytorch_model_definition(self, config: NetworkConfig) -> list[str]:
        """Generate PyTorch nn.Module class definition."""
        return pytorch_model_definition(config)

    def _pytorch_data_loading(
        self, config: NetworkConfig, data_path: str | None
    ) -> list[str]:
        """Generate PyTorch data loading and DataLoader creation."""
        return pytorch_data_loading(config, data_path)

    def _pytorch_normalization_block(self) -> list[str]:
        """Generate PyTorch input normalization block."""
        return pytorch_normalization_block()

    def _pytorch_training(self, config: NetworkConfig) -> list[str]:
        """Generate PyTorch training loop with early stopping."""
        return pytorch_training(config)

    def _pytorch_training_setup(
        self, config: NetworkConfig, optimizer_name: str, loss_name: str
    ) -> list[str]:
        """Generate PyTorch setup block."""
        return pytorch_training_setup(config, optimizer_name, loss_name)

    def _pytorch_training_loop(self, config: NetworkConfig) -> list[str]:
        """Generate PyTorch epoch loop block."""
        return pytorch_training_loop(config)

    def _pytorch_evaluation(self) -> list[str]:
        """Generate PyTorch model evaluation code."""
        return pytorch_evaluation()

    def _generate_tensorflow_script(
        self,
        config: NetworkConfig,
        data_path: str | None,
        include_data_loading: bool,
        include_training: bool,
        include_evaluation: bool,
    ) -> str:
        """Generate TensorFlow/Keras training script."""
        return generate_tensorflow_script(
            config,
            data_path,
            include_data_loading,
            include_training,
            include_evaluation,
        )

    def _tensorflow_header(self) -> list[str]:
        """Generate TensorFlow script header with imports."""
        return tensorflow_header()

    def _tensorflow_model_definition(self, config: NetworkConfig) -> list[str]:
        """Generate Keras Sequential model definition."""
        return tensorflow_model_definition(config)

    def _tensorflow_data_loading(
        self, config: NetworkConfig, data_path: str | None
    ) -> list[str]:
        """Generate TensorFlow data loading code."""
        return tensorflow_data_loading(config, data_path)

    def _tensorflow_training(self, config: NetworkConfig) -> list[str]:
        """Generate TensorFlow model compilation, callbacks, and training."""
        return tensorflow_training(config)

    def _tensorflow_evaluation(self) -> list[str]:
        """Generate TensorFlow model evaluation code."""
        return tensorflow_evaluation()

    def _generate_sklearn_script(
        self,
        config: NetworkConfig,
        data_path: str | None,
        include_data_loading: bool,
        include_training: bool,
        include_evaluation: bool,
    ) -> str:
        """Generate scikit-learn training script."""
        return generate_sklearn_script(
            config,
            data_path,
            include_data_loading,
            include_training,
            include_evaluation,
        )

    def _sklearn_header(self) -> list[str]:
        """Generate scikit-learn script header with imports."""
        return sklearn_header()

    def _sklearn_data_loading(
        self, config: NetworkConfig, data_path: str | None
    ) -> list[str]:
        """Generate scikit-learn data loading and split code."""
        return sklearn_data_loading(config, data_path)

    def _sklearn_training(
        self, config: NetworkConfig, hidden_sizes: list[int]
    ) -> list[str]:
        """Generate scikit-learn MLP model creation and training."""
        return sklearn_training(config, hidden_sizes)

    def _sklearn_evaluation(self) -> list[str]:
        """Generate scikit-learn model evaluation code."""
        return sklearn_evaluation()

    def _pytorch_activation(self, activation: ActivationFunction) -> str:
        """Convert activation to PyTorch module string."""
        if not (activation is not None):
            raise ValueError("activation must be provided")
        return pytorch_activation(activation)

    def _pytorch_optimizer(self, optimizer: Optimizer) -> str:
        """Convert optimizer to PyTorch optimizer string."""
        if not (optimizer is not None):
            raise ValueError("optimizer must be provided")
        return pytorch_optimizer(optimizer)

    def _pytorch_loss(self, loss: LossFunction) -> str:
        """Convert loss to PyTorch loss string."""
        if not (loss is not None):
            raise ValueError("loss must be provided")
        return pytorch_loss(loss)

    def _keras_loss(self, loss: LossFunction) -> str:
        """Convert loss to Keras loss string."""
        if not (loss is not None):
            raise ValueError("loss must be provided")
        return keras_loss(loss)
