"""Tests for the physics model registry."""

from __future__ import annotations

import pytest

from double_pendulum_golf.model_registry import (
    ModelConfig,
    get_model,
    list_models,
    register_model,
)


class TestModelRegistry:
    def setup_method(self) -> None:
        # Don't clear builtins, just test that they exist
        pass

    def test_builtin_models_registered(self) -> None:
        models = list_models()
        assert "double" in models
        assert "triple" in models
        assert "golfer" in models

    def test_get_double_model(self) -> None:
        config = get_model("double")
        assert config.n_dof == 2
        assert config.state_size == 4
        assert config.name == "Double Pendulum"

    def test_get_triple_model(self) -> None:
        config = get_model("triple")
        assert config.n_dof == 3
        assert config.state_size == 6

    def test_get_golfer_model(self) -> None:
        config = get_model("golfer")
        assert config.n_dof == 8
        assert config.state_size == 16

    def test_get_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError):
            get_model("nonexistent_model")

    def test_custom_model_registration(self) -> None:
        config = ModelConfig(
            name="Test Model",
            n_dof=1,
            state_size=2,
            param_class=dict,
            simulation_runner=lambda: None,
            result_class=dict,
            description="A test model",
        )
        register_model("test_custom", config)
        assert "test_custom" in list_models()
        assert get_model("test_custom").n_dof == 1
        # Clean up
        from double_pendulum_golf.model_registry import _registry

        _registry.pop("test_custom", None)
