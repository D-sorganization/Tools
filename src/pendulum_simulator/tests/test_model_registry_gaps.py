"""Gap-fill tests for model_registry.py — covers remaining uncovered lines.

Line 78: Warning when overwriting existing model registration
Lines 133-134, 155-156, 177-178: ImportError handlers (not testable without mocking)
"""

from __future__ import annotations

import logging

import pytest

from double_pendulum_golf.model_registry import (
    ModelConfig,
    clear_registry,
    get_model,
    register_model,
)
from double_pendulum_golf.physics import PendulumParams
from double_pendulum_golf.simulation import SimulationResult, run_simulation


@pytest.fixture(autouse=True)
def restore_registry():
    """Save and restore the registry state around each test."""
    from double_pendulum_golf import model_registry as reg_module

    saved = dict(reg_module._registry)
    yield
    reg_module._registry.clear()
    reg_module._registry.update(saved)


def _make_config(name: str = "Test Model", n_dof: int = 2) -> ModelConfig:
    return ModelConfig(
        name=name,
        n_dof=n_dof,
        state_size=n_dof * 2,
        param_class=PendulumParams,
        simulation_runner=run_simulation,
        result_class=SimulationResult,
        description="Test model",
    )


# ===========================================================================
# Line 78: Warning on overwrite
# ===========================================================================


class TestRegisterModelOverwrite:
    def test_overwrites_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Registering the same name twice should warn and replace."""
        cfg1 = _make_config("First Version", n_dof=2)
        cfg2 = _make_config("Second Version", n_dof=3)

        register_model("__test_overwrite__", cfg1)
        with caplog.at_level(logging.WARNING, logger="double_pendulum_golf.model_registry"):
            register_model("__test_overwrite__", cfg2)

        assert "Overwriting existing model registration" in caplog.text
        # Second config should be active
        assert get_model("__test_overwrite__").n_dof == 3

    def test_no_warn_first_registration(self, caplog: pytest.LogCaptureFixture) -> None:
        """First registration should not warn."""
        with caplog.at_level(logging.WARNING, logger="double_pendulum_golf.model_registry"):
            register_model("__test_first__", _make_config())
        assert "Overwriting" not in caplog.text


# ===========================================================================
# Verify import-error branches via monkeypatching builtins
# ===========================================================================


class TestRegisterBuiltinsImportError:
    def test_import_error_branches(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Simulate ImportError for all built-in models to cover except blocks."""
        import sys
        from double_pendulum_golf import model_registry

        # Clear registry
        clear_registry()

        # Force ImportError by setting modules to None in sys.modules
        monkeypatch.setitem(sys.modules, "double_pendulum_golf.physics", None)
        monkeypatch.setitem(sys.modules, "double_pendulum_golf.physics_triple", None)
        monkeypatch.setitem(sys.modules, "double_pendulum_golf.physics_golfer", None)

        with caplog.at_level(logging.DEBUG, logger="double_pendulum_golf.model_registry"):
            model_registry._register_builtins()

        # All 3 modules should fail to import and log at DEBUG level
        assert "Could not register double pendulum model" in caplog.text
        assert "Could not register triple pendulum model" in caplog.text
        assert "Could not register golfer model" in caplog.text

        # Registry should remain empty
        assert len(model_registry.list_models()) == 0
