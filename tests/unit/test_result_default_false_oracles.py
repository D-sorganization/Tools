"""Oracle 1 & 8 tests for default-false result contracts (#6003 Slice 4).

Oracle 1 (Default-true):
Constructing any solver/calc result class with minimal required arguments
(or without explicitly asserting success) must NEVER report success/converged is True.

Oracle 8 (Old readers):
Existing readers and serializers (model_dump, asdict, .success attribute)
receive the exact boolean contract expected for both success and failure outcomes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from calc_backend.contracts.ode_solver import ODESolverRequest, ODESolverResponse
from calc_backend.routers.ode_solver import solve_ode
from data_processor.core.state_space import (
    LocalLevelModel,
    StateSpaceConfig,
    StateSpaceModelType,
    StateSpaceResult,
)
from signal_toolkit.core import Signal
from signal_toolkit.fitting import FitResult, LinearFitter

from pressure_drop_calculator.models import PressureDropOutput
from rate_of_closure.simulation.anthropometry import GolferAnthropometry
from rate_of_closure.simulation.optimized_swing import (
    OptimizedSwingResult,
    OptimizedSwingSource,
)


def _build_minimal_ode() -> tuple[ODESolverResponse, str]:
    return (
        ODESolverResponse(
            times=[0.0, 1.0],
            solutions={"y": [1.0, 0.5]},
            variable_summaries=[],
        ),
        "success",
    )


def _build_minimal_pressure_drop() -> tuple[PressureDropOutput, str]:
    return (
        PressureDropOutput(
            pressure_drop_pa=100.0,
            pressure_drop_bar=0.001,
            pressure_drop_psi=0.0145,
            outlet_temperature_k=300.0,
        ),
        "success",
    )


def _build_minimal_fit_result() -> tuple[FitResult, str]:
    t = np.array([0.0, 1.0])
    return (
        FitResult(
            parameters={"slope": 1.0, "intercept": 0.0},
            covariance=None,
            r_squared=1.0,
            rmse=0.0,
            fitted_signal=Signal(time=t, values=t),
            residuals=np.zeros(2),
        ),
        "success",
    )


def _build_minimal_state_space() -> tuple[StateSpaceResult, str]:
    return (
        StateSpaceResult(
            fitted_values=np.array([1.0, 2.0]),
            residuals=np.zeros(2),
            filtered_states=np.zeros((2, 1)),
            smoothed_states=np.zeros((2, 1)),
            filtered_state_cov=np.zeros((2, 1, 1)),
            smoothed_state_cov=np.zeros((2, 1, 1)),
            parameters={"sigma2_eps": 1.0},
            log_likelihood=-2.5,
            model_type=StateSpaceModelType.LOCAL_LEVEL,
            n_observations=2,
            n_states=1,
        ),
        "converged",
    )


def _build_minimal_optimized_swing() -> tuple[OptimizedSwingResult, str]:
    poses = np.zeros((2, 4, 4))
    poses[0] = np.eye(4)
    poses[1] = np.eye(4)
    return (
        OptimizedSwingResult(
            time_s=np.array([0.0, 0.1]),
            clubhead_poses=poses,
            clubhead_twists=np.zeros((2, 6)),
        ),
        "success",
    )


TARGET_CLASSES: list[tuple[str, Callable[[], tuple[Any, str]]]] = [
    ("ODESolverResponse", _build_minimal_ode),
    ("PressureDropOutput", _build_minimal_pressure_drop),
    ("FitResult", _build_minimal_fit_result),
    ("StateSpaceResult", _build_minimal_state_space),
    ("OptimizedSwingResult", _build_minimal_optimized_swing),
]


class TestOracle1DefaultFalse:
    """Oracle 1: constructing with minimal args must not report success=True."""

    @pytest.mark.parametrize(
        ("class_name", "factory"),
        TARGET_CLASSES,
        ids=[name for name, _ in TARGET_CLASSES],
    )
    def test_minimal_construction_defaults_to_false(
        self, class_name: str, factory: Callable[[], tuple[Any, str]]
    ) -> None:
        instance, flag_attr = factory()
        flag_val = getattr(instance, flag_attr)
        assert flag_val is False, (
            f"{class_name}.{flag_attr} defaulted to {flag_val!r}; "
            f"must default to False when omitted on error or partial construction path."
        )

    def test_duck_typed_swing_source_defaults_missing_flag_to_false(self) -> None:
        class IncompleteOptimizerResult:
            def __init__(self) -> None:
                self.time_s = np.array([0.0, 0.1])
                poses = np.zeros((2, 4, 4))
                poses[0] = np.eye(4)
                poses[1] = np.eye(4)
                self.clubhead_poses = poses
                self.clubhead_twists = np.zeros((2, 6))

        with pytest.raises(ValueError, match="solver failure"):
            OptimizedSwingSource(IncompleteOptimizerResult())


class TestOracle8OldReaders:
    """Oracle 8: old readers receive expected boolean contracts and keys."""

    def test_ode_solver_reader_contract(self) -> None:
        req = ODESolverRequest(
            derivatives={"y": "-y"},
            initial_conditions={"y": 1.0},
            t_start=0.0,
            t_end=1.0,
            num_points=11,
        )
        resp = solve_ode(req)
        assert resp.success is True
        dump = resp.model_dump()
        assert dump["success"] is True
        assert "Solution computed successfully" in dump["message"]

        parsed_true = ODESolverResponse.model_validate({**dump, "success": True})
        assert parsed_true.success is True
        parsed_false = ODESolverResponse.model_validate({**dump, "success": False})
        assert parsed_false.success is False

    def test_pressure_drop_reader_contract(self) -> None:
        out = PressureDropOutput(
            pressure_drop_pa=100.0,
            pressure_drop_bar=0.001,
            pressure_drop_psi=0.0145,
            outlet_temperature_k=300.0,
            success=True,
        )
        assert out.success is True
        assert out.model_dump()["success"] is True

        fail_out = PressureDropOutput(
            pressure_drop_pa=0.0,
            pressure_drop_bar=0.0,
            pressure_drop_psi=0.0,
            outlet_temperature_k=300.0,
            success=False,
            error_message="Flow choked",
        )
        assert fail_out.success is False
        assert fail_out.model_dump()["success"] is False

    def test_signal_fitter_reader_contract(self) -> None:
        t = np.linspace(0, 1, 20)
        sig = Signal(time=t, values=2.0 * t + 1.0)
        res = LinearFitter().fit(sig)
        assert res.success is True
        d = dataclasses.asdict(res)
        assert d["success"] is True
        assert d["parameters"]["slope"] == pytest.approx(2.0)

    def test_state_space_reader_contract(self) -> None:
        np.random.seed(42)
        y = np.cumsum(np.random.randn(30))
        cfg = StateSpaceConfig(model_type=StateSpaceModelType.LOCAL_LEVEL)
        res = LocalLevelModel(cfg).fit(y)
        assert res.converged is True
        d = dataclasses.asdict(res)
        assert d["converged"] is True

    def test_optimized_swing_reader_contract(self) -> None:
        anthro = GolferAnthropometry()
        res = anthro.generate_delivery(duration_s=0.5, dt=1e-3)
        assert res.success is True
        d = dataclasses.asdict(res)
        assert d["success"] is True

        source = OptimizedSwingSource(res)
        assert source.duration == pytest.approx(0.5, rel=1e-3)
