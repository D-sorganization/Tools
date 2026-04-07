"""Regression and architecture tests for the split pressure-drop engine."""

from __future__ import annotations

from pathlib import Path

import pytest
from upstream_drift_tools.process_calculators.pressure_drop_calculator.engine.compressible_flow import (
    calculate_expansion_factor,
)
from upstream_drift_tools.process_calculators.pressure_drop_calculator.engine.fittings import (
    calculate_fitting_pressure_drop,
)
from upstream_drift_tools.process_calculators.pressure_drop_calculator.engine.flow_properties import (
    calculate_elevation_pressure_drop,
    classify_flow_regime,
)
from upstream_drift_tools.process_calculators.pressure_drop_calculator.engine.friction_factors import (
    friction_factor_colebrook,
)
from upstream_drift_tools.process_calculators.pressure_drop_calculator.engine.pressure_drop_calculation_engine import (
    PressureDropCalculationEngine,
)
from upstream_drift_tools.process_calculators.pressure_drop_calculator.models.pressure_drop_data_models import (
    PipeFitting,
)


def _nonblank_line_count(path: Path) -> int:
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def test_pressure_drop_engine_is_split_into_domain_modules(repo_root: Path) -> None:
    engine_dir = (
        repo_root
        / "src"
        / "shared"
        / "python"
        / "upstream_drift_tools"
        / "process_calculators"
        / "pressure_drop_calculator"
        / "engine"
    )
    for name in (
        "friction_factors.py",
        "flow_properties.py",
        "fittings.py",
        "compressible_flow.py",
    ):
        assert (engine_dir / name).exists(), f"Missing extracted module: {name}"

    facade = engine_dir / "pressure_drop_calculation_engine.py"
    assert _nonblank_line_count(facade) <= 400


def test_extracted_pressure_drop_modules_preserve_regression_values() -> None:
    assert friction_factor_colebrook(100_000.0, 0.0001) == pytest.approx(
        0.0185,
        rel=0.08,
    )
    assert classify_flow_regime(5_000.0) == "turbulent"
    assert calculate_elevation_pressure_drop(1.2, 10.0) > 0
    assert 0.0 <= calculate_expansion_factor(100_000.0, 50_000.0, 0.02, 50.0) < 1.0

    fittings = [PipeFitting("custom_fitting", quantity=2, k_factor=1.5)]
    assert calculate_fitting_pressure_drop(
        fittings, 1.2, 10.0, 50_000.0, 4.0
    ) == pytest.approx(180.0)


def test_pressure_drop_engine_facade_remains_constructible() -> None:
    engine = PressureDropCalculationEngine()
    assert engine is not None
