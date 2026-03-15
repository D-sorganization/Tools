from unittest.mock import MagicMock, patch

import pytest

from dwsim_model.units import (
    _extract_standalone_results,
    _validate_mode,
    run_full_train,
    run_gasifier,
    run_pem,
    run_trc,
)


def test_validate_mode() -> None:
    _validate_mode("conversion")
    _validate_mode("mixed")

    with pytest.raises(ValueError, match="Invalid mode"):
        _validate_mode("invalid_mode")


@patch("importlib.import_module")
def test_extract_standalone_results(mock_import_module) -> None:
    # Setup mocks
    mock_extractor_mod = MagicMock()
    mock_metrics_mod = MagicMock()

    mock_extractor = MagicMock()
    mock_results = MagicMock()
    mock_extractor.extract.return_value = mock_results
    mock_extractor_mod.ResultsExtractor.return_value = mock_extractor

    mock_calculator = MagicMock()
    mock_metrics = MagicMock()
    mock_metrics.warnings = ["Test warning"]
    mock_calculator.calculate.return_value = mock_metrics
    mock_metrics_mod.MetricsCalculator.return_value = mock_calculator

    # Configure import_module to return different mocks based on name
    def side_effect(name):
        if name == "dwsim_model.results.extractor":
            return mock_extractor_mod
        if name == "dwsim_model.results.metrics":
            return mock_metrics_mod
        return MagicMock()

    mock_import_module.side_effect = side_effect

    # Run the function
    builder = MagicMock()
    result = _extract_standalone_results(builder)

    assert result["results"] == mock_results
    assert result["metrics"] == mock_metrics
    assert result["warnings"] == ["Test warning"]

    mock_extractor.extract.assert_called_once_with(builder)


@patch("dwsim_model.units._extract_standalone_results")
@patch("importlib.import_module")
def test_run_gasifier(mock_import_module, mock_extract) -> None:
    mock_gasifier_mod = MagicMock()
    mock_import_module.return_value = mock_gasifier_mod

    mock_flowsheet = MagicMock()
    mock_gasifier_mod.GasifierStandaloneFlowsheet.return_value = mock_flowsheet

    mock_extract.return_value = {"status": "ok"}

    result = run_gasifier(mode="conversion", compound_set=["C1", "C2"])

    assert result == {"status": "ok"}
    mock_import_module.assert_called_with("dwsim_model.standalone.gasifier_model")
    mock_gasifier_mod.GasifierStandaloneFlowsheet.assert_called_once_with(
        compound_set=["C1", "C2"]
    )
    mock_flowsheet.setup_thermo.assert_called_once()
    mock_flowsheet.build_flowsheet.assert_called_once()
    mock_flowsheet.calculate.assert_called_once()
    mock_extract.assert_called_once_with(mock_flowsheet.builder)


@patch("dwsim_model.units._extract_standalone_results")
@patch("importlib.import_module")
def test_run_pem(mock_import_module, mock_extract) -> None:
    mock_mod = MagicMock()
    mock_import_module.return_value = mock_mod
    mock_flowsheet = MagicMock()
    mock_mod.PEMStandaloneFlowsheet.return_value = mock_flowsheet

    run_pem(mode="equilibrium")
    mock_import_module.assert_called_with("dwsim_model.standalone.pem_model")
    mock_flowsheet.calculate.assert_called_once()
    mock_extract.assert_called_once_with(mock_flowsheet.builder)


@patch("dwsim_model.units._extract_standalone_results")
@patch("importlib.import_module")
def test_run_trc(mock_import_module, mock_extract) -> None:
    mock_mod = MagicMock()
    mock_import_module.return_value = mock_mod
    mock_flowsheet = MagicMock()
    mock_mod.TRCStandaloneFlowsheet.return_value = mock_flowsheet

    run_trc(mode="kinetic")
    mock_import_module.assert_called_with("dwsim_model.standalone.trc_model")
    mock_flowsheet.calculate.assert_called_once()
    mock_extract.assert_called_once_with(mock_flowsheet.builder)


@patch("dwsim_model.units._extract_standalone_results")
@patch("importlib.import_module")
def test_run_full_train(mock_import_module, mock_extract) -> None:
    mock_mod = MagicMock()
    mock_import_module.return_value = mock_mod
    mock_flowsheet = MagicMock()
    mock_mod.GasificationFlowsheet.return_value = mock_flowsheet

    run_full_train(mode="mixed", config_path="test.yaml", compound_set=["A", "B"])
    mock_import_module.assert_called_with("dwsim_model.gasification")
    mock_mod.GasificationFlowsheet.assert_called_once_with(
        mode="mixed", config_path="test.yaml", compound_set=["A", "B"]
    )
    mock_flowsheet.build_flowsheet.assert_called_once()
    mock_flowsheet.run.assert_called_once()
    mock_extract.assert_called_once_with(mock_flowsheet.builder)
