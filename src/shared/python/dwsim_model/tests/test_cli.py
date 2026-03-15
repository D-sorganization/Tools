import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dwsim_model.__main__ import (
    cmd_export,
    cmd_run,
    cmd_summary,
    cmd_sweep,
    cmd_validate,
    main,
)


@pytest.fixture
def dummy_args():
    args = argparse.Namespace()
    args.verbose = False
    args.config = None
    args.scenario = "test_scenario"
    args.output = "test_results"
    args.force = False
    args.save_dwxml = False
    return args


@patch("dwsim_model.results.reporter.generate_json_report")
@patch("dwsim_model.results.reporter.generate_html_report")
@patch("dwsim_model.results.metrics.MetricsCalculator")
@patch("dwsim_model.results.extractor.ResultsExtractor")
@patch("dwsim_model.gasification.GasificationFlowsheet")
def test_cmd_run(
    mock_flowsheet_class,
    mock_extractor_class,
    mock_metrics_class,
    mock_html,
    mock_json,
    dummy_args,
):
    mock_flowsheet = MagicMock()
    mock_flowsheet_class.return_value = mock_flowsheet

    mock_extractor = MagicMock()
    mock_extractor_class.return_value = mock_extractor
    mock_results = MagicMock()
    mock_extractor.extract.return_value = mock_results

    mock_metrics_calc = MagicMock()
    mock_metrics_class.return_value = mock_metrics_calc
    mock_metrics = MagicMock()
    mock_metrics.cold_gas_efficiency = 0.8
    mock_metrics.carbon_conversion_efficiency = 0.9
    mock_metrics.h2_co_ratio = 1.5
    mock_metrics.syngas_lhv_mj_nm3 = 5.0
    mock_metrics.specific_energy_consumption_kWh_t = 100.0
    mock_metrics.tar_loading_mg_Nm3 = 10.0
    mock_metrics.mass_balance_closure = 0.99
    mock_metrics.energy_balance_closure = 0.98
    mock_metrics.warnings = ["Test warning"]
    mock_metrics_calc.calculate.return_value = mock_metrics

    # Normal success
    res = cmd_run(dummy_args)
    assert res == 0
    mock_flowsheet.run.assert_called_once()
    mock_html.assert_called_once()
    mock_json.assert_called_once()

    # With save_dwxml
    dummy_args.save_dwxml = True
    res = cmd_run(dummy_args)
    assert res == 0
    assert mock_flowsheet.builder.save.called

    # If run fails, should return 1 unless force
    mock_flowsheet.run.side_effect = RuntimeError("Solver failed")
    dummy_args.force = False
    res = cmd_run(dummy_args)
    assert res == 1

    dummy_args.force = True
    res = cmd_run(dummy_args)
    assert res == 0


@patch("dwsim_model.analysis.sweep.ParameterSweep")
def test_cmd_sweep_1d(mock_sweep_class, dummy_args):
    dummy_args.param = "varA"
    dummy_args.min = 1.0
    dummy_args.max = 2.0
    dummy_args.steps = 5
    dummy_args.param_b = None
    dummy_args.kpis = None
    dummy_args.output = "test_sweep.csv"

    import pandas as pd

    mock_df = pd.DataFrame({"varA": [1.0, 2.0], "kpi1": [10.0, 20.0]})

    mock_sweep_instance = MagicMock()
    mock_sweep_instance.sweep_1d.return_value = mock_df
    mock_sweep_class.return_value = mock_sweep_instance

    res = cmd_sweep(dummy_args)
    assert res == 0
    mock_sweep_instance.sweep_1d.assert_called_once()
    assert Path("test_sweep.csv").exists()
    Path("test_sweep.csv").unlink()


@patch("dwsim_model.analysis.sweep.ParameterSweep")
def test_cmd_sweep_2d(mock_sweep_class, dummy_args):
    dummy_args.param = "varA"
    dummy_args.min = 1.0
    dummy_args.max = 2.0
    dummy_args.steps = 5
    dummy_args.param_b = "varB"
    dummy_args.min_b = 3.0
    dummy_args.max_b = 4.0
    dummy_args.steps_b = 5
    dummy_args.kpis = ["kpi1"]
    dummy_args.output = "test_sweep_2d.csv"

    import pandas as pd

    mock_df = pd.DataFrame({"varA": [1.0], "varB": [3.0], "kpi1": [10.0]})

    mock_sweep_instance = MagicMock()
    mock_sweep_instance.sweep_2d.return_value = mock_df
    mock_sweep_class.return_value = mock_sweep_instance

    res = cmd_sweep(dummy_args)
    assert res == 0
    mock_sweep_instance.sweep_2d.assert_called_once()
    assert Path("test_sweep_2d.csv").exists()
    Path("test_sweep_2d.csv").unlink()


@patch("dwsim_model.__main__._validate_yaml_directory", return_value=True)
@patch("dwsim_model.config.schema.validate_master_config")
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.open")
@patch("yaml.safe_load")
def test_cmd_validate(
    mock_safe_load, mock_open, mock_exists, mock_val_master, mock_val_dir, dummy_args
):
    dummy_args.config = "fake_config.yaml"
    mock_val_master.return_value = MagicMock(reactor_mode="test", compound_set=["H2"])

    res = cmd_validate(dummy_args)
    assert res == 0

    mock_val_master.side_effect = Exception("Invalid schema")
    res = cmd_validate(dummy_args)
    assert res == 1


@patch("dwsim_model.gasification.GasificationFlowsheet")
def test_cmd_export(mock_flowsheet_class, dummy_args):
    mock_instance = MagicMock()
    mock_flowsheet_class.return_value = mock_instance
    dummy_args.config = None
    dummy_args.output = "test_export.dwxml"

    res = cmd_export(dummy_args)
    assert res == 0
    mock_instance.builder.save.assert_called_once_with("test_export.dwxml")

    mock_instance.builder.save.side_effect = Exception("Export error")
    res = cmd_export(dummy_args)
    assert res == 1


@patch("dwsim_model.chemistry.reactions.print_reaction_summary")
def test_cmd_summary(mock_print_summary, dummy_args):
    res = cmd_summary(dummy_args)
    assert res == 0
    mock_print_summary.assert_called_once()


@patch("dwsim_model.__main__._build_parser")
def test_main_dispatch(mock_parser):
    mock_args = MagicMock()
    mock_args.verbose = False
    mock_args.command = "summary"

    mock_p = MagicMock()
    mock_p.parse_args.return_value = mock_args
    mock_parser.return_value = mock_p

    with patch("dwsim_model.__main__.cmd_summary", return_value=0) as mock_summary:
        res = main(["summary"])
        assert res == 0
        mock_summary.assert_called_once()

    # Test invalid command
    mock_args.command = "invalid"
    res = main(["invalid"])
    assert res == 1

    # Test keyboard interrupt
    mock_args.command = "summary"
    with patch("dwsim_model.__main__.cmd_summary", side_effect=KeyboardInterrupt):
        res = main(["summary"])
        assert res == 130
