import pytest
from unittest.mock import patch, MagicMock
from dwsim_model.gasification import GasificationFlowsheet, ReactorMode

def test_gasification_flowsheet_init() -> None:
    builder = MagicMock()
    flowsheet = GasificationFlowsheet(builder=builder, mode="mixed")
    
    assert flowsheet.builder == builder
    assert flowsheet.mode == ReactorMode.MIXED
    assert flowsheet._is_built == False


@patch("dwsim_model.gasification.DEFAULT_PROPERTY_PACKAGE", "TestPackage")
def test_setup_thermo() -> None:
    builder = MagicMock()
    flowsheet = GasificationFlowsheet(builder=builder, compound_set=["Methane"])
    
    flowsheet.setup_thermo()
    
    builder.add_compound.assert_called_once_with("Methane")
    builder.add_property_package.assert_called_once_with("TestPackage")


def test_reactor_types() -> None:
    # Test CUSTOM mode
    flowsheet = GasificationFlowsheet(mode="custom", custom_reactors={"gasifier": "RCT_1"})
    types = flowsheet._get_reactor_types()
    assert types["gasifier"] == "RCT_1"
    
    # Test MIXED mode
    flowsheet = GasificationFlowsheet(mode="mixed")
    types = flowsheet._get_reactor_types()
    assert types["gasifier"] == "RCT_Conversion"
    assert types["pem"] == "RCT_Equilibrium"
    assert types["trc"] == "RCT_PFR"


@patch("dwsim_model.gasification.build_gasifier_stage")
@patch("dwsim_model.gasification.build_pem_stage")
@patch("dwsim_model.gasification.build_trc_stage")
@patch("dwsim_model.gasification.ConfigLoader")
def test_build_flowsheet(mock_config_loader, mock_trc, mock_pem, mock_gasifier) -> None:
    builder = MagicMock()
    builder.materials = {"Syngas_Pre_TRC": MagicMock(), "Syngas_Pre_Quench": MagicMock()}
    
    # Mock stage returns
    mock_gasifier.return_value = {"syngas_out": MagicMock()}
    
    flowsheet = GasificationFlowsheet(builder=builder, mode="custom")
    
    # Bypass the chemistry configuration which would fail due to missing actual DWSIM references in the mock
    flowsheet._configure_reactors = MagicMock()
    
    flowsheet.build_flowsheet()
    
    assert flowsheet._is_built == True
    mock_gasifier.assert_called_once()
    mock_pem.assert_called_once()
    mock_trc.assert_called_once()
    mock_config_loader.assert_called_once()


def test_run_unbuilt_flowsheet() -> None:
    builder = MagicMock()
    flowsheet = GasificationFlowsheet(builder=builder)
    
    with pytest.raises(RuntimeError):
        flowsheet.run()


def test_run_built_flowsheet() -> None:
    builder = MagicMock()
    flowsheet = GasificationFlowsheet(builder=builder)
    flowsheet._is_built = True
    
    flowsheet.run()
    
    builder.calculate.assert_called_once()
