from unittest.mock import MagicMock, patch

import pytest
from dwsim_model.standalone.base import StandaloneBase
from dwsim_model.standalone.gasifier_model import GasifierStandaloneFlowsheet
from dwsim_model.standalone.pem_model import PEMStandaloneFlowsheet
from dwsim_model.standalone.trc_model import TRCStandaloneFlowsheet


class DummyStandalone(StandaloneBase):
    def build_flowsheet(self) -> None:
        self._is_built = True


@patch("dwsim_model.standalone.base.get_automation")
@patch("dwsim_model.standalone.base.FlowsheetBuilder")
def test_standalone_base_setup_thermo(mock_builder_class, mock_get_automation) -> None:
    mock_builder_instance = MagicMock()
    mock_builder_class.return_value = mock_builder_instance

    model = DummyStandalone(compound_set=["TestCompound"])
    model.setup_thermo()

    mock_builder_instance.add_compound.assert_called_with("TestCompound")
    mock_builder_instance.add_property_package.assert_called_once()


@patch("dwsim_model.standalone.base.get_automation")
@patch("dwsim_model.standalone.base.FlowsheetBuilder")
def test_standalone_base_calculate(mock_builder_class, mock_get_automation) -> None:
    mock_builder_instance = MagicMock()
    mock_builder_class.return_value = mock_builder_instance

    model = DummyStandalone()

    with pytest.raises(RuntimeError):
        model.calculate()

    model.build_flowsheet()
    model.calculate()
    mock_builder_instance.calculate.assert_called_once()


@patch("dwsim_model.standalone.base.get_automation")
@patch("dwsim_model.standalone.base.FlowsheetBuilder")
@patch("dwsim_model.standalone.gasifier_model.build_gasifier_stage")
def test_gasifier_standalone(
    mock_build_stage, mock_builder_class, mock_get_automation
) -> None:
    model = GasifierStandaloneFlowsheet()
    model.build_flowsheet()

    mock_build_stage.assert_called_once()
    assert model._is_built


@patch("dwsim_model.standalone.base.get_automation")
@patch("dwsim_model.standalone.base.FlowsheetBuilder")
@patch("dwsim_model.standalone.pem_model.build_pem_stage")
def test_pem_standalone(
    mock_build_stage, mock_builder_class, mock_get_automation
) -> None:
    model = PEMStandaloneFlowsheet()
    model.build_flowsheet()

    mock_build_stage.assert_called_once()
    assert model._is_built


@patch("dwsim_model.standalone.base.get_automation")
@patch("dwsim_model.standalone.base.FlowsheetBuilder")
@patch("dwsim_model.standalone.trc_model.build_trc_stage")
def test_trc_standalone(
    mock_build_stage, mock_builder_class, mock_get_automation
) -> None:
    model = TRCStandaloneFlowsheet()
    model.build_flowsheet()

    mock_build_stage.assert_called_once()
    assert model._is_built


@patch("dwsim_model.standalone.gasifier_model.logging")
@patch("dwsim_model.standalone.gasifier_model.GasifierStandaloneFlowsheet")
def test_gasifier_main(mock_model_class, mock_logging) -> None:
    from dwsim_model.standalone.gasifier_model import main

    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance

    res = main()

    assert res == mock_instance
    mock_instance.setup_thermo.assert_called_once()
    mock_instance.build_flowsheet.assert_called_once()
    mock_instance.calculate.assert_called_once()
    mock_instance.builder.save.assert_called_once_with("Standalone_Gasifier.dwxml")


@patch("dwsim_model.standalone.pem_model.logging")
@patch("dwsim_model.standalone.pem_model.PEMStandaloneFlowsheet")
def test_pem_main(mock_model_class, mock_logging) -> None:
    from dwsim_model.standalone.pem_model import main

    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance

    res = main()

    assert res == mock_instance
    mock_instance.setup_thermo.assert_called_once()
    mock_instance.build_flowsheet.assert_called_once()
    mock_instance.calculate.assert_called_once()
    mock_instance.builder.save.assert_called_once_with("Standalone_PEM.dwxml")


@patch("dwsim_model.standalone.trc_model.logging")
@patch("dwsim_model.standalone.trc_model.TRCStandaloneFlowsheet")
def test_trc_main(mock_model_class, mock_logging) -> None:
    from dwsim_model.standalone.trc_model import main

    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance

    res = main()

    assert res == mock_instance
    mock_instance.setup_thermo.assert_called_once()
    mock_instance.build_flowsheet.assert_called_once()
    mock_instance.calculate.assert_called_once()
    mock_instance.builder.save.assert_called_once_with("Standalone_TRC.dwxml")
