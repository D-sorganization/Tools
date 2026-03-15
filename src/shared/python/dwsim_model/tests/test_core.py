from unittest.mock import MagicMock, patch

import pytest
from dwsim_model.core import FlowsheetBuilder, get_automation


def test_get_automation_missing_path() -> None:
    with patch("os.environ.get", return_value=None):
        with pytest.raises(
            RuntimeError, match="DWSIM_PATH environment variable is not set"
        ):
            get_automation(None)


@patch("dwsim_model.core.get_automation")
def test_flowsheet_builder_init(mock_get_automation) -> None:
    mock_interf = MagicMock()
    mock_obj_type = MagicMock()
    mock_get_automation.return_value = (mock_interf, mock_obj_type)

    builder = FlowsheetBuilder(dwsim_path="/dummy/path")
    assert builder.materials == {}
    assert builder.energy_streams == {}
    assert builder.operations == {}
    mock_interf.CreateFlowsheet.assert_called_once()


@patch("dwsim_model.core.get_automation")
def test_flowsheet_builder_add_compound(mock_get_automation) -> None:
    mock_interf = MagicMock()
    mock_get_automation.return_value = (mock_interf, MagicMock())

    builder = FlowsheetBuilder(dwsim_path="/dummy/path")
    builder.add_compound("Methane")

    builder.sim.AddCompound.assert_called_once_with("Methane")


@patch("dwsim_model.core.get_automation")
def test_flowsheet_builder_add_property_package(mock_get_automation) -> None:
    mock_interf = MagicMock()
    mock_interf.AvailablePropertyPackages = {"TEST_PKG": "test_pkg_obj"}
    mock_get_automation.return_value = (mock_interf, MagicMock())

    builder = FlowsheetBuilder(dwsim_path="/dummy/path")
    pkg = builder.add_property_package("TEST_PKG")

    assert pkg == "test_pkg_obj"
    builder.sim.AddPropertyPackage.assert_called_once_with("test_pkg_obj")


@patch("dwsim_model.core.get_automation")
def test_flowsheet_builder_calculate(mock_get_automation) -> None:
    mock_interf = MagicMock()
    mock_get_automation.return_value = (mock_interf, MagicMock())

    builder = FlowsheetBuilder(dwsim_path="/dummy/path")
    builder.calculate()

    mock_interf.CalculateFlowsheet2.assert_called_once_with(builder.sim)
