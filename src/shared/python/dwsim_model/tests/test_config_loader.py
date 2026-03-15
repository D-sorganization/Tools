from unittest.mock import MagicMock, patch

from dwsim_model.config_loader import ConfigLoader, _deep_merge, _is_runtime_config


def test_is_runtime_config() -> None:
    assert _is_runtime_config({"feeds": {"stream_1": {"temp": 300}}})
    assert not _is_runtime_config({"feeds": "path/to/feeds.yaml"})
    assert _is_runtime_config({"energy_streams": {"E1": 500}})

def test_deep_merge() -> None:
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"d": 3}, "e": 4}
    merged = _deep_merge(base, override)

    assert merged == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

def test_config_loader_init_with_data() -> None:
    data = {"feeds": {"S1": {"temperature_C": 25}}}
    loader = ConfigLoader(config_data=data)

    assert loader._config_data == data
    loaded = loader.load()
    assert loaded == data

@patch("dwsim_model.config_loader._load_file")
@patch("dwsim_model.config_loader.Path.exists")
def test_config_loader_load_file(mock_exists, mock_load_file) -> None:
    mock_exists.return_value = True
    # Return a basic valid master config to pass validation
    mock_load_file.return_value = {
        "model": {"name": "TestModel", "version": "1.0"},
        "reactor_mode": "mixed",
        "compound_set": "standard",
        "feeds": {},
        "reactors": {},
        "output": {"directory": "results"},
    }

    loader = ConfigLoader(config_path="dummy.yaml")
    config = loader.load()

    assert mock_load_file.called
    assert "model" in config
    assert config["reactor_mode"] == "mixed"

def test_apply_to_flowsheet() -> None:
    data = {"feeds": {"S1": {"temperature_C": 25}}, "energy_streams": {"E1": 500.0}}
    loader = ConfigLoader(config_data=data)
    loader.load()

    mock_builder = MagicMock()

    mock_s1 = MagicMock()
    mock_e1 = MagicMock()

    materials = {"S1": mock_s1}
    energies = {"E1": mock_e1}

    loader.apply_to_flowsheet(mock_builder, materials, energies)

    mock_s1.SetPropertyValue.assert_called_with("Temperature", 298.15)
    mock_e1.SetPropertyValue.assert_called_with("PROP_ES_0", 0.5)
