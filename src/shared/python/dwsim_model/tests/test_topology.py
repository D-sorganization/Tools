from unittest.mock import MagicMock

import pytest
from dwsim_model.topology import build_gasifier_stage, build_pem_stage, build_trc_stage


def test_build_gasifier_stage() -> None:
    mock_builder = MagicMock()
    mock_connect = MagicMock()

    mock_builder.add_object.side_effect = lambda type_name, name, x, y: f"mock_{name}"

    res = build_gasifier_stage(mock_builder, "ReactorType", mock_connect)

    assert "reactor" in res
    assert "syngas_out" in res
    assert "glass_out" in res
    assert mock_builder.add_object.call_count > 10
    assert mock_connect.call_count > 10


def test_build_pem_stage() -> None:
    mock_builder = MagicMock()
    mock_connect = MagicMock()

    mock_builder.add_object.side_effect = lambda type_name, name, x, y: f"mock_{name}"

    res = build_pem_stage(mock_builder, "ReactorType", mock_connect)

    assert "syngas_in" in res
    assert "reactor" in res
    assert mock_builder.add_object.call_count > 10
    assert mock_connect.call_count > 10


def test_build_trc_stage() -> None:
    mock_builder = MagicMock()
    mock_connect = MagicMock()

    mock_builder.add_object.side_effect = lambda type_name, name, x, y: f"mock_{name}"

    res = build_trc_stage(mock_builder, "ReactorType", mock_connect)

    assert "syngas_in" in res
    assert "reactor" in res
    assert "syngas_out" in res
    assert mock_builder.add_object.call_count > 5
    assert mock_connect.call_count > 5


def test_build_topology_validation() -> None:
    with pytest.raises(ValueError):
        build_gasifier_stage(None, "ReactorType", MagicMock())

    with pytest.raises(ValueError):
        build_gasifier_stage(MagicMock(), "", MagicMock())

    with pytest.raises(ValueError):
        build_gasifier_stage(MagicMock(), "ReactorType", None)
