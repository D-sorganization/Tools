from unittest.mock import MagicMock, mock_open, patch

import pytest
from dwsim_model.chemistry.reactions import (
    ReactorAdapter,
    ReactorConfigurationError,
    _load_reactor_contract,
    configure_gasifier,
    configure_pem,
    configure_trc,
)
from dwsim_model.config.schema import KineticParameters, ReactionEntry, ReactorConfig


def test_load_reactor_contract_not_found():
    with patch("dwsim_model.chemistry.reactions.Path.exists", return_value=False):
        with pytest.raises(ReactorConfigurationError):
            _load_reactor_contract("missing.yaml")


def test_load_reactor_contract():
    yaml_content = """
reactor:
    name: test_rct
    type: RCT_Equilibrium
    mode: isothermal
    temperature_C: 800.0
    pressure_Pa: 101325.0
reactions:
    - name: rxn1
      conversion: 0.9
"""
    with patch("dwsim_model.chemistry.reactions.Path.exists", return_value=True):
        with patch(
            "dwsim_model.chemistry.reactions.Path.open",
            mock_open(read_data=yaml_content),
        ):
            # Assuming validate_reactor_config works correctly
            # We mock it to just return a dummy config because it's tested elsewhere
            with patch(
                "dwsim_model.chemistry.reactions.validate_reactor_config"
            ) as m_val:
                m_val.return_value = "dummy_config"
                res = _load_reactor_contract("test.yaml")
                assert res == "dummy_config"
                m_val.assert_called_once()


@pytest.fixture
def mock_reactor_config():
    return ReactorConfig(
        name="test_reactor",
        type="RCT_Conversion",
        mode="isothermal",
        temperature_C=500.0,
        pressure_Pa=200000.0,
        volume_m3=None,
        length_m=None,
        diameter_m=None,
        reactions=[],
    )


def test_reactor_adapter_apply(mock_reactor_config):
    reactor_obj = MagicMock()
    sim = MagicMock()

    adapter = ReactorAdapter(reactor_obj, sim, mock_reactor_config)
    adapter.apply()

    # Should attempt to set pressure
    reactor_obj.SetPropertyValue.assert_any_call("PROP_CR_0", 200000.0)
    # Mode = isothermal (0)
    reactor_obj.SetPropertyValue.assert_any_call("Calculation Mode", 0)


def test_reactor_adapter_unsupported_mode(mock_reactor_config):
    reactor_obj = MagicMock()
    mock_reactor_config.mode = "specified_duty"
    adapter = ReactorAdapter(reactor_obj, MagicMock(), mock_reactor_config)
    with pytest.raises(ReactorConfigurationError, match="specified_duty"):
        adapter._set_operation_mode()

    mock_reactor_config.mode = "invalid"
    adapter = ReactorAdapter(reactor_obj, MagicMock(), mock_reactor_config)
    with pytest.raises(
        ReactorConfigurationError, match="unsupported reactor mode 'invalid'"
    ):
        adapter._set_operation_mode()


def test_reactor_adapter_geometry():
    config = ReactorConfig(
        name="test_pfr",
        type="RCT_PFR",
        mode="isothermal",
        temperature_C=500.0,
        pressure_Pa=1e5,
        volume_m3=10.0,
        length_m=5.0,
        diameter_m=2.0,
        reactions=[],
    )
    reactor_obj = MagicMock()
    adapter = ReactorAdapter(reactor_obj, MagicMock(), config)
    adapter._set_geometry()

    reactor_obj.SetPropertyValue.assert_any_call("PROP_PF_2", 10.0)
    reactor_obj.SetPropertyValue.assert_any_call("PROP_PF_3", 5.0)


def test_reactor_adapter_add_reaction(mock_reactor_config):
    reactor_obj = MagicMock()
    reactions_col = MagicMock()
    reactor_obj.Reactions = reactions_col

    sim = MagicMock()
    sim.AddReaction.return_value = MagicMock(ID="RXN_99")

    rxn = ReactionEntry(
        name="R1", conversion=0.5, base_component="CH4", stoichiometry="CH4"
    )
    mock_reactor_config.reactions = [rxn]

    adapter = ReactorAdapter(reactor_obj, sim, mock_reactor_config)
    adapter._add_reaction(rxn)

    sim.AddReaction.assert_called_once_with("R1", "Conversion", "CH4", 0.5)
    reactions_col.Add.assert_called_once_with("RXN_99")


def test_reactor_adapter_reaction_kinetics(mock_reactor_config):
    reactor_obj = MagicMock()
    sim = MagicMock()
    r_obj = MagicMock()
    sim.AddReaction.return_value = r_obj

    kinetics = KineticParameters(
        pre_exponential_A=1.0, activation_energy_J_mol=2.0, reaction_order_n=1.5
    )
    rxn = ReactionEntry(name="R1", stoichiometry="CH4", kinetics=kinetics)
    mock_reactor_config.reactions = [rxn]
    mock_reactor_config.type = "RCT_PFR"  # Uses Kinetic

    adapter = ReactorAdapter(reactor_obj, sim, mock_reactor_config)
    adapter._apply_reaction_details(r_obj, rxn)

    assert r_obj.PreExponentialFactor == 1.0
    assert r_obj.ActivationEnergy == 2.0
    assert r_obj.ReactionOrder == 1.5


def test_resolve_reaction_type(mock_reactor_config):
    adapter = ReactorAdapter(MagicMock(), MagicMock(), mock_reactor_config)

    mock_reactor_config.type = "RCT_Conversion"
    assert adapter._resolve_reaction_type() == "Conversion"

    mock_reactor_config.type = "RCT_Equilibrium"
    assert adapter._resolve_reaction_type() == "Equilibrium"

    mock_reactor_config.type = "RCT_PFR"
    assert adapter._resolve_reaction_type() == "Kinetic"

    mock_reactor_config.type = "Unknown"
    with pytest.raises(ReactorConfigurationError):
        adapter._resolve_reaction_type()


@patch("dwsim_model.chemistry.reactions.ReactorAdapter")
@patch("dwsim_model.chemistry.reactions._load_reactor_contract")
def test_configure_functions(mock_load, mock_adapter):
    mock_load.return_value = MagicMock(reactions=[1, 2])

    sim = MagicMock()
    obj = MagicMock()

    configure_gasifier(obj, sim)
    mock_load.assert_called_with("gasifier_reactions.yaml")

    configure_pem(obj, sim)
    mock_load.assert_called_with("pem_reactions.yaml")

    configure_trc(obj, sim)
    mock_load.assert_called_with("trc_reactions.yaml")
