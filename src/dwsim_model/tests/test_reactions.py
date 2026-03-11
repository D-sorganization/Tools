from __future__ import annotations

from types import SimpleNamespace

import pytest

from dwsim_model.chemistry.reactions import (
    ReactorAdapter,
    ReactorConfigurationError,
    configure_gasifier,
    configure_pem,
    configure_trc,
)
from dwsim_model.config.schema import validate_reactor_config


class FakeReactionList:
    def __init__(self):
        self.ids: list[str] = []

    def Add(self, reaction_id: str) -> None:
        self.ids.append(reaction_id)


class FakeReactor:
    def __init__(self):
        self.Reactions = FakeReactionList()
        self.properties: dict[str, float] = {}

    def SetPropertyValue(self, name: str, value: float) -> None:
        self.properties[name] = value


class FakeSimulation:
    def __init__(self):
        self.created: list[SimpleNamespace] = []

    def AddReaction(
        self, name: str, reaction_type: str, base_component: str, conversion: float
    ):
        reaction = SimpleNamespace(
            ID=f"{name}:{reaction_type}",
            Name=name,
            Type=reaction_type,
            BaseComponent=base_component,
            Conversion=conversion,
            PreExponentialFactor=None,
            ActivationEnergy=None,
            ReactionOrder=None,
        )
        self.created.append(reaction)
        return reaction


def test_validate_conversion_reactor_requires_conversion():
    with pytest.raises(ValueError, match="requires conversion"):
        validate_reactor_config(
            {
                "name": "Downdraft_Gasifier",
                "type": "RCT_Conversion",
                "temperature_C": 900.0,
                "pressure_Pa": 101325.0,
                "mode": "isothermal",
                "reactions": [
                    {
                        "name": "Partial Oxidation",
                        "stoichiometry": "2C + O2 -> 2CO",
                        "base_component": "Oxygen",
                    }
                ],
            }
        )


def test_validate_pfr_requires_geometry_and_kinetics():
    with pytest.raises(ValueError, match="requires kinetics"):
        validate_reactor_config(
            {
                "name": "TRC_Reactor",
                "type": "RCT_PFR",
                "temperature_C": 950.0,
                "pressure_Pa": 101325.0,
                "mode": "adiabatic",
                "volume_m3": 2.0,
                "length_m": 3.0,
                "diameter_m": 0.92,
                "reactions": [
                    {
                        "name": "Tar Cracking",
                        "stoichiometry": "Tar -> CO + H2",
                        "base_component": "Naphthalene",
                    }
                ],
            }
        )


def test_reactor_adapter_applies_conversion_contract():
    reactor = FakeReactor()
    sim = FakeSimulation()
    config = validate_reactor_config(
        {
            "name": "Downdraft_Gasifier",
            "type": "RCT_Conversion",
            "temperature_C": 900.0,
            "pressure_Pa": 101325.0,
            "mode": "isothermal",
            "reactions": [
                {
                    "name": "Partial Oxidation",
                    "stoichiometry": "2C + O2 -> 2CO",
                    "base_component": "Oxygen",
                    "conversion": 0.9,
                }
            ],
        }
    )

    ReactorAdapter(reactor, sim, config).apply()

    assert reactor.properties["PROP_CR_0"] == pytest.approx(101325.0)
    assert reactor.Reactions.ids == ["Partial Oxidation:Conversion"]


def test_reactor_adapter_applies_pfr_geometry_and_kinetics():
    reactor = FakeReactor()
    sim = FakeSimulation()
    config = validate_reactor_config(
        {
            "name": "TRC_Reactor",
            "type": "RCT_PFR",
            "temperature_C": 950.0,
            "pressure_Pa": 101325.0,
            "mode": "adiabatic",
            "volume_m3": 2.0,
            "length_m": 3.0,
            "diameter_m": 0.92,
            "reactions": [
                {
                    "name": "Tar Cracking",
                    "stoichiometry": "Tar -> CO + H2",
                    "base_component": "Naphthalene",
                    "kinetics": {
                        "pre_exponential_A": 9.2e9,
                        "activation_energy_J_mol": 2.04e5,
                        "reaction_order_n": 1.0,
                    },
                }
            ],
        }
    )

    ReactorAdapter(reactor, sim, config).apply()

    created = sim.created[0]
    assert reactor.properties["PROP_PF_0"] == pytest.approx(101325.0)
    assert reactor.properties["PROP_PF_2"] == pytest.approx(2.0)
    assert reactor.properties["PROP_PF_3"] == pytest.approx(3.0)
    assert created.PreExponentialFactor == pytest.approx(9.2e9)
    assert created.ActivationEnergy == pytest.approx(2.04e5)
    assert created.ReactionOrder == pytest.approx(1.0)


def test_reactor_adapter_fails_without_reactions_collection():
    sim = FakeSimulation()
    config = validate_reactor_config(
        {
            "name": "PEM_Reactor",
            "type": "RCT_Equilibrium",
            "temperature_C": 1400.0,
            "pressure_Pa": 101325.0,
            "mode": "isothermal",
            "reactions": [
                {
                    "name": "Water Gas Shift",
                    "stoichiometry": "CO + H2O -> CO2 + H2",
                    "type": "equilibrium",
                }
            ],
        }
    )

    reactor = SimpleNamespace(
        SetPropertyValue=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ReactorConfigurationError, match="Reactions collection"):
        ReactorAdapter(reactor, sim, config).apply()


def test_configure_gasifier_loads_and_applies_contract_file():
    reactor = FakeReactor()
    sim = FakeSimulation()

    configure_gasifier(reactor, sim)

    assert reactor.properties["PROP_CR_0"] == pytest.approx(101325.0)
    assert reactor.Reactions.ids


def test_configure_pem_loads_and_applies_contract_file():
    reactor = FakeReactor()
    sim = FakeSimulation()

    configure_pem(reactor, sim)

    assert reactor.properties["PROP_EQ_0"] == pytest.approx(101325.0)
    assert reactor.properties["PROP_EQ_1"] == pytest.approx(1673.15)
    assert reactor.Reactions.ids


def test_configure_trc_loads_and_applies_contract_file():
    reactor = FakeReactor()
    sim = FakeSimulation()

    configure_trc(reactor, sim)

    assert reactor.properties["PROP_PF_0"] == pytest.approx(101325.0)
    assert reactor.properties["PROP_PF_2"] == pytest.approx(2.0)
    assert reactor.properties["PROP_PF_3"] == pytest.approx(3.0)
    assert reactor.Reactions.ids
