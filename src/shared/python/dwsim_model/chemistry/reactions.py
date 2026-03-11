"""
chemistry/reactions.py
=====================
Validated reactor-configuration adapters for the gasification train.

This module is the Python-side contract between YAML reactor definitions and
the DWSIM automation API. Unsupported API paths raise explicit errors instead
of downgrading to "manual GUI setup required".
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from dwsim_model.config.schema import ReactorConfig, validate_reactor_config

logger = logging.getLogger(__name__)

_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "config" / "reactors"
)

_OPERATION_MODES = {
    "isothermal": 0,
    "adiabatic": 1,
}

_PROPERTY_MAP = {
    "RCT_Conversion": {
        "pressure_Pa": "PROP_CR_0",
    },
    "RCT_Equilibrium": {
        "pressure_Pa": "PROP_EQ_0",
        "temperature_K": "PROP_EQ_1",
    },
    "RCT_PFR": {
        "pressure_Pa": "PROP_PF_0",
        "volume_m3": "PROP_PF_2",
        "length_m": "PROP_PF_3",
    },
}


class ReactorConfigurationError(RuntimeError):
    """Raised when the reactor contract cannot be applied to the runtime."""


def _load_reactor_contract(filename: str) -> ReactorConfig:
    """Load and validate a reactor YAML file as a runtime contract."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise ReactorConfigurationError(f"Reactor config not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    reactor_payload = dict(raw.get("reactor", {}))
    reactor_payload["reactions"] = raw.get("reactions", [])

    return validate_reactor_config(
        reactor_payload,
        reactor_name=reactor_payload.get("name", path.stem),
    )


class ReactorAdapter:
    """Adapter that applies a validated ReactorConfig to a DWSIM reactor."""

    def __init__(self, reactor_obj, sim, config: ReactorConfig):
        self.reactor_obj = reactor_obj
        self.sim = sim
        self.config = config

    def apply(self) -> None:
        self._set_temperature()
        self._set_pressure()
        self._set_operation_mode()
        self._set_geometry()

        for reaction in self.config.reactions:
            self._add_reaction(reaction)

    def _set_temperature(self) -> None:
        temperature_k = float(self.config.temperature_C) + 273.15
        self._set_mapped_property("temperature_K", temperature_k, required=False)

    def _set_pressure(self) -> None:
        pressure_pa = float(self.config.pressure_Pa)
        self._set_mapped_property("pressure_Pa", pressure_pa, required=True)

    def _set_operation_mode(self) -> None:
        if self.config.mode not in _OPERATION_MODES:
            if self.config.mode == "specified_duty":
                raise ReactorConfigurationError(
                    f"{self.config.name}: specified_duty mode is not supported yet."
                )
            raise ReactorConfigurationError(
                f"{self.config.name}: unsupported reactor mode '{self.config.mode}'."
            )

        self._try_set_property("Calculation Mode", _OPERATION_MODES[self.config.mode])

    def _set_geometry(self) -> None:
        geometry = {
            "volume_m3": self.config.volume_m3,
            "length_m": self.config.length_m,
            "diameter_m": self.config.diameter_m,
        }

        for field_name, value in geometry.items():
            if value is None:
                continue
            self._set_mapped_property(field_name, float(value), required=False)

    def _add_reaction(self, reaction) -> None:
        reaction_type = self._resolve_reaction_type()

        try:
            reaction_obj = self.sim.AddReaction(
                reaction.name,
                reaction_type,
                reaction.base_component or "",
                float(reaction.conversion or 0.0),
            )
        except AttributeError as exc:
            raise ReactorConfigurationError(
                f"{self.config.name}: simulation runtime does not support AddReaction."
            ) from exc
        except Exception as exc:
            raise ReactorConfigurationError(
                f"{self.config.name}: failed to create reaction '{reaction.name}': {exc}"
            ) from exc

        if reaction_obj is None:
            raise ReactorConfigurationError(
                f"{self.config.name}: AddReaction returned None for '{reaction.name}'."
            )

        self._apply_reaction_details(reaction_obj, reaction)
        self._attach_reaction(reaction_obj)

    def _resolve_reaction_type(self) -> str:
        if self.config.type == "RCT_Conversion":
            return "Conversion"
        if self.config.type == "RCT_Equilibrium":
            return "Equilibrium"
        if self.config.type == "RCT_PFR":
            return "Kinetic"
        raise ReactorConfigurationError(
            f"{self.config.name}: unsupported DWSIM reactor type '{self.config.type}'."
        )

    def _apply_reaction_details(self, reaction_obj, reaction) -> None:
        if reaction.kinetics is None:
            return

        details = {
            "PreExponentialFactor": reaction.kinetics.pre_exponential_A,
            "ActivationEnergy": reaction.kinetics.activation_energy_J_mol,
            "ReactionOrder": reaction.kinetics.reaction_order_n,
        }
        for attr_name, value in details.items():
            if not self._try_attr_on(reaction_obj, attr_name, float(value)):
                raise ReactorConfigurationError(
                    f"{self.config.name}: reaction '{reaction.name}' does not support "
                    f"kinetic field '{attr_name}'."
                )

    def _attach_reaction(self, reaction_obj) -> None:
        reactions = getattr(self.reactor_obj, "Reactions", None)
        if reactions is None or not hasattr(reactions, "Add"):
            raise ReactorConfigurationError(
                f"{self.config.name}: reactor does not expose a writable Reactions collection."
            )

        reaction_id = getattr(reaction_obj, "ID", None)
        if reaction_id is None:
            raise ReactorConfigurationError(
                f"{self.config.name}: reaction object missing ID for attachment."
            )

        try:
            reactions.Add(reaction_id)
        except Exception as exc:
            raise ReactorConfigurationError(
                f"{self.config.name}: failed to attach reaction '{reaction_id}': {exc}"
            ) from exc

    @staticmethod
    def _try_attr_on(target, attr_name: str, value: float) -> bool:
        if not hasattr(target, attr_name):
            return False
        try:
            setattr(target, attr_name, value)
            return True
        except Exception:
            return False

    def _set_mapped_property(
        self, field_name: str, value: float, *, required: bool = False
    ) -> bool:
        property_name = _PROPERTY_MAP.get(self.config.type, {}).get(field_name)
        if property_name is None:
            if required:
                raise ReactorConfigurationError(
                    f"{self.config.name}: no supported runtime property for '{field_name}'."
                )
            logger.info(
                "%s: field '%s' is currently metadata-only for reactor type %s.",
                self.config.name,
                field_name,
                self.config.type,
            )
            return False

        return self._try_set_property(property_name, value, required=required)

    def _try_set_property(
        self, prop_name: str, value: float, *, required: bool = False
    ) -> bool:
        if not hasattr(self.reactor_obj, "SetPropertyValue"):
            if required:
                raise ReactorConfigurationError(
                    f"{self.config.name}: reactor has no SetPropertyValue for '{prop_name}'."
                )
            return False

        try:
            self.reactor_obj.SetPropertyValue(prop_name, value)
            return True
        except Exception as exc:
            if required:
                raise ReactorConfigurationError(
                    f"{self.config.name}: failed to set '{prop_name}' via SetPropertyValue: {exc}"
                ) from exc
            return False


def configure_gasifier(gasifier_obj, sim) -> None:
    config = _load_reactor_contract("gasifier_reactions.yaml")
    ReactorAdapter(gasifier_obj, sim, config).apply()
    logger.info("Gasifier configured with %d reactions.", len(config.reactions))


def configure_pem(pem_obj, sim) -> None:
    config = _load_reactor_contract("pem_reactions.yaml")
    ReactorAdapter(pem_obj, sim, config).apply()
    logger.info("PEM configured with %d reactions.", len(config.reactions))


def configure_trc(trc_obj, sim) -> None:
    config = _load_reactor_contract("trc_reactions.yaml")
    ReactorAdapter(trc_obj, sim, config).apply()
    logger.info("TRC configured with %d reactions.", len(config.reactions))


def print_reaction_summary() -> None:
    """Print a summary of validated reactor contracts."""
    for label, filename in (
        ("Gasifier", "gasifier_reactions.yaml"),
        ("PEM", "pem_reactions.yaml"),
        ("TRC", "trc_reactions.yaml"),
    ):
        config = _load_reactor_contract(filename)
        print(f"\n{label} ({config.type}):")
        print(f"  Temperature: {config.temperature_C} C")
        print(f"  Pressure: {config.pressure_Pa} Pa")
        for reaction in config.reactions:
            print(f"  - {reaction.name}")


if __name__ == "__main__":
    print_reaction_summary()
