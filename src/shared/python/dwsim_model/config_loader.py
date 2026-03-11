"""
config_loader.py
================
Loads simulation parameters from external config files (YAML or legacy JSON)
so users can adjust feeds, energy values, and reactor settings without editing
Python code.

Upgrade notes vs. original version
-----------------------------------
* BUG FIX: ``apply_to_flowsheet`` now actually applies stream compositions.
* BUG FIX: Path resolution uses ``pathlib`` — no longer breaks when the script
  is run from a directory other than the project root.
* YAML support: reads ``.yaml`` / ``.yml`` files in addition to ``.json``.
* Unit awareness: temperature accepts both ``temperature_C`` and ``temperature_K``.
* Structured error reporting: collects all failures and reports them together.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dwsim_model.config.schema import (
    MasterConfig,
    ScenarioConfig,
    validate_master_config,
    validate_reactor_config,
    validate_stream_config,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_default_config() -> Path | None:
    """
    Walk up from this file's location looking for a known config file.

    Search order:
    1. ``<project_root>/config/master_config.yaml``
    2. ``<project_root>/config/master_config.yml``
    3. ``<project_root>/config/feed_conditions.json``  (legacy)
    """
    here = Path(__file__).resolve().parent  # .../src/dwsim_model/
    project_root = here.parent.parent  # .../DWSIM_Model/
    config_dir = project_root / "config"

    candidates = [
        config_dir / "master_config.yaml",
        config_dir / "master_config.yml",
        config_dir / "feed_conditions.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_file(path: Path) -> dict:
    """Load a YAML or JSON file and return its contents as a dict."""
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as fh:
        if suffix in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(fh) or {}
        elif suffix == ".json":
            return json.load(fh)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* and return the merged dict."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _is_runtime_config(raw: dict[str, Any]) -> bool:
    """Return True when the config is already expanded to stream dictionaries."""
    feeds = raw.get("feeds")
    if isinstance(feeds, dict) and feeds:
        return all(isinstance(value, dict) for value in feeds.values())
    return "energy_streams" in raw


# ─────────────────────────────────────────────────────────────────────────────
# ConfigLoader
# ─────────────────────────────────────────────────────────────────────────────


class ConfigLoader:
    """
    Parse and apply external configuration files to a DWSIM flowsheet.

    Parameters
    ----------
    config_path:
        Explicit path to the config file.  If None, the loader searches for a
        default config (see :func:`_find_default_config`).
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config_data: dict[str, Any] | None = None,
    ):
        if config_path is not None:
            self.config_path: Path | None = Path(config_path)
        else:
            self.config_path = _find_default_config()

        self._config_data = dict(config_data) if config_data is not None else None
        self.config: dict[str, Any] = {}
        self._errors: list[str] = []

    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> dict:
        """Load and return the configuration dictionary."""
        if self._config_data is not None:
            self.config = self._resolve_sub_configs(self._config_data)
            logger.info(
                "Config loaded from runtime data: "
                f"{len(self.config.get('feeds', {}))} feed streams, "
                f"{len(self.config.get('energy_streams', {}))} energy streams."
            )
            return self.config

        if self.config_path is None or not self.config_path.exists():
            logger.warning(
                "No config file found — using DWSIM defaults for all streams. "
                "Create config/master_config.yaml to customise the simulation."
            )
            return {}

        logger.info(f"Loading config from: {self.config_path}")
        try:
            raw = _load_file(self.config_path)
        except Exception as exc:
            logger.error(f"Failed to read config file {self.config_path}: {exc}")
            return {}

        self.config = self._resolve_sub_configs(raw)
        logger.info(
            f"Config loaded: "
            f"{len(self.config.get('feeds', {}))} feed streams, "
            f"{len(self.config.get('energy_streams', {}))} energy streams."
        )
        return self.config

    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_sub_configs(self, raw: dict) -> dict:
        """
        If ``raw`` contains file-path references (master_config style), load
        those files and merge them.  If ``raw`` already has a ``feeds`` key
        (legacy JSON style), return it unchanged.
        """
        if _is_runtime_config(raw):
            resolved = dict(raw)
        else:
            master = validate_master_config(raw)
            resolved = self._expand_master_config(master, raw)

        self._validate_resolved_config(resolved)
        return resolved

    def _expand_master_config(
        self, master: MasterConfig, raw: dict[str, Any]
    ) -> dict[str, Any]:
        """Expand master config references into a runtime-ready dictionary."""
        resolved: dict[str, Any] = {
            "model": master.model.model_dump(),
            "reactor_mode": master.reactor_mode,
            "compound_set": master.compound_set,
            "feeds": {},
            "reactors": {},
            "energy_streams": {},
            "equipment": {},
            "output": master.output.model_dump(),
            "scenario": {},
            "targets": {},
        }

        for _section, sub_path in master.feeds.items():
            sub = _load_file(self._resolve_ref_path(sub_path))
            for stream_name, props in sub.items():
                if isinstance(props, dict):
                    resolved["feeds"][stream_name] = props

        for reactor_name, sub_path in master.reactors.items():
            sub = _load_file(self._resolve_ref_path(sub_path))
            resolved["reactors"][reactor_name] = sub

        if master.energy:
            energy_data = _load_file(self._resolve_ref_path(master.energy))
            resolved["energy_streams"].update(energy_data.get("energy_streams", {}))

        if master.equipment:
            resolved["equipment"] = _load_file(self._resolve_ref_path(master.equipment))

        scenario_ref = raw.get("scenario")
        if scenario_ref:
            scenario_data = _load_file(self._resolve_ref_path(str(scenario_ref)))
            resolved["scenario"] = self._validate_scenario_config(
                scenario_data
            ).model_dump()
            overrides = scenario_data.get("overrides", {})
            resolved = _deep_merge(resolved, overrides)
            resolved["targets"] = scenario_data.get("targets", {})

        return resolved

    def _resolve_ref_path(self, ref: str | Path) -> Path:
        """Resolve config references relative to either the config dir or project root."""
        ref_path = Path(ref)
        if ref_path.is_absolute():
            return ref_path
        if self.config_path is None:
            return ref_path

        config_dir = self.config_path.parent
        project_root = config_dir.parent
        candidates = [config_dir / ref_path, project_root / ref_path]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Referenced config file not found: {ref_path}")

    def _validate_scenario_config(
        self, scenario_data: dict[str, Any]
    ) -> ScenarioConfig:
        """Validate a scenario config block."""
        scenario = scenario_data.get("scenario", {})
        return ScenarioConfig.model_validate(
            {
                "name": scenario.get("name", "unnamed"),
                "description": scenario.get("description", ""),
                "overrides": scenario_data.get("overrides", {}),
                "targets": scenario_data.get("targets"),
            }
        )

    def _validate_resolved_config(self, resolved: dict[str, Any]) -> None:
        """Validate resolved streams and reactors before runtime application."""
        for stream_name, props in resolved.get("feeds", {}).items():
            validate_stream_config(props, stream_name=stream_name)

        for reactor_name, reactor_block in resolved.get("reactors", {}).items():
            reactor_payload = dict(reactor_block.get("reactor", {}))
            reactor_payload["reactions"] = reactor_block.get("reactions", [])
            validate_reactor_config(
                reactor_payload,
                reactor_name=reactor_payload.get("name", reactor_name),
            )

    # ─────────────────────────────────────────────────────────────────────────

    def apply_to_flowsheet(
        self,
        builder,
        materials: dict,
        energy_streams: dict,
    ) -> None:
        """
        Push loaded configuration values into DWSIM stream objects.

        Parameters
        ----------
        builder:
            The FlowsheetBuilder instance.
        materials:
            Dict mapping stream name → DWSIM MaterialStream object.
        energy_streams:
            Dict mapping stream name → DWSIM EnergyStream object.
            NOTE: The original code passed ``b.energies`` here — that attribute
            does not exist.  The correct attribute is ``b.energy_streams``.
        """
        if not self.config:
            logger.debug("apply_to_flowsheet: config is empty, nothing to apply.")
            return

        self._errors = []
        self._apply_material_streams(materials, self.config.get("feeds", {}))
        self._apply_energy_streams(
            energy_streams, self.config.get("energy_streams", {})
        )

        if self._errors:
            logger.warning(
                f"Config application finished with {len(self._errors)} error(s):\n"
                + "\n".join(f"  • {e}" for e in self._errors)
            )
        else:
            logger.info("All config values applied successfully.")

    # ─────────────────────────────────────────────────────────────────────────

    def _apply_material_streams(self, materials: dict, feeds: dict) -> None:
        """Apply T, P, flow, and composition to material streams."""
        for stream_name, props in feeds.items():
            if stream_name not in materials:
                self._errors.append(
                    f"Stream '{stream_name}' in config not found in flowsheet."
                )
                continue

            stream = materials[stream_name]
            try:
                self._set_stream_conditions(stream, stream_name, props)
                self._set_stream_composition(stream, stream_name, props)
            except Exception as exc:
                self._errors.append(f"{stream_name}: {exc}")

    def _set_stream_conditions(self, stream, name: str, props: dict) -> None:
        """Set T, P, and mass-flow on a material stream."""
        if "temperature_C" in props:
            t_k = float(props["temperature_C"]) + 273.15
            stream.SetPropertyValue("Temperature", t_k)
            logger.debug(f"{name}: T = {props['temperature_C']} °C ({t_k:.2f} K)")
        elif "temperature_K" in props:
            stream.SetPropertyValue("Temperature", float(props["temperature_K"]))

        if "pressure_Pa" in props:
            stream.SetPropertyValue("Pressure", float(props["pressure_Pa"]))

        if "mass_flow_kg_s" in props:
            stream.SetPropertyValue("MassFlow", float(props["mass_flow_kg_s"]))

    def _set_stream_composition(self, stream, name: str, props: dict) -> None:
        """
        Apply mole fractions from a ``components`` dict.

        Fractions are normalised if they don't sum to exactly 1.0
        (tolerance ±0.02).
        """
        components: dict = props.get("components", {})
        if not components:
            return

        total = sum(float(v) for v in components.values())
        if total == 0:
            self._errors.append(f"{name}: all component fractions are zero — skipped.")
            return

        if abs(total - 1.0) > 0.02:
            logger.warning(
                f"{name}: component fractions sum to {total:.4f} — normalising."
            )

        for compound, fraction in components.items():
            norm_frac = float(fraction) / total
            try:
                stream.SetPropertyValue(f"MoleFraction.{compound}", norm_frac)
                logger.debug(f"{name}: x({compound}) = {norm_frac:.4f}")
            except Exception as exc:
                logger.warning(
                    f"{name}: could not set fraction for '{compound}': {exc}"
                )

    def _apply_energy_streams(self, energy_streams: dict, energy_config: dict) -> None:
        """Apply power values to energy streams."""
        for e_name, e_val in energy_config.items():
            if e_name not in energy_streams:
                self._errors.append(
                    f"Energy stream '{e_name}' in config not found in flowsheet."
                )
                continue
            try:
                self._set_energy_stream_value(energy_streams[e_name], float(e_val))
                logger.info(f"Applied {e_val} W to energy stream '{e_name}'.")
            except Exception as exc:
                self._errors.append(f"Energy stream {e_name}: {exc}")

    @staticmethod
    def _set_energy_stream_value(stream, value_watts: float) -> None:
        """Apply an energy flow using the DWSIM property identifier supported by the runtime."""
        try:
            stream.SetPropertyValue("PROP_ES_0", value_watts / 1000.0)
            return
        except Exception:
            pass

        stream.SetPropertyValue("EnergyFlow", value_watts)
