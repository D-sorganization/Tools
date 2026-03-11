from __future__ import annotations

from pathlib import Path

import pytest

from dwsim_model.config_loader import ConfigLoader


def test_load_master_config_resolves_feed_and_energy_subfiles() -> None:
    loader = ConfigLoader(config_path=Path("config/master_config.yaml"))

    config = loader.load()

    assert "Gasifier_Biomass_Feed" in config["feeds"]
    assert "PEM_Steam_Feed" in config["feeds"]
    assert "E_PEM_AC_Power" in config["energy_streams"]
    assert config["model"]["name"] == "Plasma-Assisted Gasification Train"


def test_load_master_config_applies_scenario_overrides() -> None:
    loader = ConfigLoader(config_path=Path("config/master_config.yaml"))

    config = loader.load()

    assert config["targets"]["h2_co_ratio_target"] == pytest.approx(1.8)
    assert config["feeds"]["Gasifier_Biomass_Feed"]["mass_flow_kg_s"] == pytest.approx(
        10.0
    )
    assert config["energy_streams"]["E_PEM_AC_Power"] == pytest.approx(5_000_000.0)


def test_load_master_config_raises_for_invalid_feed_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    feeds_dir = config_dir / "feeds"
    energy_dir = config_dir / "energy"
    scenarios_dir = config_dir / "scenarios"
    feeds_dir.mkdir(parents=True)
    energy_dir.mkdir(parents=True)
    scenarios_dir.mkdir(parents=True)

    (feeds_dir / "gasifier.yaml").write_text(
        "\n".join(
            [
                "Gasifier_Biomass_Feed:",
                "  pressure_Pa: 100.0",
                "  mass_flow_kg_s: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    (energy_dir / "energy.yaml").write_text(
        "energy_streams:\n  E_PEM_AC_Power: 1000.0\n",
        encoding="utf-8",
    )
    (scenarios_dir / "baseline.yaml").write_text(
        "\n".join(
            [
                "scenario:",
                "  name: Baseline",
                "overrides: {}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "master_config.yaml").write_text(
        "\n".join(
            [
                'reactor_mode: "mixed"',
                'compound_set: "standard"',
                "feeds:",
                '  gasifier: "feeds/gasifier.yaml"',
                'energy: "energy/energy.yaml"',
                'scenario: "scenarios/baseline.yaml"',
            ]
        ),
        encoding="utf-8",
    )

    loader = ConfigLoader(config_path=config_dir / "master_config.yaml")

    with pytest.raises(ValueError, match="Gasifier_Biomass_Feed"):
        loader.load()
