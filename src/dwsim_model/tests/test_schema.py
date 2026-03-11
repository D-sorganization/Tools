"""
tests/test_schema.py
====================
Unit tests for the Pydantic config validation schemas.

These tests verify that:
  - Valid configs pass without errors
  - Invalid values are caught with informative messages
  - The most common user mistakes are caught (kPa vs Pa, percentages vs fractions)
"""

import pytest
from pydantic import ValidationError

from dwsim_model.config.schema import (
    ReactionEntry,
    ReactorConfig,
    StreamConfig,
    UltimateAnalysis,
    validate_master_config,
    validate_stream_config,
)

# ─────────────────────────────────────────────────────────────────────────────
# UltimateAnalysis
# ─────────────────────────────────────────────────────────────────────────────


class TestUltimateAnalysis:

    def test_valid_analysis_passes(self):
        ua = UltimateAnalysis(C=0.501, H=0.062, O=0.421, N=0.008, S=0.005, Cl=0.003)
        assert pytest.approx(0.501) == ua.C

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="sum"):
            UltimateAnalysis(C=0.5, H=0.5, O=0.5, N=0.0, S=0.0, Cl=0.0)

    def test_fraction_out_of_range(self):
        with pytest.raises(ValidationError):
            UltimateAnalysis(C=1.5, H=0.0, O=0.0, N=0.0, S=0.0, Cl=0.0)

    def test_tolerates_small_rounding_error(self):
        """Sum of 0.999 should pass (tolerance ±0.02)."""
        ua = UltimateAnalysis(C=0.500, H=0.062, O=0.420, N=0.008, S=0.005, Cl=0.004)
        assert ua.C == 0.500

    def test_sulfur_max_capped(self):
        """Sulfur is capped at 0.05 per the schema."""
        with pytest.raises(ValidationError):
            UltimateAnalysis(C=0.44, H=0.06, O=0.42, N=0.01, S=0.07, Cl=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# StreamConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestStreamConfig:

    def test_valid_stream_config(self):
        sc = StreamConfig(
            temperature_C=25.0,
            pressure_Pa=101325.0,
            mass_flow_kg_s=1.0,
        )
        assert sc.pressure_Pa == 101325.0

    def test_low_pressure_caught(self):
        """
        Catches the common mistake of entering pressure in kPa instead of Pa.
        101.325 Pa would be 0.001 atm — clearly wrong.
        """
        with pytest.raises(ValidationError, match="kPa"):
            StreamConfig(pressure_Pa=500.0)  # looks like 500 Pa, probably meant 500 kPa

    def test_components_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="fractions"):
            StreamConfig(
                pressure_Pa=101325,
                components={"CO": 0.9, "H2": 0.5},  # sums to 1.4
            )

    def test_components_near_one_passes(self):
        sc = StreamConfig(
            pressure_Pa=101325,
            components={"CO": 0.35, "H2": 0.25, "CO2": 0.15, "H2O": 0.15, "CH4": 0.10},
        )
        assert "CO" in sc.components

    def test_temperature_below_absolute_zero_rejected(self):
        with pytest.raises(ValidationError):
            StreamConfig(temperature_C=-300.0)

    def test_negative_mass_flow_rejected(self):
        with pytest.raises(ValidationError):
            StreamConfig(mass_flow_kg_s=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ReactorConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestReactorConfig:

    def test_valid_reactor_config(self):
        rc = ReactorConfig(
            name="Gasifier",
            type="RCT_Conversion",
            temperature_C=850.0,
            pressure_Pa=101325.0,
        )
        assert rc.mode == "isothermal"  # default

    def test_reaction_entries(self):
        rc = ReactorConfig(
            name="Gasifier",
            type="RCT_Conversion",
            temperature_C=850.0,
            pressure_Pa=101325.0,
            reactions=[
                ReactionEntry(
                    name="Water Gas",
                    stoichiometry="C + H2O -> CO + H2",
                    base_component="Water",
                    conversion=0.70,
                )
            ],
        )
        assert len(rc.reactions) == 1
        assert rc.reactions[0].conversion == 0.70

    def test_conversion_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ReactionEntry(
                name="Bad",
                stoichiometry="A -> B",
                conversion=1.5,
            )

    def test_low_pressure_rejected_in_reactor(self):
        with pytest.raises(ValidationError):
            ReactorConfig(
                name="Gasifier",
                type="RCT_Conversion",
                temperature_C=850.0,
                pressure_Pa=500.0,  # too low — probably kPa mistake
            )


# ─────────────────────────────────────────────────────────────────────────────
# MasterConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestMasterConfig:

    def test_valid_master_config(self):
        cfg = validate_master_config(
            {
                "reactor_mode": "mixed",
                "compound_set": "standard",
            }
        )
        assert cfg.reactor_mode == "mixed"
        assert cfg.compound_set == "standard"

    def test_invalid_reactor_mode_rejected(self):
        with pytest.raises(ValidationError, match="reactor_mode"):
            validate_master_config(
                {
                    "reactor_mode": "fantasy_mode",
                    "compound_set": "standard",
                }
            )

    def test_invalid_compound_set_rejected(self):
        with pytest.raises(ValidationError, match="compound_set"):
            validate_master_config(
                {
                    "reactor_mode": "mixed",
                    "compound_set": "mega_extended",
                }
            )

    def test_defaults_applied(self):
        """Empty dict should use default values."""
        cfg = validate_master_config({})
        assert cfg.reactor_mode == "mixed"
        assert cfg.compound_set == "standard"

    def test_output_config_defaults(self):
        cfg = validate_master_config({})
        assert "json" in cfg.output.formats
        assert cfg.output.save_dwxml is True


# ─────────────────────────────────────────────────────────────────────────────
# validate_stream_config helper
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateStreamConfigHelper:

    def test_valid_stream_passes(self):
        sc = validate_stream_config(
            {"temperature_C": 200.0, "pressure_Pa": 500000.0}, stream_name="Steam_Feed"
        )
        assert sc.temperature_C == 200.0

    def test_error_includes_stream_name(self):
        with pytest.raises(ValueError, match="My_Stream"):
            validate_stream_config(
                {"pressure_Pa": 100.0}, stream_name="My_Stream"  # too low
            )
