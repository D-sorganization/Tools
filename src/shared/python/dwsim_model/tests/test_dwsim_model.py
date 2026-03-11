"""Pure-Python unit tests for dwsim_model shared library.

These tests run without a DWSIM runtime installation and verify the
constants, chemistry, and package metadata that are pure Python.
"""

from __future__ import annotations

import pytest

# ─── Package metadata ──────────────────────────────────────────────────────────


def test_package_version_exposed():
    import dwsim_model

    assert hasattr(dwsim_model, "__version__")
    assert isinstance(dwsim_model.__version__, str)
    assert dwsim_model.__version__


# ─── Constants ─────────────────────────────────────────────────────────────────


def test_syngas_core_contains_key_species():
    from dwsim_model.constants import SYNGAS_CORE

    assert "Carbon monoxide" in SYNGAS_CORE
    assert "Hydrogen" in SYNGAS_CORE
    assert "Carbon dioxide" in SYNGAS_CORE
    assert "Water" in SYNGAS_CORE


def test_compound_sets_are_supersets():
    from dwsim_model.constants import (
        COMPOUNDS_EXTENDED,
        COMPOUNDS_MINIMAL,
        COMPOUNDS_STANDARD,
        SYNGAS_CORE,
    )

    core = set(SYNGAS_CORE)
    assert core.issubset(set(COMPOUNDS_MINIMAL))
    assert core.issubset(set(COMPOUNDS_STANDARD))
    assert core.issubset(set(COMPOUNDS_EXTENDED))
    assert set(COMPOUNDS_STANDARD).issubset(set(COMPOUNDS_EXTENDED))


def test_kelvin_offset_value():
    from dwsim_model.constants import KELVIN_OFFSET

    assert KELVIN_OFFSET == pytest.approx(273.15)


def test_standard_pressure_is_one_atm():
    from dwsim_model.constants import STANDARD_PRESSURE_PA

    assert STANDARD_PRESSURE_PA == pytest.approx(101_325.0)


def test_stream_names_has_required_keys():
    from dwsim_model.constants import STREAM_NAMES

    required = {"biomass_feed", "final_syngas", "gasifier_oxygen"}
    assert required.issubset(set(STREAM_NAMES.keys()))


def test_default_property_package_is_peng_robinson():
    from dwsim_model.constants import DEFAULT_PROPERTY_PACKAGE, PP_PENG_ROBINSON

    assert DEFAULT_PROPERTY_PACKAGE == PP_PENG_ROBINSON


# ─── Chemistry / BiomassDecomposer ─────────────────────────────────────────────


def test_biomass_feed_default_ultimate_sums_to_one():
    from dwsim_model.chemistry.biomass_decomposer import BiomassFeed

    feed = BiomassFeed()
    total = sum(feed.ultimate_daf.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_decompose_returns_dict_summing_to_one():
    from dwsim_model.chemistry.biomass_decomposer import BiomassDecomposer, BiomassFeed

    feed = BiomassFeed()
    dec = BiomassDecomposer()
    result = dec.decompose(feed)

    assert isinstance(result, dict)
    assert len(result) > 0
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-4)


def test_decompose_all_fractions_non_negative():
    from dwsim_model.chemistry.biomass_decomposer import BiomassDecomposer, BiomassFeed

    feed = BiomassFeed()
    dec = BiomassDecomposer()
    result = dec.decompose(feed)

    for compound, frac in result.items():
        assert frac >= 0.0, f"{compound} has negative mole fraction {frac}"


def test_decompose_raises_on_invalid_moisture_ash():
    from dwsim_model.chemistry.biomass_decomposer import BiomassDecomposer, BiomassFeed

    # moisture=0.55 + ash=0.50 → daf_frac = -0.05, passes __post_init__ but fails decompose
    feed = BiomassFeed(moisture_ar=0.55, ash_ar=0.50)
    dec = BiomassDecomposer()

    with pytest.raises(ValueError, match="daf_frac"):
        dec.decompose(feed)


@pytest.mark.parametrize(
    "moisture,ash",
    [
        (0.0, 0.0),
        (0.10, 0.05),
        (0.20, 0.10),
    ],
)
def test_decompose_various_moisture_levels(moisture: float, ash: float):
    from dwsim_model.chemistry.biomass_decomposer import BiomassDecomposer, BiomassFeed

    feed = BiomassFeed(moisture_ar=moisture, ash_ar=ash)
    dec = BiomassDecomposer()
    result = dec.decompose(feed)

    assert sum(result.values()) == pytest.approx(1.0, abs=1e-4)
