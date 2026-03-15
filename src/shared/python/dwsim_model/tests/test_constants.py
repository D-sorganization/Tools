from dwsim_model.constants import (
    C2_HYDROCARBONS,
    COMPOUNDS_EXTENDED,
    COMPOUNDS_MINIMAL,
    COMPOUNDS_STANDARD,
    DEFAULT_PROPERTY_PACKAGE,
    ENERGY_STREAM_NAMES,
    INERTS,
    KELVIN_OFFSET,
    PP_PENG_ROBINSON,
    PP_SRK,
    STANDARD_PRESSURE_PA,
    STANDARD_TEMPERATURE_C,
    STREAM_NAMES,
    SYNGAS_CORE,
    TAR_SURROGATES,
    TRACE_CONTAMINANTS,
)


def test_compound_lists_not_empty():
    assert len(SYNGAS_CORE) > 0
    assert len(C2_HYDROCARBONS) > 0
    assert len(TAR_SURROGATES) > 0
    assert len(TRACE_CONTAMINANTS) > 0
    assert len(INERTS) > 0


def test_compound_aggregates():
    assert len(COMPOUNDS_MINIMAL) > len(SYNGAS_CORE)
    assert len(COMPOUNDS_STANDARD) > len(COMPOUNDS_MINIMAL)
    assert len(COMPOUNDS_EXTENDED) > len(COMPOUNDS_STANDARD)


def test_constants_values():
    assert KELVIN_OFFSET == 273.15
    assert STANDARD_PRESSURE_PA == 101325.0
    assert STANDARD_TEMPERATURE_C == 15.0


def test_property_packages_defined():
    assert PP_PENG_ROBINSON == "Peng-Robinson (PR)"
    assert PP_SRK == "Soave-Redlich-Kwong (SRK)"
    assert DEFAULT_PROPERTY_PACKAGE in (PP_PENG_ROBINSON, PP_SRK)


def test_stream_names_dictionaries():
    assert "biomass_feed" in STREAM_NAMES
    assert "quench_water" in STREAM_NAMES
    assert "gasifier_heat_loss" in ENERGY_STREAM_NAMES
    assert "pem_ac_power" in ENERGY_STREAM_NAMES
