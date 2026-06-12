"""Reference-anchored conversion-accuracy tests (#3384, #3388, #3389).

Each test pins an authoritative reference value so the underlying physical fix
cannot silently regress.
"""

from __future__ import annotations

import pytest
from sidekick.calculators.conversion.service import (
    IncompatibleUnitsError,
    UnitConversionService,
)


@pytest.fixture()
def service() -> UnitConversionService:
    return UnitConversionService()


# --------------------------------------------------------------------------- #
# #3384 — bare "nm" resolves to the SI nanometer, not torque N·m
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_bare_nm_is_nanometer_length(service: UnitConversionService) -> None:
    result = service.convert(1.0, "nm", "m")
    assert result.value == pytest.approx(1e-9, rel=1e-12)


@pytest.mark.unit
def test_nanometer_micrometer_roundtrip(service: UnitConversionService) -> None:
    # 1000 nm == 1 micrometre (length table key is the ASCII "um").
    result = service.convert(1000.0, "nm", "um")
    assert result.value == pytest.approx(1.0, rel=1e-9)


@pytest.mark.unit
def test_torque_reachable_via_unambiguous_spelling(
    service: UnitConversionService,
) -> None:
    # The torque base unit stays usable via non-colliding spellings. "kN·m"
    # cleans to "knm" (distinct from length "nm"), so torque-to-torque works.
    assert service.convert(1.0, "newton_meter", "kN·m").value == pytest.approx(
        0.001, rel=1e-9
    )
    assert service.convert(1000.0, "newton_metre", "kN·m").value == pytest.approx(
        1.0, rel=1e-9
    )


@pytest.mark.unit
def test_nm_not_treated_as_torque(service: UnitConversionService) -> None:
    # Bare "nm" (length) is incompatible with a torque target — proving it is no
    # longer silently classified as torque.
    with pytest.raises(IncompatibleUnitsError):
        service.convert(1.0, "nm", "kN·m")


# --------------------------------------------------------------------------- #
# #3388 — BTU/SCF (60 °F) <-> MJ/Nm³ (0 °C) applies the molar-volume ratio
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.scientific
def test_btu_scf_to_mj_nm3_methane_reference(service: UnitConversionService) -> None:
    # 1010 BTU/SCF methane HHV -> ~39.8 MJ/Nm³ (GPSA Engineering Data Book),
    # not the old basis-ignoring 37.63 MJ/Nm³ (-5.4%).
    value = service.heating_value(1010.0, "BTU/SCF", "MJ/Nm3", gas_density_stp=0.6774)
    assert value == pytest.approx(39.8, abs=0.1)


@pytest.mark.unit
def test_btu_scf_mj_nm3_roundtrip(service: UnitConversionService) -> None:
    mj = service.heating_value(1010.0, "BTU/SCF", "MJ/Nm3", gas_density_stp=0.6774)
    back = service.heating_value(mj, "MJ/Nm3", "BTU/SCF", gas_density_stp=0.6774)
    assert back == pytest.approx(1010.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# #3389 (ppm part) — ppmv <-> mg/Nm³ uses the 0 °C molar volume (22.414 L/mol)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.scientific
def test_ppmv_benzene_to_mg_nm3_reference(service: UnitConversionService) -> None:
    # 1 ppmv benzene (MW 78.11) -> 78.11 / 22.414 = 3.4849 mg/Nm³ (0 °C basis),
    # not 3.195 mg/Nm³ from the 25 °C molar volume (24.45 L/mol).
    value = service.tar_concentration(1.0, "ppm_mass", "mg/Nm3", molecular_weight=78.11)
    assert value == pytest.approx(3.4849, abs=1e-3)


@pytest.mark.unit
def test_ppmv_mg_nm3_roundtrip(service: UnitConversionService) -> None:
    mg = service.tar_concentration(1.0, "ppm_mass", "mg/Nm3", molecular_weight=78.11)
    back = service.tar_concentration(mg, "mg/Nm3", "ppm_mass", molecular_weight=78.11)
    assert back == pytest.approx(1.0, rel=1e-9)
