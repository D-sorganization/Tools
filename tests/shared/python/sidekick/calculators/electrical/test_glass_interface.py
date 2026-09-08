"""Contract tests for GlassPropertiesInterface provider validation and fallback.

Covers issue #5062: invalid provider output (NaN/Inf/zero/negative), provider
exceptions, absolute-zero temperature domain, cache capacity contracts, failed
response cache hygiene, and provider-switch cache invalidation. Dimensional
oracles: 1 S/cm = 100 S/m and sigma*rho = 1 in SI.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pytest
from sidekick.calculators.electrical.glass_contracts import (
    ConductivityProvider,
    GlassFallbackPolicy,
    celsius_to_kelvin,
    s_per_cm_to_s_per_m,
)
from sidekick.calculators.electrical.glass_interface import GlassPropertiesInterface


class ScriptedProvider:
    """Test double implementing the public ConductivityProvider protocol.

    Each call consumes one script entry: a float conductivity (S/m) or an
    Exception instance to raise. Records the Kelvin temperature of each call.
    """

    def __init__(self, script: list[float | Exception]) -> None:
        self.script: list[float | Exception] = list(script)
        self.calls: list[float] = []

    def provide_conductivity(
        self,
        temperature_kelvin: float,
        composition: dict[str, float] | None,
        power_density: float,
    ) -> float:
        self.calls.append(temperature_kelvin)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture()
def good_provider() -> ScriptedProvider:
    return ScriptedProvider([50.0, 60.0, 70.0])


def test_provider_protocol_is_structural(good_provider: ScriptedProvider) -> None:
    assert isinstance(good_provider, ConductivityProvider)


def test_unit_conversion_oracle_s_per_cm_to_si() -> None:
    assert s_per_cm_to_s_per_m(1.0) == pytest.approx(100.0)
    assert s_per_cm_to_s_per_m(0.05) == pytest.approx(5.0)


def test_sigma_times_rho_is_one_si_oracle() -> None:
    interface = GlassPropertiesInterface()
    conductivity = interface.get_conductivity(1200.0)
    resistivity = interface.get_resistivity(1200.0)
    assert conductivity * resistivity == pytest.approx(1.0)


def test_provider_path_sigma_times_rho_is_one(
    good_provider: ScriptedProvider,
) -> None:
    interface = GlassPropertiesInterface(conductivity_provider=good_provider)
    conductivity = interface.get_conductivity(1000.0, composition={"SiO2": 1.0})
    assert conductivity == pytest.approx(50.0)
    assert interface.get_resistivity(1000.0, composition={"SiO2": 1.0}) == (
        pytest.approx(1.0 / 50.0)
    )


def test_provider_receives_kelvin(good_provider: ScriptedProvider) -> None:
    interface = GlassPropertiesInterface(conductivity_provider=good_provider)
    interface.get_conductivity(1000.0)
    assert good_provider.calls == [pytest.approx(1000.0 + 273.15)]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_strict_mode_rejects_invalid_provider_output(bad: float) -> None:
    provider = ScriptedProvider([bad])
    interface = GlassPropertiesInterface(
        conductivity_provider=provider,
        fallback_policy=GlassFallbackPolicy.STRICT,
    )
    with pytest.raises(ValueError, match="conductivity"):
        interface.get_conductivity(1000.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_demo_mode_rejects_invalid_output_without_silent_pass_through(
    bad: float,
) -> None:
    provider = ScriptedProvider([bad])
    interface = GlassPropertiesInterface(
        conductivity_provider=provider,
        fallback_policy=GlassFallbackPolicy.DEMO,
    )
    result = interface.get_conductivity(1000.0)
    assert math.isfinite(result)
    assert result > 0.0
    report = interface.get_last_provider_report()
    assert report is not None
    assert report.fallback_reason is not None
    assert report.source == "default_model"


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad composition"),
        TypeError("wrong argument type"),
        ZeroDivisionError("division by zero"),
    ],
)
def test_strict_mode_surfaces_provider_exception(exc: Exception) -> None:
    provider = ScriptedProvider([exc])
    interface = GlassPropertiesInterface(
        conductivity_provider=provider,
        fallback_policy=GlassFallbackPolicy.STRICT,
    )
    with pytest.raises(type(exc)):
        interface.get_conductivity(1000.0)


def test_demo_mode_records_provider_provenance_and_reason() -> None:
    provider = ScriptedProvider([ValueError("bad composition")])
    interface = GlassPropertiesInterface(
        conductivity_provider=provider,
        fallback_policy=GlassFallbackPolicy.DEMO,
    )
    result = interface.get_conductivity(1000.0)
    assert result == pytest.approx(GlassPropertiesInterface().get_conductivity(1000.0))
    report = interface.get_last_provider_report()
    assert report is not None
    assert report.fallback_reason is not None
    assert "ValueError" in report.fallback_reason


def test_legacy_policy_warns_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = ScriptedProvider([ValueError("bad composition")])
    interface = GlassPropertiesInterface(
        conductivity_provider=provider,
        fallback_policy=GlassFallbackPolicy.LEGACY,
    )
    with caplog.at_level(logging.WARNING):
        result = interface.get_conductivity(1000.0)
    assert result > 0.0
    assert any("failed" in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize(
    "temperature_celsius",
    [-273.15, -300.0, -273.15 - 1e-6, float("nan"), float("inf"), None],
)
def test_temperature_at_or_below_absolute_zero_rejected(
    temperature_celsius: Any,
) -> None:
    interface = GlassPropertiesInterface()
    with pytest.raises((ValueError, TypeError)):
        interface.get_conductivity(temperature_celsius)


def test_temperature_just_above_absolute_zero_reaches_provider() -> None:
    provider = ScriptedProvider([50.0])
    interface = GlassPropertiesInterface(conductivity_provider=provider)
    assert interface.get_conductivity(-273.0) == pytest.approx(50.0)
    assert provider.calls[0] == pytest.approx(0.15)


def test_kelvin_conversion_rejects_non_physical_domain() -> None:
    with pytest.raises(ValueError):
        celsius_to_kelvin(-273.15)
    with pytest.raises(ValueError):
        celsius_to_kelvin(float("nan"))


@pytest.mark.parametrize("bad_size", [0, -1, 1.5, True, "10", None, float("nan")])
def test_invalid_cache_capacity_rejected(bad_size: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        GlassPropertiesInterface(cache_max_size=bad_size)


def test_failed_provider_response_is_not_cached() -> None:
    provider = ScriptedProvider([float("nan"), 50.0])
    interface = GlassPropertiesInterface(
        conductivity_provider=provider,
        fallback_policy=GlassFallbackPolicy.DEMO,
    )
    first = interface.get_conductivity(1000.0)
    second = interface.get_conductivity(1000.0)
    assert len(provider.calls) == 2
    assert second == pytest.approx(50.0)
    assert all(math.isfinite(value) for value in second.__class__ and [first])
    cached_values = list(interface._temperature_dependent_data.values())
    assert all(value > 0.0 and math.isfinite(value) for value in cached_values)


def test_provider_switch_invalidates_cache() -> None:
    first = ScriptedProvider([10.0])
    interface = GlassPropertiesInterface(
        conductivity_provider=first,
        fallback_policy=GlassFallbackPolicy.STRICT,
    )
    assert interface.get_conductivity(1000.0) == pytest.approx(10.0)
    second = ScriptedProvider([20.0])
    interface.set_conductivity_provider(second)
    assert interface.get_conductivity(1000.0) == pytest.approx(20.0)
    assert len(second.calls) == 1


def test_legacy_calculator_switch_invalidates_cache() -> None:
    interface = GlassPropertiesInterface(
        external_calculator=lambda temp, comp, power: 10.0
    )
    assert interface.get_conductivity(1000.0) == pytest.approx(10.0)
    interface.set_external_calculator(lambda temp, comp, power: 20.0)
    assert interface.get_conductivity(1000.0) == pytest.approx(20.0)


def test_composition_domain_validated_before_provider_call() -> None:
    provider = ScriptedProvider([50.0])
    interface = GlassPropertiesInterface(conductivity_provider=provider)
    with pytest.raises(ValueError):
        interface.get_conductivity(1000.0, composition={"SiO2": float("nan")})
    with pytest.raises(TypeError):
        interface.get_conductivity(1000.0, composition={1: 0.5})  # type: ignore[dict-item]
    assert provider.calls == []


def test_provider_and_legacy_calculator_are_mutually_exclusive(
    good_provider: ScriptedProvider,
) -> None:
    with pytest.raises(ValueError):
        GlassPropertiesInterface(
            external_calculator=lambda temp, comp, power: 1.0,
            conductivity_provider=good_provider,
        )


def test_resistivity_never_returns_infinity_for_rejected_output() -> None:
    provider = ScriptedProvider([0.0])
    interface = GlassPropertiesInterface(
        conductivity_provider=provider,
        fallback_policy=GlassFallbackPolicy.STRICT,
    )
    with pytest.raises(ValueError):
        interface.get_resistivity(1000.0)
