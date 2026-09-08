"""Public glass conductivity contracts: units, provider protocol, policies.

Single owned conversion/validation boundary for glass conductivity work
(issue #5062). Providers receive absolute temperature in kelvin and return
conductivity in SI units (S/m); invalid values are rejected before caching.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Unit-conversion and domain constants (single owned conversion boundary)
S_PER_CM_TO_SI = 100.0  # 1 S/cm = 100 S/m
_CELSIUS_TO_KELVIN_OFFSET = 273.15  # K; absolute zero is -273.15 degC

# Provenance sources reported for each computed conductivity
SOURCE_PROVIDER = "provider"
SOURCE_DEFAULT_MODEL = "default_model"


def celsius_to_kelvin(temperature_celsius: float) -> float:
    """Convert Celsius to kelvin (strictly positive).

    Rejects non-finite input and temperatures at/below absolute zero.

    Args:
        temperature_celsius: Degrees Celsius; finite, above absolute zero.

    Returns:
        Temperature in kelvin.

    Raises:
        TypeError: If the temperature is not a real number.
        ValueError: If non-finite or at/below absolute zero.
    """
    try:
        temperature = float(temperature_celsius)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"temperature must be a real number in Celsius, got {temperature_celsius!r}"
        ) from exc
    if not math.isfinite(temperature):
        raise ValueError(f"temperature must be finite, got {temperature_celsius!r}")
    if temperature <= -_CELSIUS_TO_KELVIN_OFFSET:
        raise ValueError(
            "temperature must be strictly above absolute zero "
            f"(-{_CELSIUS_TO_KELVIN_OFFSET} degC), got {temperature} degC"
        )
    return temperature + _CELSIUS_TO_KELVIN_OFFSET


def s_per_cm_to_s_per_m(conductivity_s_per_cm: float) -> float:
    """Convert conductivity from S/cm to SI S/m (1 S/cm = 100 S/m).

    Args:
        conductivity_s_per_cm: Conductivity in S per centimeter.

    Returns:
        Conductivity in S per meter.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If the value is non-finite.
    """
    try:
        value = float(conductivity_s_per_cm)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"conductivity must be a real number in S/cm, got {conductivity_s_per_cm!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"conductivity must be finite, got {conductivity_s_per_cm!r}")
    return value * S_PER_CM_TO_SI


@runtime_checkable
class ConductivityProvider(Protocol):
    """Public protocol for external glass conductivity providers.

    Contract: ``temperature_kelvin`` is absolute temperature (> 0 K);
    ``composition`` maps component names to finite mole fractions or is
    ``None``; ``power_density`` is a finite heating term; the return value
    is conductivity in S/m, finite and strictly positive (invalid values
    are rejected before caching).
    """

    def provide_conductivity(
        self,
        temperature_kelvin: float,
        composition: Mapping[str, float] | None,
        power_density: float,
    ) -> float:
        """Return finite, strictly positive conductivity in S/m.

        Args:
            temperature_kelvin: Absolute temperature in kelvin (> 0 K).
            composition: Component mole fractions, or ``None``.
            power_density: Finite heating term.

        Returns:
            Conductivity in S/m.
        """
        ...


class GlassFallbackPolicy(enum.Enum):
    """Explicit consumer-selected fallback policy for provider failure.

    Attributes:
        STRICT: Scientific mode. Provider exceptions and invalid output are
            surfaced; no substitution ever occurs.
        DEMO: Default mode. Provider failure falls back to the built-in
            default model, recorded with provenance and fallback reason
            (never silent).
        LEGACY: Named compatibility policy reproducing the historical
            behavior: a provider exception logs a warning and substitutes
            the default model without a provenance report. Invalid output
            values are still rejected at the public boundary; the interface
            never returns non-finite material properties.
    """

    STRICT = "strict"
    DEMO = "demo"
    LEGACY = "legacy"


@dataclass(frozen=True)
class ProviderResultReport:
    """Provenance record for the most recent conductivity computation.

    Attributes:
        source: ``"provider"`` or ``"default_model"``.
        fallback_reason: Reason for substitution, or ``None`` when none occurred.
        policy: The fallback policy in force for that computation.
    """

    source: str
    fallback_reason: str | None
    policy: GlassFallbackPolicy
