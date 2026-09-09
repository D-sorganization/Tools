"""Glass Properties Interface Module

Interface for external glass property calculators with default models;
extracted from electrode_advisor.py. Issue #5062 contracts (provider
protocol, fallback policies, validation) live in ``glass_contracts``.

Author: Chemical Equilibrium Calculator Team
Date: July 8, 2025
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

from shared.python.sidekick.calculators.electrical.glass_contracts import (
    SOURCE_DEFAULT_MODEL,
    SOURCE_PROVIDER,
    ConductivityProvider,
    GlassFallbackPolicy,
    ProviderResultReport,
    celsius_to_kelvin,
)
from shared.python.sidekick.utils.unit_constants import R_UNIVERSAL

__all__ = ["GlassPropertiesInterface"]

_logger = logging.getLogger(__name__)

# Performance constants
_CACHE_MAX_SIZE = 1000  # Maximum cache entries before LRU eviction
_R_GAS = R_UNIVERSAL  # Gas constant J/(mol·K) - pre-computed to avoid repeated lookups
_POWER_DENSITY_HEATING_COEFFICIENT = 0.0001  # K per unit power density


class GlassPropertiesInterface:
    """Interface for external glass property calculators.

    DbC: positive-integer ``cache_max_size``; Celsius temperatures validated
    once via :func:`celsius_to_kelvin`; only validated finite-positive
    results are cached (failed responses never); non-finite values are
    never returned; rejected inputs fail before any side effect.
    """

    def __init__(
        self,
        external_calculator: Callable | None = None,
        cache_max_size: int = _CACHE_MAX_SIZE,
        fallback_policy: GlassFallbackPolicy = GlassFallbackPolicy.DEMO,
        conductivity_provider: ConductivityProvider | None = None,
    ) -> None:
        """Initialize the interface (see class docstring for DbC contracts).

        Args:
            external_calculator: Legacy callable ``(temperature_celsius,
                composition, power_density) -> S/m``.
            cache_max_size: Positive-integer cache capacity (default 1000).
            fallback_policy: Explicit policy (default ``DEMO``).
            conductivity_provider: :class:`ConductivityProvider` (kelvin in,
                S/m out); mutually exclusive with ``external_calculator``.

        Raises:
            TypeError: Non-int capacity or non-protocol provider.
            ValueError: Non-positive capacity or both providers supplied.
        """
        self._validate_cache_capacity(cache_max_size)
        if conductivity_provider is not None and external_calculator is not None:
            raise ValueError(
                "conductivity_provider and external_calculator are mutually "
                "exclusive; supply exactly one"
            )
        if conductivity_provider is not None and not isinstance(
            conductivity_provider, ConductivityProvider
        ):
            raise TypeError(
                "conductivity_provider must implement the ConductivityProvider "
                "protocol (provide_conductivity method)"
            )
        self.external_calculator = external_calculator
        self.conductivity_provider = conductivity_provider
        self.fallback_policy = fallback_policy
        self._cache_max_size = cache_max_size
        self._default_properties = {
            "base_conductivity": 1.0,  # S/m at reference
            "activation_energy": 80000,  # J/mol
            "reference_temp": 1473.15,  # K (1200 C)
            "metal_conductivity": 10000.0,  # S/m - Very high for molten metal
        }
        # Use OrderedDict for LRU cache behavior
        self._temperature_dependent_data: OrderedDict[
            tuple[float, Any, float], float
        ] = OrderedDict()
        self._current_properties: dict[str, Any] = {}  # Store current glass properties
        self._last_report: ProviderResultReport | None = None
        # Pre-compute reference term for Arrhenius equation
        self._arrhenius_ref_term = (
            self._default_properties["activation_energy"]
            / _R_GAS
            / self._default_properties["reference_temp"]
        )

    @staticmethod
    def _validate_cache_capacity(cache_max_size: int) -> None:
        """Enforce the documented positive-integer cache capacity contract."""
        if isinstance(cache_max_size, bool) or not isinstance(cache_max_size, int):
            raise TypeError(
                f"cache_max_size must be an int, got {type(cache_max_size).__name__}"
            )
        if cache_max_size < 1:
            raise ValueError(
                "cache_max_size must be a positive integer (>= 1), "
                f"got {cache_max_size!r}"
            )

    def get_conductivity(
        self,
        temperature_celsius: float,
        composition: dict[str, float] | None = None,
        power_density: float = 0,
        is_metal: bool = False,
    ) -> float:
        """Get validated conductivity in S/m from provider or default model.

        Args:
            temperature_celsius: Degrees Celsius (above absolute zero).
            composition: Optional mole fractions with finite values.
            power_density: Finite heating term.
            is_metal: When True, return the constant metal conductivity.

        Returns:
            Finite, strictly positive conductivity in S/m.

        Raises:
            TypeError: Non-real inputs or non-string composition keys.
            ValueError: Non-physical temperature, non-finite composition or
                power density, or rejected provider output (STRICT/LEGACY).
        """
        temperature_kelvin = celsius_to_kelvin(temperature_celsius)
        self._validate_composition(composition)
        self._validate_finite_scalar(power_density, "power_density")
        if is_metal:
            # Metal has very high conductivity, relatively constant with temperature
            return self._default_properties["metal_conductivity"]

        # Build cache key - use frozenset for O(1) hashing instead of sorted tuple
        comp_key = frozenset(composition.items()) if composition else None
        cache_key = (float(temperature_celsius), comp_key, float(power_density))

        # Check cache with LRU promotion (move to end on access)
        if cache_key in self._temperature_dependent_data:
            # Move to end for LRU behavior
            self._temperature_dependent_data.move_to_end(cache_key)
            return self._temperature_dependent_data[cache_key]

        conductivity, report, cacheable = self._compute_conductivity(
            temperature_kelvin,
            temperature_celsius,
            composition,
            power_density,
        )
        if cacheable:
            self._cache_result(cache_key, conductivity)
        self._last_report = report
        return conductivity

    def _compute_conductivity(
        self,
        temperature_kelvin: float,
        temperature_celsius: float,
        composition: dict[str, float] | None,
        power_density: float,
    ) -> tuple[float, ProviderResultReport, bool]:
        """Compute a validated conductivity, its report, and cacheability."""
        if self.conductivity_provider is not None:
            return self._from_callable(
                lambda: self.conductivity_provider.provide_conductivity(  # type: ignore[union-attr]
                    temperature_kelvin, composition, power_density
                ),
                temperature_celsius,
                power_density,
            )
        if self.external_calculator is not None:
            return self._from_callable(
                lambda: self.external_calculator(  # type: ignore[misc]
                    temperature_celsius, composition, power_density
                ),
                temperature_celsius,
                power_density,
            )
        default = self._default_conductivity_model(temperature_celsius, power_density)
        report = ProviderResultReport(
            source=SOURCE_DEFAULT_MODEL,
            fallback_reason=None,
            policy=self.fallback_policy,
        )
        return default, report, True

    def _from_callable(
        self,
        call: Callable[[], float],
        temperature_celsius: float,
        power_density: float,
    ) -> tuple[float, ProviderResultReport, bool]:
        """Adjudicate provider output by policy (raises per policy contract)."""
        try:
            raw_value = float(call())
        except (ValueError, TypeError, ArithmeticError) as exc:
            if self.fallback_policy is GlassFallbackPolicy.STRICT:
                raise
            return self._on_provider_failure(
                f"provider raised {type(exc).__name__}: {exc}",
                temperature_celsius,
                power_density,
            )
        if not (math.isfinite(raw_value) and raw_value > 0.0):
            reason = (
                f"provider returned invalid conductivity {raw_value!r} "
                "(must be finite and positive in S/m)"
            )
            if self.fallback_policy is not GlassFallbackPolicy.DEMO:
                # STRICT surfaces the rejection; LEGACY retains boundary
                # validation (never returns non-finite values).
                raise ValueError(reason)
            _logger.warning("%s; using default model", reason)
            value, report = self._default_model_result(
                reason, temperature_celsius, power_density
            )
            return value, report, False
        report = ProviderResultReport(
            source=SOURCE_PROVIDER, fallback_reason=None, policy=self.fallback_policy
        )
        return raw_value, report, True

    def _on_provider_failure(
        self,
        reason: str,
        temperature_celsius: float,
        power_density: float,
    ) -> tuple[float, ProviderResultReport, bool]:
        """Substitute default model after a provider exception (DEMO/LEGACY)."""
        if self.fallback_policy is GlassFallbackPolicy.LEGACY:
            _logger.warning(
                "External calculator failed: %s. Using default model.", reason
            )
        else:
            _logger.debug("Provider failed (%s); substituting default model", reason)
        value, report = self._default_model_result(
            None if self.fallback_policy is GlassFallbackPolicy.LEGACY else reason,
            temperature_celsius,
            power_density,
        )
        return value, report, False

    def _default_model_result(
        self,
        reason: str | None,
        temperature_celsius: float,
        power_density: float,
    ) -> tuple[float, ProviderResultReport]:
        """Substitute the default model result with explicit provenance."""
        default = self._default_conductivity_model(temperature_celsius, power_density)
        report = ProviderResultReport(
            source=SOURCE_DEFAULT_MODEL,
            fallback_reason=reason,
            policy=self.fallback_policy,
        )
        return default, report

    def _cache_result(
        self, cache_key: tuple[float, Any, float], conductivity: float
    ) -> None:
        """Cache a validated result with LRU eviction."""
        self._temperature_dependent_data[cache_key] = conductivity
        # Evict oldest entries if cache exceeds max size
        while len(self._temperature_dependent_data) > self._cache_max_size:
            self._temperature_dependent_data.popitem(last=False)

    @staticmethod
    def _validate_finite_scalar(value: float, name: str) -> None:
        """Validate that a scalar input is a finite real number."""
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")

    @staticmethod
    def _validate_composition(composition: dict[str, float] | None) -> None:
        """Validate composition keys (strings) and values (finite) pre-call."""
        if composition is None:
            return
        for name, fraction in composition.items():
            if not isinstance(name, str):
                raise TypeError(f"composition keys must be strings, got {name!r}")
            if not math.isfinite(float(fraction)):
                raise ValueError(
                    f"composition value for {name!r} must be finite, got {fraction!r}"
                )

    def get_resistivity(
        self,
        temperature_celsius: float,
        composition: dict[str, float] | None = None,
        power_density: float = 0,
        is_metal: bool = False,
    ) -> float:
        """Get resistivity as the validated reciprocal of conductivity (1/sigma).

        Args:
            temperature_celsius: Degrees Celsius (above absolute zero).
            composition: Optional mole fractions with finite values.
            power_density: Finite heating term.
            is_metal: When True, use the constant metal conductivity.

        Returns:
            Finite, strictly positive resistivity in ohm*m (sigma*rho = 1 SI).

        Raises:
            ValueError: If the conductivity is rejected; never infinity.
        """
        conductivity = self.get_conductivity(
            temperature_celsius,
            composition,
            power_density,
            is_metal,
        )
        if not (math.isfinite(conductivity) and conductivity > 0.0):
            raise ValueError(
                f"cannot form resistivity from conductivity {conductivity!r} S/m"
            )
        return 1.0 / conductivity

    def set_external_calculator(self, calculator: Callable) -> None:
        """Set the legacy calculator ``(celsius, composition, power) -> S/m``.

        Invalidates the conductivity cache.

        Raises:
            ValueError: If the calculator is ``None``.
        """
        if calculator is None:
            raise ValueError("calculator must be provided")
        self.external_calculator = calculator
        # Clear cache when calculator changes
        self._temperature_dependent_data.clear()

    def set_conductivity_provider(self, provider: ConductivityProvider) -> None:
        """Set the protocol provider (kelvin in, S/m out); invalidates cache.

        Raises:
            TypeError: If the provider does not implement the protocol.
        """
        if not isinstance(provider, ConductivityProvider):
            raise TypeError(
                "provider must implement the ConductivityProvider protocol "
                "(provide_conductivity method)"
            )
        self.conductivity_provider = provider
        self._temperature_dependent_data.clear()

    def get_last_provider_report(self) -> ProviderResultReport | None:
        """Return provenance for the most recent computation (``None`` before any)."""
        return self._last_report

    def update_properties(self, properties: dict) -> None:
        """Update current glass properties.

        Args:
            properties: Mapping merged into current properties.
        """
        self._current_properties.update(properties)

    def get_current_properties(self) -> dict:
        """Get a copy of the current glass properties mapping."""
        return self._current_properties.copy()

    def _default_conductivity_model(
        self,
        temperature_celsius: float,
        power_density: float = 0,
    ) -> float:
        """Default Arrhenius conductivity model (returns S/m).

        Args:
            temperature_celsius: Degrees Celsius.
            power_density: Heating term raising the effective temperature.

        Returns:
            Conductivity in S/m from the Arrhenius model.
        """
        # Apply power density heating effect upfront
        temp_kelvin = celsius_to_kelvin(temperature_celsius)
        if power_density > 0:
            # Local heating from power density
            temp_kelvin += power_density * _POWER_DENSITY_HEATING_COEFFICIENT

        # Arrhenius equation: sigma = sigma_0 * exp(-Ea/R * (1/T - 1/T_ref))
        # Pre-computed: _arrhenius_ref_term = -Ea/(R * T_ref)
        # So: exponent = -Ea/(R*T) - _arrhenius_ref_term = -Ea/R * (1/T - 1/T_ref)
        props = self._default_properties
        ea_over_r = props["activation_energy"] / _R_GAS
        exponent = -ea_over_r / temp_kelvin + self._arrhenius_ref_term

        return float(props["base_conductivity"] * np.exp(exponent))

    def clear_cache(self) -> None:
        """Clear temperature-dependent data cache."""
        self._temperature_dependent_data.clear()
