<<<<<<< HEAD
"""Result formatting and rendering helpers for the pressure drop calculator.

This module contains ``_format_results``, all ``_print_*`` helpers,
``print_results``, ``_generate_recommendations``, and ``_wrap_text``.
"""
=======
"""Result formatting and presentation helpers for pressure drop calculations."""
>>>>>>> origin/main

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


<<<<<<< HEAD
def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to specified width."""
    if text is None:
        raise ValueError("text must be provided")
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += (" " if current_line else "") + word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines if lines else [""]


def _format_results(results: Any) -> dict[str, Any]:
    """Format results into a comprehensive dictionary."""
    return {
        # Pressure drops
=======
PRESSURE_CONVERSIONS_TO_PA = {
    "Pa": 1.0,
    "kPa": 1000.0,
    "MPa": 1e6,
    "bar": 1e5,
    "mbar": 100.0,
    "atm": 101325.0,
    "psi": 6894.76,
    "psia": 6894.76,
    "psig": 6894.76,
}


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between units."""
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    if from_unit == "K":
        temp_k = value
    elif from_unit == "C":
        temp_k = value + 273.15
    elif from_unit == "F":
        temp_k = (value - 32) * 5 / 9 + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
    if to_unit == "K":
        return temp_k
    if to_unit == "C":
        return temp_k - 273.15
    if to_unit == "F":
        return (temp_k - 273.15) * 9 / 5 + 32
    raise ValueError(f"Unknown temperature unit: {to_unit}")


def convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """Convert pressure between units."""
    if from_unit not in PRESSURE_CONVERSIONS_TO_PA:
        raise ValueError(f"Unknown pressure unit: {from_unit}")
    if to_unit not in PRESSURE_CONVERSIONS_TO_PA:
        raise ValueError(f"Unknown pressure unit: {to_unit}")
    pressure_pa = value * PRESSURE_CONVERSIONS_TO_PA[from_unit]
    return pressure_pa / PRESSURE_CONVERSIONS_TO_PA[to_unit]


def format_results(results: Any) -> dict[str, Any]:
    """Format engine results into the public dictionary contract."""
    return {
>>>>>>> origin/main
        "pressure_drop_pa": results.total_pressure_drop,
        "pressure_drop_bar": results.total_pressure_drop / 1e5,
        "pressure_drop_psi": results.total_pressure_drop / 6894.76,
        "pressure_drop_kpa": results.total_pressure_drop / 1000.0,
<<<<<<< HEAD
        # Pressure drop components
=======
>>>>>>> origin/main
        "friction_loss_pa": results.friction_pressure_drop,
        "friction_loss_bar": results.friction_pressure_drop / 1e5,
        "fitting_loss_pa": results.fitting_pressure_drop,
        "fitting_loss_bar": results.fitting_pressure_drop / 1e5,
        "elevation_loss_pa": results.elevation_pressure_drop,
<<<<<<< HEAD
        # Outlet pressure
        "outlet_pressure_pa": results.outlet_pressure,
        "outlet_pressure_bar": results.outlet_pressure / 1e5,
        "outlet_pressure_psi": results.outlet_pressure / 6894.76,
        # Flow characteristics
=======
        "outlet_pressure_pa": results.outlet_pressure,
        "outlet_pressure_bar": results.outlet_pressure / 1e5,
        "outlet_pressure_psi": results.outlet_pressure / 6894.76,
>>>>>>> origin/main
        "friction_factor": results.friction_factor,
        "reynolds_number": results.flow_properties.reynolds_number,
        "flow_velocity_m_s": results.flow_properties.velocity,
        "flow_velocity_ft_s": results.flow_properties.velocity * 3.28084,
        "mach_number": results.flow_properties.mach_number,
        "flow_regime": results.flow_regime,
<<<<<<< HEAD
        # Gas properties
=======
>>>>>>> origin/main
        "density_kg_m3": results.flow_properties.density,
        "viscosity_pa_s": results.flow_properties.viscosity,
        "compressibility_factor": results.flow_properties.compressibility_factor,
        "molecular_weight": results.flow_properties.molecular_weight,
<<<<<<< HEAD
        # Performance metrics
        "erosional_velocity_m_s": results.erosional_velocity,
        "erosion_ratio": results.erosion_ratio,
        "erosion_ratio_percent": results.erosion_ratio * 100,
        # Additional
        "pressure_drop_per_100ft_pa": results.pressure_drop_per_100ft,
        "velocity_pressure_pa": results.velocity_pressure,
        # Warnings
=======
        "erosional_velocity_m_s": results.erosional_velocity,
        "erosion_ratio": results.erosion_ratio,
        "erosion_ratio_percent": results.erosion_ratio * 100,
        "pressure_drop_per_100ft_pa": results.pressure_drop_per_100ft,
        "velocity_pressure_pa": results.velocity_pressure,
>>>>>>> origin/main
        "warnings": results.warnings,
    }


<<<<<<< HEAD
def _print_summary_section(results: dict[str, Any]) -> None:
    """Log the pressure-drop summary section."""
    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " SUMMARY ".center(78) + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(
        f"│  Total Pressure Drop:  {results['pressure_drop_bar']:10.4f} bar  "
        f"│  {results['pressure_drop_psi']:10.4f} psi  │  {results['pressure_drop_kpa']:10.2f} kPa  │"
    )
    logger.info(
        f"│  Outlet Pressure:      {results['outlet_pressure_bar']:10.4f} bar  "
        f"│  {results['outlet_pressure_psi']:10.4f} psi  │                 │"
    )
    logger.info("└" + "─" * 78 + "┘")


def _print_breakdown_section(results: dict[str, Any]) -> None:
    """Log the pressure-drop breakdown by component."""

    def safe_percent(num: float, denom: float) -> float:
        return (num / denom * 100) if denom != 0 else 0.0

    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " PRESSURE DROP BREAKDOWN ".center(78) + "│")
    logger.info("├" + "─" * 38 + "┬" + "─" * 19 + "┬" + "─" * 19 + "┤")
    logger.info(
        "│  Component                           │     Value (bar)   │    Percentage   │"
    )
    logger.info("├" + "─" * 38 + "┼" + "─" * 19 + "┼" + "─" * 19 + "┤")

    dp_total = results["pressure_drop_pa"]
    friction_pct = safe_percent(results["friction_loss_pa"], dp_total)
    fitting_pct = safe_percent(results["fitting_loss_pa"], dp_total)
    elevation_pct = safe_percent(results["elevation_loss_pa"], dp_total)

    logger.info(
        f"│  Friction (pipe wall)                │ {results['friction_loss_bar']:17.6f} │ {friction_pct:15.1f}% │"
    )
    logger.info(
        f"│  Fittings & valves                   │ {results['fitting_loss_bar']:17.6f} │ {fitting_pct:15.1f}% │"
    )
    if abs(results["elevation_loss_pa"]) > 0.1:
        logger.info(
            f"│  Elevation change                    │ {results['elevation_loss_pa'] / 1e5:17.6f} │ {elevation_pct:15.1f}% │"
        )
    logger.info("└" + "─" * 38 + "┴" + "─" * 19 + "┴" + "─" * 19 + "┘")


def _print_flow_and_gas_sections(results: dict[str, Any]) -> None:
    """Log flow characteristics and gas property sections."""
    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " FLOW CHARACTERISTICS ".center(78) + "│")
    logger.info("├" + "─" * 38 + "┬" + "─" * 39 + "┤")
    logger.info(
        f"│  Flow Velocity:     {results['flow_velocity_m_s']:10.2f} m/s   │  {results['flow_velocity_ft_s']:10.2f} ft/s                  │"
    )
    logger.info(
        f"│  Reynolds Number:   {results['reynolds_number']:10.0f}        │  Flow Regime: {results['flow_regime']:18s}   │"
    )
    logger.info(
        f"│  Mach Number:       {results['mach_number']:10.4f}        │  Friction Factor: {results['friction_factor']:14.6f}   │"
    )
    logger.info("└" + "─" * 38 + "┴" + "─" * 39 + "┘")

    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " GAS PROPERTIES ".center(78) + "│")
    logger.info("├" + "─" * 38 + "┬" + "─" * 39 + "┤")
    logger.info(
        f"│  Density:           {results['density_kg_m3']:10.4f} kg/m³  │  Molecular Weight: {results['molecular_weight']:12.2f} kg/kmol│"
    )
    logger.info(
        f"│  Viscosity:         {results['viscosity_pa_s'] * 1e6:10.4f} µPa·s  │  Compressibility (Z): {results['compressibility_factor']:10.4f}     │"
    )
    logger.info("└" + "─" * 38 + "┴" + "─" * 39 + "┘")


def _print_safety_section(results: dict[str, Any]) -> None:
    """Log the safety metrics section."""
    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " SAFETY METRICS ".center(78) + "│")
    logger.info("├" + "─" * 38 + "┬" + "─" * 39 + "┤")

    erosion_ratio = results["erosion_ratio_percent"]
    if erosion_ratio < 50:
        erosion_status = "✅ SAFE"
    elif erosion_ratio < 80:
        erosion_status = "⚠️  CAUTION"
    else:
        erosion_status = "❌ DANGER"

    logger.info(
        f"│  Erosional Velocity: {results['erosional_velocity_m_s']:9.2f} m/s   │  Status: {erosion_status:26s}  │"
    )
    logger.info(
        f"│  Erosion Ratio:      {erosion_ratio:9.1f} %     │  (Velocity/Erosional limit)         │"
    )
    logger.info("└" + "─" * 38 + "┴" + "─" * 39 + "┘")


def _print_warnings_and_recommendations(
    results: dict[str, Any], show_recommendations: bool
) -> None:
    """Log warnings and engineering recommendations."""
    if results is None:
        raise ValueError("results must be provided")
    if results.get("warnings"):
        warnings = results["warnings"]
        if isinstance(warnings, list) and len(warnings) > 0:
            logger.info("\n┌" + "─" * 78 + "┐")
            logger.warning("│" + " ⚠️  WARNINGS ".center(78) + "│")
            logger.info("├" + "─" * 78 + "┤")
            for warning in warnings:
                wrapped = _wrap_text(warning, 74)
                for line in wrapped:
                    logger.info(f"│  {line:74s}  │")
            logger.info("└" + "─" * 78 + "┘")

    if show_recommendations:
        recommendations = _generate_recommendations(results)
        if recommendations:
            logger.info("\n┌" + "─" * 78 + "┐")
            logger.info("│" + " 💡 RECOMMENDATIONS ".center(78) + "│")
            logger.info("├" + "─" * 78 + "┤")
            for rec in recommendations:
                wrapped = _wrap_text(rec, 74)
                for line in wrapped:
                    logger.info(f"│  {line:74s}  │")
            logger.info("└" + "─" * 78 + "┘")


=======
>>>>>>> origin/main
def print_results(
    results: dict[str, Any],
    title: str = "PRESSURE DROP CALCULATION RESULTS",
    show_recommendations: bool = True,
) -> None:
<<<<<<< HEAD
    """Print results in a beautifully formatted table with recommendations.

    Args:
        results: Results dictionary from calculate_pressure_drop
        title: Title for the output
        show_recommendations: Whether to show engineering recommendations
    """
    if results is None:
        raise ValueError("results must be provided")
    logger.info("\n" + "═" * 80)
    logger.info(f"  {title}  ".center(80, "═"))
    logger.info("═" * 80)

    _print_summary_section(results)
    _print_breakdown_section(results)
    _print_flow_and_gas_sections(results)
    _print_safety_section(results)
    _print_warnings_and_recommendations(results, show_recommendations)

    logger.info("═" * 80 + "\n")


def _generate_recommendations(results: dict[str, Any]) -> list[str]:
    """Generate engineering recommendations based on calculation results."""
    recommendations = []

    # High pressure drop
=======
    """Print results in a concise formatted log table."""
    logger.info("%s", title)
    logger.info("Pressure drop: %.4f bar", results["pressure_drop_bar"])
    logger.info("Outlet pressure: %.4f bar", results["outlet_pressure_bar"])
    logger.info("Flow regime: %s", results["flow_regime"])
    logger.info("Velocity: %.2f m/s", results["flow_velocity_m_s"])
    if results.get("warnings"):
        for warning in results["warnings"]:
            logger.warning("%s", warning)
    if show_recommendations:
        for recommendation in generate_recommendations(results):
            logger.info("Recommendation: %s", recommendation)


def generate_recommendations(results: dict[str, Any]) -> list[str]:
    """Generate engineering recommendations based on calculation results."""
    recommendations: list[str] = []
>>>>>>> origin/main
    dp_ratio = results["pressure_drop_pa"] / (
        results["outlet_pressure_pa"] + results["pressure_drop_pa"]
    )
    if dp_ratio > 0.20:
        recommendations.append(
<<<<<<< HEAD
            f"High pressure drop ({dp_ratio * 100:.0f}% of inlet). Consider: larger pipe diameter, "
            "shorter pipe run, or fewer fittings."
        )

    # Erosion concerns
    erosion_ratio = results["erosion_ratio"]
    if erosion_ratio > 0.8:
        recommendations.append(
            "Velocity exceeds 80% of erosional limit. Consider larger pipe diameter to "
            "reduce velocity and extend pipe life."
        )
    elif erosion_ratio > 0.5:
        recommendations.append(
            "Velocity is 50-80% of erosional limit. Monitor pipe condition and consider "
            "velocity reduction for longer service life."
        )

    # Fitting losses
    if results["fitting_loss_pa"] > results["friction_loss_pa"]:
        recommendations.append(
            "Fitting losses exceed pipe friction. Consider using long-radius elbows, "
            "full-port valves, or reducing number of fittings."
        )

    # High Mach number
    if results["mach_number"] > 0.3:
        recommendations.append(
            f"High Mach number ({results['mach_number']:.3f}). Compressibility effects significant. "
            "Verify calculations and consider acoustic vibration analysis."
        )

    # Low Reynolds number
    if results["reynolds_number"] < 4000:
        recommendations.append(
            f"Low Reynolds number ({results['reynolds_number']:.0f}). Flow may be transitional "
            "or laminar - friction factor has higher uncertainty in this regime."
        )

    # Very high Reynolds number
    if results["reynolds_number"] > 1e7:
        recommendations.append(
            f"Very high Reynolds number ({results['reynolds_number']:.0e}). Ensure turbulent flow "
            "correlations are valid. Consider CFD analysis for critical applications."
        )

=======
            f"High pressure drop ({dp_ratio * 100:.0f}% of inlet). Consider: larger pipe diameter, shorter pipe run, or fewer fittings."
        )
    erosion_ratio = results["erosion_ratio"]
    if erosion_ratio > 0.8:
        recommendations.append(
            "Velocity exceeds 80% of erosional limit. Consider larger pipe diameter to reduce velocity and extend pipe life."
        )
    elif erosion_ratio > 0.5:
        recommendations.append(
            "Velocity is 50-80% of erosional limit. Monitor pipe condition and consider velocity reduction for longer service life."
        )
    if results["fitting_loss_pa"] > results["friction_loss_pa"]:
        recommendations.append(
            "Fitting losses exceed pipe friction. Consider using long-radius elbows, full-port valves, or reducing number of fittings."
        )
    if results["mach_number"] > 0.3:
        recommendations.append(
            f"High Mach number ({results['mach_number']:.3f}). Compressibility effects significant. Verify calculations and consider acoustic vibration analysis."
        )
    if results["reynolds_number"] < 4000:
        recommendations.append(
            f"Low Reynolds number ({results['reynolds_number']:.0f}). Flow may be transitional or laminar - friction factor has higher uncertainty in this regime."
        )
    if results["reynolds_number"] > 1e7:
        recommendations.append(
            f"Very high Reynolds number ({results['reynolds_number']:.0e}). Ensure turbulent flow correlations are valid. Consider CFD analysis for critical applications."
        )
>>>>>>> origin/main
    return recommendations
