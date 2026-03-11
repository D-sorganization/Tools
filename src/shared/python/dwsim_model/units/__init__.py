"""
Per-unit-operation convenience API for the DWSIM Gasification Model.

Provides thin, callable wrappers around the standalone flowsheet classes
so that external consumers can run individual process stages without
understanding the DWSIM internals.

Usage::

    from dwsim_model.units import run_gasifier
    results = run_gasifier(mode="conversion")

    from dwsim_model.units import run_pem
    results = run_pem(config_path="config/master_config.yaml")

    from dwsim_model.units import run_trc
    results = run_trc()

Design notes:
    - Lazy imports: DWSIM runtime is only loaded when a run_* function is
      called, not when this module is imported.  This means ``import
      dwsim_model.units`` succeeds even without DWSIM installed.
    - DbC: All inputs are validated before any DWSIM work begins.
    - DRY: Delegates to existing standalone classes and topology functions.
      No process logic is duplicated here.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Valid modes (shared validation — DRY)
# ─────────────────────────────────────────────────────────────────────────────

VALID_MODES = frozenset({"conversion", "equilibrium", "kinetic", "mixed"})

# Default reactor type per unit per mode — single source of truth
_DEFAULT_REACTOR_TYPES: dict[str, dict[str, str]] = {
    "gasifier": {
        "conversion": "RCT_Conversion",
        "equilibrium": "RCT_Equilibrium",
        "kinetic": "RCT_PFR",
        "mixed": "RCT_Conversion",
    },
    "pem": {
        "conversion": "RCT_Conversion",
        "equilibrium": "RCT_Equilibrium",
        "kinetic": "RCT_PFR",
        "mixed": "RCT_Equilibrium",
    },
    "trc": {
        "conversion": "RCT_Conversion",
        "equilibrium": "RCT_Equilibrium",
        "kinetic": "RCT_PFR",
        "mixed": "RCT_PFR",
    },
}


def _validate_mode(mode: str) -> None:
    """DbC: Validate mode before any work is done.

    Raises
    ------
    ValueError
        If mode is not in VALID_MODES.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}. Valid modes: {sorted(VALID_MODES)}")


def _extract_standalone_results(builder: Any) -> dict[str, Any]:
    """Extract results from a solved standalone flowsheet builder.

    DRY: Uses the shared ResultsExtractor and MetricsCalculator so that
    KPI definitions are not duplicated.

    Parameters
    ----------
    builder
        A solved FlowsheetBuilder instance.

    Returns
    -------
    dict
        Keys: ``streams``, ``energy_streams``, ``metrics``, ``warnings``.
    """
    extractor_module = importlib.import_module("dwsim_model.results.extractor")
    metrics_module = importlib.import_module("dwsim_model.results.metrics")

    extractor = extractor_module.ResultsExtractor()
    results = extractor.extract(builder)

    calculator = metrics_module.MetricsCalculator()
    metrics = calculator.calculate(results)

    return {
        "results": results,
        "metrics": metrics,
        "warnings": getattr(metrics, "warnings", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_gasifier(
    mode: str = "conversion",
    config_path: str | Path | None = None,
    compound_set: list[str] | None = None,
) -> dict[str, Any]:
    """Run an isolated gasifier stage and return results.

    Parameters
    ----------
    mode
        Reactor fidelity mode.  One of: conversion, equilibrium, kinetic, mixed.
    config_path
        Path to a YAML config file for feed conditions.  Optional.
    compound_set
        List of DWSIM compound names.  Defaults to COMPOUNDS_STANDARD.

    Returns
    -------
    dict
        Simulation results including stream data and KPIs.

    Raises
    ------
    ValueError
        If mode is invalid.
    RuntimeError
        If DWSIM runtime is not available or the solve fails.
    """
    _validate_mode(mode)

    gasifier_module = importlib.import_module("dwsim_model.standalone.gasifier_model")

    logger.info("Running isolated gasifier stage (mode=%s)", mode)

    flowsheet = gasifier_module.GasifierStandaloneFlowsheet(compound_set=compound_set)
    flowsheet.setup_thermo()
    flowsheet.build_flowsheet()
    flowsheet.calculate()

    return _extract_standalone_results(flowsheet.builder)


def run_pem(
    mode: str = "equilibrium",
    config_path: str | Path | None = None,
    compound_set: list[str] | None = None,
) -> dict[str, Any]:
    """Run an isolated PEM (Plasma Entrained Melter) stage and return results.

    Parameters
    ----------
    mode
        Reactor fidelity mode.  One of: conversion, equilibrium, kinetic, mixed.
    config_path
        Path to a YAML config file for feed conditions.  Optional.
    compound_set
        List of DWSIM compound names.  Defaults to COMPOUNDS_STANDARD.

    Returns
    -------
    dict
        Simulation results including stream data and KPIs.

    Raises
    ------
    ValueError
        If mode is invalid.
    RuntimeError
        If DWSIM runtime is not available or the solve fails.
    """
    _validate_mode(mode)

    pem_module = importlib.import_module("dwsim_model.standalone.pem_model")

    logger.info("Running isolated PEM stage (mode=%s)", mode)

    flowsheet = pem_module.PEMStandaloneFlowsheet(compound_set=compound_set)
    flowsheet.setup_thermo()
    flowsheet.build_flowsheet()
    flowsheet.calculate()

    return _extract_standalone_results(flowsheet.builder)


def run_trc(
    mode: str = "kinetic",
    config_path: str | Path | None = None,
    compound_set: list[str] | None = None,
) -> dict[str, Any]:
    """Run an isolated TRC (Thermal Reduction Chamber) stage and return results.

    Parameters
    ----------
    mode
        Reactor fidelity mode.  One of: conversion, equilibrium, kinetic, mixed.
    config_path
        Path to a YAML config file for feed conditions.  Optional.
    compound_set
        List of DWSIM compound names.  Defaults to COMPOUNDS_STANDARD.

    Returns
    -------
    dict
        Simulation results including stream data and KPIs.

    Raises
    ------
    ValueError
        If mode is invalid.
    RuntimeError
        If DWSIM runtime is not available or the solve fails.
    """
    _validate_mode(mode)

    trc_module = importlib.import_module("dwsim_model.standalone.trc_model")

    logger.info("Running isolated TRC stage (mode=%s)", mode)

    flowsheet = trc_module.TRCStandaloneFlowsheet(compound_set=compound_set)
    flowsheet.setup_thermo()
    flowsheet.build_flowsheet()
    flowsheet.calculate()

    return _extract_standalone_results(flowsheet.builder)


def run_full_train(
    mode: str = "mixed",
    config_path: str | Path | None = None,
    compound_set: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full 7-unit gasification train and return results.

    Parameters
    ----------
    mode
        Reactor fidelity mode.  One of: conversion, equilibrium, kinetic, mixed.
    config_path
        Path to a master_config.yaml.  Optional.
    compound_set
        List of DWSIM compound names.  Defaults to COMPOUNDS_STANDARD.

    Returns
    -------
    dict
        Simulation results including stream data and KPIs.

    Raises
    ------
    ValueError
        If mode is invalid.
    RuntimeError
        If DWSIM runtime is not available or the solve fails.
    """
    _validate_mode(mode)

    gasification_module = importlib.import_module("dwsim_model.gasification")

    logger.info("Running full gasification train (mode=%s)", mode)

    flowsheet = gasification_module.GasificationFlowsheet(
        mode=mode,
        config_path=str(config_path) if config_path else None,
        compound_set=list(compound_set) if compound_set else None,
    )
    flowsheet.build_flowsheet()
    flowsheet.run()

    return _extract_standalone_results(flowsheet.builder)
