"""
analysis/sweep.py
=================
Parameter sweep engine for the DWSIM gasification model.

Allows systematic exploration of how a KPI (e.g. Cold Gas Efficiency or
H2/CO ratio) responds to changes in one or two input parameters without
manually editing YAML files.

Design notes for newer Python users
-------------------------------------
A "parameter sweep" means: run the simulation many times, each time with
a slightly different input value, and collect the results.  This is how
engineers find the best operating conditions without an expensive
optimiser.

The sweep engine works by:
1. Loading the base configuration from YAML files.
2. Deep-copying that config and patching a single key at a specific
   nested path (e.g. feeds → Gasifier_Biomass_Feed → mass_flow_kg_s).
3. Running the simulation for each patched config.
4. Collecting all results into a pandas DataFrame for easy analysis.

Because we can't call DWSIM in test environments, the engine also accepts
a *model_factory* callable that you can replace with a mock for unit tests.

Usage (1-D sweep)
-----------------
    from dwsim_model.analysis.sweep import ParameterSweep
    import numpy as np

    ps = ParameterSweep(base_config_path="config/master_config.yaml")
    df = ps.sweep_1d(
        param_path="feeds.Gasifier_Biomass_Feed.mass_flow_kg_s",
        values=np.linspace(2.0, 6.0, 9),
        kpis=["cold_gas_efficiency", "h2_co_ratio"],
    )
    df.to_csv("results/sweep_biomass_flow.csv", index=False)

Usage (2-D sweep)
-----------------
    df2d = ps.sweep_2d(
        param_a_path="feeds.Gasifier_Steam_Feed.mass_flow_kg_s",
        values_a=[0.5, 1.0, 1.5, 2.0],
        param_b_path="feeds.Gasifier_Oxygen_Feed.mass_flow_kg_s",
        values_b=[2.0, 3.0, 4.0],
        kpis=["cold_gas_efficiency", "h2_co_ratio", "tar_loading_mg_Nm3"],
    )
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# pandas is optional; the sweep will still work and return a list of dicts
# if pandas is not installed.
try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    logger.warning(
        "pandas not installed — sweep results returned as list of dicts. "
        "Install with: pip install pandas"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: nested dict access
# ─────────────────────────────────────────────────────────────────────────────


def _set_nested(d: dict, dot_path: str, value: Any) -> dict:
    """
    Set a value in a nested dict using a dot-separated path.

    Example
    -------
    >>> cfg = {"feeds": {"Steam": {"mass_flow_kg_s": 1.0}}}
    >>> _set_nested(cfg, "feeds.Steam.mass_flow_kg_s", 2.5)
    {'feeds': {'Steam': {'mass_flow_kg_s': 2.5}}}
    """
    keys = dot_path.split(".")
    node = d
    for key in keys[:-1]:
        if key not in node:
            raise KeyError(
                f"Path '{dot_path}' not found in config — missing key '{key}'."
            )
        node = node[key]
    node[keys[-1]] = value
    return d


def _get_nested(d: dict, dot_path: str, default=None) -> Any:
    """Get a value from a nested dict using a dot-separated path."""
    keys = dot_path.split(".")
    node = d
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ─────────────────────────────────────────────────────────────────────────────
# Default model runner
# ─────────────────────────────────────────────────────────────────────────────


def _default_model_runner(config: dict) -> dict:
    """
    Run the gasification model with the given config dict and return KPIs.

    This default implementation imports and runs the full DWSIM model.
    In test environments (no DWSIM), replace this with a mock via the
    *model_factory* argument to ParameterSweep.

    Returns
    -------
    dict: {kpi_name: value, ...}  — all available KPIs from GasificationMetrics
    """
    from dwsim_model.gasification import GasificationFlowsheet
    from dwsim_model.results.extractor import ResultsExtractor
    from dwsim_model.results.metrics import MetricsCalculator

    flowsheet = GasificationFlowsheet(runtime_config=config)
    flowsheet.build_flowsheet()  # fix: was build() - method name mismatch
    flowsheet.run()  # fix: was solve() - method name mismatch

    extractor = ResultsExtractor(compound_names=list(flowsheet.compound_set))
    results = extractor.extract(flowsheet.builder)

    calculator = MetricsCalculator()
    metrics = calculator.calculate(results)

    return metrics.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep class
# ─────────────────────────────────────────────────────────────────────────────


class ParameterSweep:
    """
    Runs a batch of simulations over a grid of parameter values.

    Parameters
    ----------
    base_config_path:
        Path to the master YAML config file.  All sweep runs start from
        this config and patch only the specified parameter(s).
    model_runner:
        Optional callable with signature ``(config: dict) -> dict``.
        Defaults to the DWSIM model runner above.  Provide a mock here
        for testing without DWSIM.
    """

    def __init__(
        self,
        base_config_path: str | Path | None = None,
        model_runner: Optional[Callable[[dict], dict]] = None,
    ):
        self.base_config_path = Path(base_config_path) if base_config_path else None
        self._runner = model_runner or _default_model_runner
        self._base_config: dict = {}

        if self.base_config_path:
            self._base_config = self._load_config(self.base_config_path)

    # ── Config loading ────────────────────────────────────────────────────

    def _load_config(self, path: Path) -> dict:
        """Load a YAML config file into a plain dict."""
        import yaml

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def set_base_config(self, config: dict) -> None:
        """Directly set the base config dict (useful for testing)."""
        self._base_config = config

    # ── 1-D sweep ─────────────────────────────────────────────────────────

    def sweep_1d(
        self,
        param_path: str,
        values: Sequence[float],
        kpis: Optional[list[str]] = None,
        label: Optional[str] = None,
    ):
        """
        Sweep one parameter over a sequence of values.

        Parameters
        ----------
        param_path:
            Dot-separated path into the config dict.
            e.g. ``"feeds.Gasifier_Biomass_Feed.mass_flow_kg_s"``
        values:
            Sequence of values to try.
        kpis:
            List of KPI keys to extract from results.  If None, all KPIs
            are returned.
        label:
            Human-readable name for the parameter (used in output column).

        Returns
        -------
        pandas.DataFrame if pandas is installed, else list of dicts.
        Each row corresponds to one simulation run.
        """
        label = label or param_path.rsplit(".", maxsplit=1)[-1]
        rows = []

        logger.info(f"Starting 1-D sweep: '{param_path}' over {len(values)} values")

        for i, val in enumerate(values):
            config = copy.deepcopy(self._base_config)
            try:
                _set_nested(config, param_path, float(val))
            except KeyError as exc:
                logger.warning(f"[sweep_1d] Could not set {param_path}={val}: {exc}")
                continue

            row: dict[str, Any] = {label: float(val)}
            t0 = time.perf_counter()

            try:
                kpi_dict = self._runner(config)
                elapsed = time.perf_counter() - t0
                logger.debug(
                    f"  Run {i + 1}/{len(values)}: {label}={val} → "
                    f"CGE={kpi_dict.get('cold_gas_efficiency', '?'):.3f}  "
                    f"({elapsed:.1f}s)"
                )
                if kpis:
                    kpi_dict = {k: kpi_dict.get(k) for k in kpis}
                row.update(kpi_dict)
                row["run_time_s"] = round(elapsed, 2)
                row["converged"] = kpi_dict.get("converged")  # type: ignore[assignment]
            except Exception as exc:
                logger.error(f"  Run {i + 1}: {label}={val} FAILED: {exc}")
                row["error"] = str(exc)  # type: ignore[assignment]

            rows.append(row)

        logger.info(f"1-D sweep complete — {len(rows)} successful runs.")
        return _to_dataframe(rows)

    # ── 2-D sweep ─────────────────────────────────────────────────────────

    def sweep_2d(
        self,
        param_a_path: str,
        values_a: Sequence[float],
        param_b_path: str,
        values_b: Sequence[float],
        kpis: Optional[list[str]] = None,
        label_a: Optional[str] = None,
        label_b: Optional[str] = None,
    ):
        """
        Sweep two parameters over a 2-D grid (len(values_a) × len(values_b) runs).

        Parameters
        ----------
        param_a_path, param_b_path:
            Dot-separated config paths for the two parameters.
        values_a, values_b:
            Sequences of values to try for each parameter.
        kpis:
            KPI keys to return.  None = all.
        label_a, label_b:
            Column names for the swept parameters.

        Returns
        -------
        pandas.DataFrame or list of dicts.
        """
        label_a = label_a or param_a_path.rsplit(".", maxsplit=1)[-1]
        label_b = label_b or param_b_path.rsplit(".", maxsplit=1)[-1]
        total_runs = len(values_a) * len(values_b)

        logger.info(
            f"Starting 2-D sweep: '{param_a_path}' × '{param_b_path}' "
            f"= {total_runs} runs"
        )

        rows = []
        run_idx = 0

        for val_a in values_a:
            for val_b in values_b:
                run_idx += 1
                config = copy.deepcopy(self._base_config)

                try:
                    _set_nested(config, param_a_path, float(val_a))
                    _set_nested(config, param_b_path, float(val_b))
                except KeyError as exc:
                    logger.warning(
                        f"[sweep_2d] Run {run_idx}: could not set params: {exc}"
                    )
                    continue

                row: dict[str, Any] = {label_a: float(val_a), label_b: float(val_b)}
                t0 = time.perf_counter()

                try:
                    kpi_dict = self._runner(config)
                    elapsed = time.perf_counter() - t0
                    logger.debug(
                        f"  Run {run_idx}/{total_runs}: "
                        f"{label_a}={val_a}, {label_b}={val_b} → "
                        f"CGE={kpi_dict.get('cold_gas_efficiency', '?'):.3f}  "
                        f"({elapsed:.1f}s)"
                    )
                    if kpis:
                        kpi_dict = {k: kpi_dict.get(k) for k in kpis}
                    row.update(kpi_dict)
                    row["run_time_s"] = round(elapsed, 2)
                except Exception as exc:
                    logger.error(f"  Run {run_idx}: ({val_a}, {val_b}) FAILED: {exc}")
                    row["error"] = str(exc)  # type: ignore[assignment]

                rows.append(row)

        logger.info(f"2-D sweep complete — {len(rows)} runs finished.")
        return _to_dataframe(rows)

    # ── Sensitivity (one-at-a-time) ────────────────────────────────────────

    def sensitivity_oat(
        self,
        params: dict[str, tuple[float, float]],
        kpis: Optional[list[str]] = None,
        n_steps: int = 5,
    ):
        """
        One-at-a-time (OAT) sensitivity analysis.

        For each parameter in *params*, vary it over [min_val, max_val] while
        holding all other parameters at their base values.

        Parameters
        ----------
        params:
            {param_path: (min_val, max_val)} — parameter ranges to explore.
        kpis:
            KPI keys to return.
        n_steps:
            Number of evenly spaced values between min and max (inclusive).

        Returns
        -------
        pandas.DataFrame or list of dicts.
        All rows include a 'swept_param' column identifying which parameter
        was varied.
        """
        try:
            import numpy as np
        except ImportError:
            # AUTO-FIXED: explicitly fail with clear missing dependency requirement instead of crashing silently
            raise ImportError(
                "numpy is required for sensitivity_oat. Install it with: pip install numpy"
            )

        all_rows = []

        for param_path, (min_val, max_val) in params.items():
            values = np.linspace(min_val, max_val, n_steps).tolist()
            label = param_path.split(".")[-1]
            df = self.sweep_1d(param_path, values, kpis=kpis, label=label)

            # Add identifier column
            if _HAS_PANDAS and isinstance(df, pd.DataFrame):
                df.insert(0, "swept_param", param_path)
                all_rows.append(df)
            else:
                for row in df:
                    row["swept_param"] = param_path
                    all_rows.append(row)

        if _HAS_PANDAS and all_rows and isinstance(all_rows[0], pd.DataFrame):
            return pd.concat(all_rows, ignore_index=True)
        return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _to_dataframe(rows: list[dict]):
    """Convert list of dicts to DataFrame if pandas available."""
    if _HAS_PANDAS:
        return pd.DataFrame(rows)
    return rows
