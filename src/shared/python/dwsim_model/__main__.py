"""
__main__.py
===========
Command-line interface for the DWSIM Gasification Model.

Run with:
    python -m dwsim_model <subcommand> [options]

Subcommands
-----------
run       — Build, solve, and report a single simulation scenario.
sweep     — Run a 1-D or 2-D parameter sweep and save results to CSV.
validate  — Validate all YAML config files against Pydantic schemas.
export    — Export the current flowsheet to a DWSIM GUI (.dwxml) file.
summary   — Print a human-readable summary of the reaction configuration.

Examples
--------
    # Run the baseline scenario and save results to results/
    python -m dwsim_model run --scenario baseline

    # Run a custom config with verbose logging
    python -m dwsim_model run --config my_config.yaml --verbose

    # Sweep biomass flow rate from 2.0 to 6.0 kg/s in 9 steps
    python -m dwsim_model sweep \\
        --param feeds.Gasifier_Biomass_Feed.mass_flow_kg_s \\
        --min 2.0 --max 6.0 --steps 9 \\
        --kpis cold_gas_efficiency h2_co_ratio \\
        --output results/sweep_biomass.csv

    # Validate config files only (no simulation)
    python -m dwsim_model validate --config config/master_config.yaml

    # Print reaction summary
    python -m dwsim_model summary
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stdout)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: run
# ─────────────────────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> int:
    """Build, solve, and report a single scenario."""
    from dwsim_model.gasification import GasificationFlowsheet
    from dwsim_model.results.extractor import ResultsExtractor
    from dwsim_model.results.metrics import MetricsCalculator
    from dwsim_model.results.reporter import generate_html_report, generate_json_report

    logger = logging.getLogger("dwsim_model.cli.run")

    # Resolve config path
    config_path = Path(args.config) if args.config else None
    scenario = args.scenario or "baseline"

    logger.info(f"Starting scenario '{scenario}' ...")

    # Build flowsheet
    flowsheet = GasificationFlowsheet(config_path=config_path)
    flowsheet.build_flowsheet()  # fix: was build() - method name mismatch

    # Solve
    logger.info("Solving flowsheet ...")
    try:
        flowsheet.run()  # fix: was solve() - method name mismatch
        logger.info("Flowsheet solved successfully.")
    except Exception as exc:
        logger.error(f"Solve failed: {exc}")
        if not args.force:
            return 1
        logger.warning("--force flag set — continuing with partial results.")

    # Extract results
    extractor = ResultsExtractor()
    results = extractor.extract(flowsheet.builder)

    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate(results)

    # Print KPI summary to console
    _print_kpi_table(metrics)

    # Write reports
    out_dir = Path(args.output or "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"{scenario}_report.html"
    json_path = out_dir / f"{scenario}_report.json"

    generate_html_report(
        results,
        metrics,
        html_path,
        scenario_name=scenario,
        model_version="2.0",
    )
    generate_json_report(results, metrics, json_path, scenario_name=scenario)

    logger.info(f"Reports written to {out_dir}/")
    print(f"\n✓  HTML report: {html_path}")
    print(f"✓  JSON report: {json_path}")

    # Save .dwxml if requested
    if args.save_dwxml:
        try:
            dwxml_path = out_dir / f"{scenario}.dwxml"
            flowsheet.builder.save(
                str(dwxml_path)
            )  # fix: was sim.SaveToFile() - wrong DWSIM API
            logger.info(f"DWSIM file saved: {dwxml_path}")
            print(f"✓  DWXML file:   {dwxml_path}")
        except Exception as exc:
            logger.warning(f"Could not save DWXML: {exc}")

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: sweep
# ─────────────────────────────────────────────────────────────────────────────


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run a 1-D or 2-D parameter sweep."""
    import numpy as np

    from dwsim_model.analysis.sweep import ParameterSweep

    logger = logging.getLogger("dwsim_model.cli.sweep")

    config_path = Path(args.config) if args.config else None
    ps = ParameterSweep(base_config_path=config_path)

    kpis = args.kpis if args.kpis else None

    if args.param_b:
        # 2-D sweep
        logger.info(
            f"2-D sweep: '{args.param}' x '{args.param_b}' - "
            f"{args.steps} x {args.steps_b or args.steps} points"
        )
        values_a = np.linspace(args.min, args.max, args.steps)
        min_b = args.min_b if args.min_b is not None else args.min
        max_b = args.max_b if args.max_b is not None else args.max
        steps_b = args.steps_b or args.steps
        values_b = np.linspace(min_b, max_b, steps_b)
        df = ps.sweep_2d(args.param, values_a, args.param_b, values_b, kpis=kpis)
    else:
        # 1-D sweep
        values = np.linspace(args.min, args.max, args.steps)
        logger.info(
            f"1-D sweep: '{args.param}' from {args.min} to {args.max} "
            f"in {args.steps} steps"
        )
        df = ps.sweep_1d(args.param, values, kpis=kpis)

    # Save output
    out_path = Path(args.output) if args.output else Path("results/sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(out_path, index=False)
        print(f"\n✓  Sweep results saved to {out_path}")
    except AttributeError:
        # No pandas — df is a list of dicts
        import json

        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(df, indent=2, default=str))
        print(f"\n✓  Sweep results saved to {json_path} (pandas not installed)")

    # Print a quick table of results to console
    _print_sweep_summary(df)

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: validate
# ─────────────────────────────────────────────────────────────────────────────


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate YAML config files against Pydantic schemas."""
    import yaml

    from dwsim_model.config.schema import validate_master_config

    logger = logging.getLogger("dwsim_model.cli.validate")

    config_path = Path(args.config) if args.config else _find_default_config()
    if not config_path or not config_path.exists():
        print(f"✗  Config file not found: {config_path}", file=sys.stderr)
        return 1

    print(f"Validating: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = validate_master_config(raw)
        print(
            f"✓  master_config: VALID (reactor_mode={cfg.reactor_mode}, "
            f"compound_set={cfg.compound_set})"
        )
    except Exception as exc:
        print(f"✗  master_config: INVALID\n   {exc}", file=sys.stderr)
        logger.debug("Validation error detail:", exc_info=True)
        return 1

    # Validate feed sub-files
    feed_dir = config_path.parent / "feeds"
    ok = _validate_yaml_directory(feed_dir, "feed", logger)

    # Validate reactor sub-files
    reactor_dir = config_path.parent / "reactors"
    ok &= _validate_yaml_directory(reactor_dir, "reactor", logger)

    print(
        "\n"
        + (
            "✓  All configs valid."
            if ok
            else "✗  Some configs have issues — see above."
        )
    )
    return 0 if ok else 1


def _validate_yaml_directory(directory: Path, label: str, logger) -> bool:
    """Load and report on all YAML files in a directory."""
    import yaml

    if not directory.exists():
        return True
    ok = True
    for yaml_file in sorted(directory.glob("*.yaml")):
        try:
            with yaml_file.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if data is None:
                print(f"  ⚠  {yaml_file.name}: empty file")
            else:
                print(f"  ✓  {yaml_file.name}: {len(data)} top-level keys")
        except Exception as exc:
            print(f"  ✗  {yaml_file.name}: PARSE ERROR — {exc}", file=sys.stderr)
            ok = False
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: export
# ─────────────────────────────────────────────────────────────────────────────


def cmd_export(args: argparse.Namespace) -> int:
    """Build the flowsheet and export to a DWSIM GUI file."""
    from dwsim_model.gasification import GasificationFlowsheet

    logger = logging.getLogger("dwsim_model.cli.export")
    config_path = Path(args.config) if args.config else None

    logger.info("Building flowsheet for export ...")
    flowsheet = GasificationFlowsheet(config_path=config_path)
    flowsheet.build_flowsheet()  # fix: was build() - method name mismatch

    out_path = Path(args.output or "Gasification_Model_GUI.dwxml")

    try:
        flowsheet.builder.save(
            str(out_path)
        )  # fix: was sim.SaveToFile() - wrong DWSIM API
        print(f"✓  Flowsheet exported to {out_path}")
        print("   Open with DWSIM → File → Open Simulation")
        return 0
    except Exception as exc:
        logger.error(f"Export failed: {exc}")
        print(
            "✗  Could not save DWXML. This requires DWSIM to be installed.\n"
            f"   Error: {exc}",
            file=sys.stderr,
        )
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: summary
# ─────────────────────────────────────────────────────────────────────────────


def cmd_summary(args: argparse.Namespace) -> int:
    """Print reaction and stream configuration summary."""
    from dwsim_model.chemistry.reactions import print_reaction_summary

    print_reaction_summary()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Console output helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_kpi_table(metrics) -> None:
    """Print a formatted KPI summary table to stdout."""
    m = metrics

    def fmt(val, fmt_spec=".3f", suffix=""):
        return f"{val:{fmt_spec}}{suffix}" if val is not None else "—"

    print("\n" + "─" * 55)
    print(f"  {'KPI':<35} {'Value':>15}")
    print("─" * 55)

    rows = [
        ("Cold Gas Efficiency", fmt(getattr(m, "cold_gas_efficiency", None), ".1%")),
        (
            "Carbon Conversion",
            fmt(getattr(m, "carbon_conversion_efficiency", None), ".1%"),
        ),
        ("H₂/CO Ratio", fmt(getattr(m, "h2_co_ratio", None), ".2f")),
        ("Syngas LHV (MJ/Nm³)", fmt(getattr(m, "syngas_lhv_mj_nm3", None), ".2f")),
        (
            "Specific Energy (kWh/t)",
            fmt(getattr(m, "specific_energy_consumption_kWh_t", None), ".0f"),
        ),
        ("Tar Loading (mg/Nm³)", fmt(getattr(m, "tar_loading_mg_Nm3", None), ".1f")),
        ("Mass Balance Closure", fmt(getattr(m, "mass_balance_closure", None), ".4f")),
        (
            "Energy Balance Closure",
            fmt(getattr(m, "energy_balance_closure", None), ".4f"),
        ),
    ]

    for label, value in rows:
        print(f"  {label:<35} {value:>15}")

    print("─" * 55)

    warnings = getattr(m, "warnings", [])
    if warnings:
        print(f"\n  ⚠  {len(warnings)} diagnostic warning(s):")
        for w in warnings:
            print(f"     • {w}")
    print()


def _print_sweep_summary(df) -> None:
    """Print the first few rows of sweep results to console."""
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame) and not df.empty:
            print("\nSweep results preview (first 10 rows):")
            print(df.head(10).to_string(index=False))
    except (ImportError, Exception):
        if isinstance(df, list) and df:
            print("\nSweep results preview:")
            for row in df[:10]:
                print("  ", row)


def _find_default_config() -> Path | None:
    """Search for master_config.yaml starting from this file's location."""
    here = Path(__file__).resolve().parent
    project_root = here.parent.parent
    candidates = [
        project_root / "config" / "master_config.yaml",
        project_root / "config" / "master_config.yml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dwsim_model",
        description="DWSIM Gasification Model — headless runner and utilities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    subs = parser.add_subparsers(dest="command", metavar="<subcommand>")
    subs.required = True

    # ── run ──
    run_p = subs.add_parser("run", help="Run a single scenario.")
    run_p.add_argument("--config", help="Path to master_config.yaml")
    run_p.add_argument(
        "--scenario", default="baseline", help="Scenario name (default: baseline)"
    )
    run_p.add_argument("--output", help="Output directory (default: results/)")
    run_p.add_argument(
        "--save-dwxml", action="store_true", help="Also save a DWSIM .dwxml file"
    )
    run_p.add_argument(
        "--force",
        action="store_true",
        help="Continue even if the solver doesn't converge",
    )

    # ── sweep ──
    sw_p = subs.add_parser("sweep", help="Parameter sweep (1-D or 2-D).")
    sw_p.add_argument("--config", help="Path to master_config.yaml")
    sw_p.add_argument(
        "--param",
        required=True,
        help="Dot-path of parameter A, e.g. feeds.Gasifier_Biomass_Feed.mass_flow_kg_s",
    )
    sw_p.add_argument("--min", type=float, required=True, help="Min value for param A")
    sw_p.add_argument("--max", type=float, required=True, help="Max value for param A")
    sw_p.add_argument(
        "--steps", type=int, default=5, help="Number of steps (default: 5)"
    )
    sw_p.add_argument("--param-b", help="Second parameter path (for 2-D sweep)")
    sw_p.add_argument("--min-b", type=float, help="Min value for param B")
    sw_p.add_argument("--max-b", type=float, help="Max value for param B")
    sw_p.add_argument(
        "--steps-b", type=int, help="Steps for param B (default: same as --steps)"
    )
    sw_p.add_argument("--kpis", nargs="+", help="KPI names to record (default: all)")
    sw_p.add_argument(
        "--output",
        default="results/sweep.csv",
        help="Output CSV path (default: results/sweep.csv)",
    )

    # ── validate ──
    val_p = subs.add_parser("validate", help="Validate YAML config files.")
    val_p.add_argument("--config", help="Path to master_config.yaml")

    # ── export ──
    exp_p = subs.add_parser("export", help="Export flowsheet to DWSIM GUI file.")
    exp_p.add_argument("--config", help="Path to master_config.yaml")
    exp_p.add_argument("--output", help="Output .dwxml path")

    # ── summary ──
    subs.add_parser("summary", help="Print reaction configuration summary.")

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    dispatch = {
        "run": cmd_run,
        "sweep": cmd_sweep,
        "validate": cmd_validate,
        "export": cmd_export,
        "summary": cmd_summary,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.getLogger("dwsim_model.cli").error(
            f"Unexpected error: {exc}", exc_info=True
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
