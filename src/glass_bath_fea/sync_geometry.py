#!/usr/bin/env python3
"""Synchronise electrode configuration to Glass Bath FEA geometry.

Loads electrode advisor configuration, validates geometry compatibility,
generates the FEA-compatible geometry, and exports a JSON sync report.

Usage:
    python -m glass_bath_fea.sync_geometry [--output sync_report.json]

See issue #575.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Entry point for geometry synchronisation."""
    parser = argparse.ArgumentParser(
        description="Synchronise Electrode Advisor config to Glass Bath FEA geometry"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="geometry_sync_report.json",
        help="Path to output JSON report (default: geometry_sync_report.json)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate geometry, do not generate FEA config",
    )
    parser.add_argument(
        "--bath-diameter",
        type=float,
        default=None,
        help="Override bath diameter [inches]",
    )
    parser.add_argument(
        "--glass-depth",
        type=float,
        default=None,
        help="Override glass depth [inches]",
    )
    parser.add_argument(
        "--electrode-diameter",
        type=float,
        default=None,
        help="Override electrode tip diameter [inches]",
    )
    args = parser.parse_args(argv)

    # Import here so --help works without full dependency chain
    from upstream_drift_tools.calculators.electrical.config import ElectrodeConfig

    from glass_bath_fea.interfaces.geometry_sync import GeometrySynchronizer

    # Build electrode config with optional overrides
    ec = ElectrodeConfig()
    if args.bath_diameter is not None:
        ec.bath_diameter = args.bath_diameter
    if args.glass_depth is not None:
        ec.glass_depth = args.glass_depth
    if args.electrode_diameter is not None:
        ec.tip_diameter = args.electrode_diameter

    sync = GeometrySynchronizer(electrode_config=ec)

    # Validate
    result = sync.validate()
    if result.warnings:
        for w in result.warnings:
            logger.warning(w)

    if not result.is_valid:
        for e in result.errors:
            logger.error(e)
        logger.error("Geometry validation FAILED -- aborting.")
        return 1

    logger.info("Geometry validation passed.")

    if args.validate_only:
        return 0

    # Sync and generate report
    fea_config = sync.sync()
    logger.info(
        "FEA config generated: bath=%.1f in, glass=%.1f in, "
        "electrode=%.1f in, insertion=%.1f in, T=%.0f C",
        fea_config.bath_diameter,
        fea_config.glass_depth,
        fea_config.electrode_diameter,
        fea_config.electrode_insertion_depth,
        fea_config.operating_temperature,
    )

    positions = sync.get_electrode_positions_fea()
    logger.info("Generated %d electrode positions in FEA coordinates.", len(positions))

    # Export report
    output_path = Path(args.output)
    sync.export_sync_report(output_path)
    logger.info("Report written to %s", output_path)

    # Print summary to stdout
    summary = {
        "status": "OK",
        "bath_diameter_in": fea_config.bath_diameter,
        "glass_depth_in": fea_config.glass_depth,
        "electrode_count": len(positions),
        "electrode_diameter_in": fea_config.electrode_diameter,
        "insertion_depth_in": fea_config.electrode_insertion_depth,
        "output_file": str(output_path),
    }
    logger.info(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
