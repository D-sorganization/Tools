"""CLI for generating drafting artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vessel_drafter.exporters.step_export import (
    export_cylindrical_bath_layout_step,
    export_default_layout_step,
    export_vessel_drafter_default_step,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Programmatic drafting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export-electrode-advisor-default",
        help="Export the default electrode advisor STEP artifact.",
    )
    export_parser.add_argument(
        "--output",
        default="generated/electrode_advisor_default/electrode_advisor_default_layout.step",
        help="STEP output path.",
    )
    export_parser.add_argument(
        "--manifest",
        default="generated/electrode_advisor_default/electrode_advisor_default_layout.json",
        help="JSON manifest output path.",
    )

    cylindrical_parser = subparsers.add_parser(
        "export-cylindrical-bath-layout",
        help="Export the cylindrical bath radial-electrode STEP artifact.",
    )
    cylindrical_parser.add_argument(
        "--output",
        default="generated/cylindrical_bath_layout/cylindrical_bath_layout.step",
        help="STEP output path.",
    )
    cylindrical_parser.add_argument(
        "--manifest",
        default="generated/cylindrical_bath_layout/cylindrical_bath_layout.json",
        help="JSON manifest output path.",
    )

    vessel_parser = subparsers.add_parser(
        "export-vessel-drafter-default",
        help="Export the default vessel drafter STEP artifact.",
    )
    vessel_parser.add_argument(
        "--output",
        default="generated/vessel_drafter_default/vessel_drafter_default.step",
        help="STEP output path.",
    )
    vessel_parser.add_argument(
        "--manifest",
        default="generated/vessel_drafter_default/vessel_drafter_default.json",
        help="JSON manifest output path.",
    )

    subparsers.add_parser(
        "launch-vessel-drafter-gui",
        help="Launch the PyQt6 vessel drafter GUI.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "export-electrode-advisor-default":
        step_path = export_default_layout_step(
            output_path=Path(args.output),
            manifest_path=Path(args.manifest),
        )
        sys.stdout.write(str(step_path) + "\n")
        return 0

    if args.command == "export-cylindrical-bath-layout":
        step_path = export_cylindrical_bath_layout_step(
            output_path=Path(args.output),
            manifest_path=Path(args.manifest),
        )
        sys.stdout.write(str(step_path) + "\n")
        return 0

    if args.command == "export-vessel-drafter-default":
        step_path = export_vessel_drafter_default_step(
            output_path=Path(args.output),
            manifest_path=Path(args.manifest),
        )
        sys.stdout.write(str(step_path) + "\n")
        return 0

    if args.command == "launch-vessel-drafter-gui":
        from vessel_drafter.gui.vessel_drafter_window import (
            launch as launch_vessel_drafter,
        )

        result: int = launch_vessel_drafter()
        return result

    parser.error(f"Unknown command: {args.command}")
    return 2  # type: ignore[unreachable]


if __name__ == "__main__":
    raise SystemExit(main())
