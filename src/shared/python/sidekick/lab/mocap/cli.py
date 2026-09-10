"""Headless command-line interface for markerless mocap workflows (TOOLS-M10 #4727)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .service import MocapService, MocapServiceConfig

logger = logging.getLogger(__name__)


def _emit(text: str, *, file: Any = None) -> None:
    target = file if file is not None else sys.stdout
    target.write(f"{text}\n")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command line parser for mocap operations."""
    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging"
    )
    common_parent.add_argument(
        "--json", action="store_true", help="Format output as JSON"
    )

    parser = argparse.ArgumentParser(
        prog="sidekick-mocap",
        description="Headless markerless motion capture workflow CLI",
        parents=[common_parent],
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Mocap subcommand to execute"
    )

    # discover
    sub_disc = subparsers.add_parser(
        "discover",
        parents=[common_parent],
        help="Discover capture devices and sensor capabilities",
    )
    sub_disc.add_argument("--device-type", default=None, help="Filter devices by type")

    # capture
    sub_cap = subparsers.add_parser(
        "capture",
        parents=[common_parent],
        help="Execute synchronized multi-camera capture",
    )
    sub_cap.add_argument(
        "--cameras", nargs="+", default=["cam_0"], help="List of camera identifiers"
    )
    sub_cap.add_argument(
        "--duration", type=float, default=1.0, help="Capture duration in seconds"
    )
    sub_cap.add_argument(
        "--fps", type=float, default=30.0, help="Capture frame rate in Hz"
    )
    sub_cap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("capture_output"),
        help="Output directory",
    )
    sub_cap.add_argument(
        "--no-store", action="store_true", help="Enforce privacy no-store policy"
    )
    sub_cap.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Use synthetic frame sources",
    )

    # calibrate
    sub_cal = subparsers.add_parser(
        "calibrate",
        parents=[common_parent],
        help="Solve intrinsic and extrinsic camera calibration",
    )
    sub_cal.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="Path to calibration observations JSON",
    )
    sub_cal.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for camera layout JSON",
    )
    sub_cal.add_argument(
        "--pattern-kind", default="checkerboard", help="Target pattern type"
    )

    # reconstruct
    sub_rec = subparsers.add_parser(
        "reconstruct",
        parents=[common_parent],
        help="Triangulate 3D keypoints and reconstruct trajectories",
    )
    sub_rec.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="Path to multi-view 2D observations JSON",
    )
    sub_rec.add_argument(
        "--layout",
        type=Path,
        required=True,
        help="Path to calibrated camera layout JSON",
    )
    sub_rec.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for reconstructed 3D JSON",
    )
    sub_rec.add_argument(
        "--filter",
        default="none",
        choices=["none", "savgol", "butterworth"],
        help="Temporal filter",
    )

    # export
    sub_exp = subparsers.add_parser(
        "export",
        parents=[common_parent],
        help="Export motion capture data to C3D or JSON",
    )
    sub_exp.add_argument(
        "--input", type=Path, required=True, help="Input trajectory JSON file"
    )
    sub_exp.add_argument(
        "--format", default="c3d", choices=["c3d", "json"], help="Output format"
    )
    sub_exp.add_argument(
        "--output", type=Path, required=True, help="Destination output file path"
    )
    sub_exp.add_argument(
        "--unit-scale", type=float, default=1.0, help="Coordinate scale factor"
    )

    return parser


def _cmd_discover(args: argparse.Namespace, service: MocapService) -> int:
    caps = service.capabilities()
    data = {
        "status": "success",
        "devices": caps.supported_devices,
        "backends": caps.supported_backends,
        "export_formats": caps.supported_export_formats,
    }
    if args.json:
        _emit(json.dumps(data, indent=2))
    else:
        _emit("=== Available Cameras / Capture Devices ===")
        for dev in caps.supported_devices:
            _emit(f"  - {dev}")
        _emit("\n=== Supported Pose Inference Backends ===")
        for bk in caps.supported_backends:
            _emit(f"  - {bk}")
    return 0


def _cmd_capture(args: argparse.Namespace, service: MocapService) -> int:
    task_id = service.start_capture(
        camera_ids=args.cameras,
        duration_seconds=args.duration,
        fps=args.fps,
        output_dir=args.output_dir,
        synthetic=args.synthetic,
    )
    result = service.wait_task(task_id, timeout_seconds=args.duration + 5.0)
    out_payload = {
        "success": result.success,
        "task_id": result.task_id,
        "no_store": service.config.no_store,
        "message": result.message,
    }
    if args.json:
        _emit(json.dumps(out_payload, indent=2))
    else:
        _emit(
            f"Capture completed: task={task_id}, "
            f"success={result.success}, no_store={service.config.no_store}"
        )
    return 0 if result.success else 1


def _cmd_calibrate(args: argparse.Namespace, service: MocapService) -> int:
    res = service.calibrate(args.observations, output_path=args.output)
    if args.json:
        _emit(json.dumps(res, indent=2))
    else:
        _emit(
            f"Calibration solved for {res.get('camera_count', 0)} "
            f"cameras -> {args.output}"
        )
    return 0 if res.get("success") else 1


def _cmd_reconstruct(args: argparse.Namespace, service: MocapService) -> int:
    res = service.reconstruct(
        observations_path=args.observations,
        layout_path=args.layout,
        output_path=args.output,
        filter_kind=args.filter,
    )
    if args.json:
        _emit(json.dumps(res, indent=2))
    else:
        _emit(
            f"Reconstruction completed: {res.get('reconstructed_frames', 0)} "
            f"frames -> {args.output}"
        )
    return 0 if res.get("success") else 1


def _cmd_export(args: argparse.Namespace, service: MocapService) -> int:
    res = service.export(
        input_path=args.input,
        format_kind=args.format,
        output_path=args.output,
        unit_scale=args.unit_scale,
    )
    if args.json:
        _emit(json.dumps(res, indent=2))
    else:
        _emit(f"Exported trajectory to {args.output} (format={args.format})")
    return 0 if res.get("success") else 1


def cli_main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint returning standard process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    if args.command is None:
        parser.print_usage(sys.stderr)
        return 2

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    no_store = getattr(args, "no_store", False)
    service = MocapService(MocapServiceConfig(no_store=no_store))

    try:
        if args.command == "discover":
            return _cmd_discover(args, service)
        if args.command == "capture":
            return _cmd_capture(args, service)
        if args.command == "calibrate":
            return _cmd_calibrate(args, service)
        if args.command == "reconstruct":
            return _cmd_reconstruct(args, service)
        if args.command == "export":
            return _cmd_export(args, service)
        _emit(f"Unknown command: {args.command}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, canceling service tasks")
        service.cancel()
        return 130
    except Exception as exc:
        logger.exception("Mocap CLI execution error: %s", exc)
        if args.json:
            _emit(json.dumps({"success": False, "error": str(exc)}, indent=2))
        else:
            _emit(f"Error: {exc}", file=sys.stderr)
        return 1
