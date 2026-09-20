"""Command line for the putting launch monitor.

    python -m putting_launch_monitor calibrate --camera ID --frame F.jpg \\
        --corners x,y x,y x,y x,y --mat-mm W L --out cal.json
    python -m putting_launch_monitor run    --calibration cal.json [--gspro]
    python -m putting_launch_monitor replay --calibration cal.json --video V.mkv
    python -m putting_launch_monitor probe-gspro
    python -m putting_launch_monitor snapshot --camera ID --out frame.jpg

Everything the GUI will do is reachable here first, so the pipeline is
exercised end to end on this machine before a widget exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from shared.python.camera import CaptureMode, FfmpegDirectShowSource, VideoFileSource
from shared.python.contracts import require

from .calibration import Calibration, default_calibration_path
from .gspro import GsproClient
from .monitor import FrameEvent, GsproSink, LogSink, PuttingMonitor
from .validate import add_validate_parser, cmd_validate

logger = logging.getLogger("putting_launch_monitor")
DEVICE_ID = "camera-putting-monitor"


def parse_point(text: str) -> tuple[float, float]:
    """``"x,y"`` -> ``(x, y)``. Precondition: two comma-separated numbers."""
    parts = text.split(",")
    require(len(parts) == 2, "point must be x,y", text)
    return float(parts[0]), float(parts[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putting_launch_monitor")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="write a calibration from mat corners")
    cal.add_argument("--camera", required=True, help="PnP instance id of the camera")
    cal.add_argument(
        "--corners",
        nargs=4,
        required=True,
        metavar="X,Y",
        help="mat corners in pixels: near-left near-right far-right far-left",
    )
    cal.add_argument("--mat-mm", nargs=2, type=float, required=True, metavar=("W", "L"))
    cal.add_argument("--target-deg", type=float, default=0.0)
    cal.add_argument("--colour", default="white", choices=["white", "orange", "yellow"])
    cal.add_argument("--fps", type=int, default=60)
    cal.add_argument(
        "--out", type=Path, default=None, help="default: the per-user calibration"
    )

    run = sub.add_parser("run", help="watch the camera and report putts")
    run.add_argument("--calibration", type=Path, default=None)
    run.add_argument("--gspro", action="store_true", help="send putts to GSPro")
    run.add_argument(
        "--gspro-always", action="store_true", help="send even outside putting mode"
    )
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=921)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument(
        "--width", type=int, default=960, help="decode width (speed vs. precision)"
    )

    rep = sub.add_parser("replay", help="run the pipeline over a recorded video")
    rep.add_argument("--calibration", type=Path, default=None)
    rep.add_argument("--video", type=Path, required=True)
    rep.add_argument("--fps", type=float, default=None, help="override the file's rate")
    rep.add_argument("--gspro", action="store_true")

    sub.add_parser("probe-gspro", help="connect to GSPro and print the player state")

    snap = sub.add_parser("snapshot", help="grab one frame for calibration")
    snap.add_argument("--camera", required=True)
    snap.add_argument("--out", type=Path, required=True)

    add_validate_parser(sub)
    return parser


def cmd_calibrate(args: argparse.Namespace) -> int:
    corners = tuple(parse_point(c) for c in args.corners)
    cal = Calibration(
        camera_instance_id=args.camera,
        mat_corners_px=corners,
        mat_width_mm=args.mat_mm[0],
        mat_length_mm=args.mat_mm[1],
        target_deg=args.target_deg,
        colour=args.colour,
        fps=args.fps,
    )
    err = cal.reprojection_error_px()
    out = args.out or default_calibration_path()
    cal.save(out)
    logger.info("calibration written to %s (reprojection %.3f px)", out, err)
    logger.info("ball radius bounds: %s", cal.detector())
    return 0


def _sink(args: argparse.Namespace) -> LogSink | GsproSink:
    if not getattr(args, "gspro", False):
        return LogSink()
    client = GsproClient(
        device_id=DEVICE_ID,
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", 921),
    )
    client.connect()
    client.heartbeat()
    logger.info(
        "GSPro connected; player=%s putting=%s", client.player, client.putting_mode
    )
    return GsproSink(client, always=getattr(args, "gspro_always", False))


def _report(event: FrameEvent) -> None:
    if event.putt is not None:
        p = event.putt
        logger.info(
            "PUTT %s: %.2f mph  HLA %+.2f deg  (%d pts, r2=%.3f, %.0f mm) -> %s",
            "accepted" if p.accepted else "rejected",
            p.speed_mph,
            p.hla_deg,
            p.launch.points,
            p.launch.r2,
            p.launch.span_mm,
            event.outcome or p.reason,
        )


def cmd_run(args: argparse.Namespace) -> int:
    cal = Calibration.load(args.calibration or default_calibration_path())
    mode = CaptureMode(width=cal.capture_width, height=cal.capture_height, fps=cal.fps)
    source = FfmpegDirectShowSource(cal.camera_instance_id, mode, width=args.width)
    scaled = scaled_calibration(cal, args.width)
    monitor = PuttingMonitor(scaled, source, _sink(args))
    monitor.add_observer(_report)
    logger.info(
        "watching %s at %dx? (%d fps); waiting for a resting ball",
        cal.camera_instance_id,
        args.width,
        cal.fps,
    )
    frames = monitor.run(max_frames=args.max_frames)
    logger.info("done: %d frames, %d putts", frames, len(monitor.putts))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    cal = Calibration.load(args.calibration or default_calibration_path())
    source = VideoFileSource(args.video, fps=args.fps)
    monitor = PuttingMonitor(cal, source, _sink(args))
    monitor.add_observer(_report)
    frames = monitor.run()
    accepted = [p for p in monitor.putts if p.accepted]
    logger.info(
        "replay: %d frames, %d putts (%d accepted)",
        frames,
        len(monitor.putts),
        len(accepted),
    )
    return 0 if accepted else 1


def cmd_probe(args: argparse.Namespace) -> int:
    client = GsproClient(device_id=DEVICE_ID)
    try:
        client.connect()
        reply = client.heartbeat()
    except OSError as exc:
        logger.error(
            "GSPro not reachable on 127.0.0.1:921: %s (is GSPro Connect running?)", exc
        )
        return 2
    logger.info(
        "GSPro replied %s %s; player=%s; putting mode=%s",
        reply.code,
        reply.message,
        client.player,
        client.putting_mode,
    )
    client.close()
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    import cv2

    from .monitor import frame_to_bgr

    source = FfmpegDirectShowSource(args.camera, CaptureMode())
    source.initialize()
    source.start_capture()
    try:
        for _ in range(10):  # let exposure settle
            packet = source.read_frame()
    finally:
        source.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), frame_to_bgr(packet))
    logger.info("wrote %s (%dx%d)", args.out, *packet.resolution_px)
    return 0


def scaled_calibration(cal: Calibration, width: int) -> Calibration:
    """The same calibration expressed at the decode width."""
    scale = width / cal.capture_width
    if abs(scale - 1.0) < 1e-9:
        return cal
    corners = tuple((x * scale, y * scale) for x, y in cal.mat_corners_px)
    roi = cal.roi
    if roi is not None:
        from .detect import RegionOfInterest

        roi = RegionOfInterest(
            int(roi.x0 * scale),
            int(roi.y0 * scale),
            int(roi.x1 * scale),
            int(roi.y1 * scale),
        )
    data = cal.to_dict()
    data["mat_corners_px"] = [list(c) for c in corners]
    data["roi"] = None
    scaled = Calibration.from_dict(data)
    return Calibration(**{**scaled.__dict__, "roi": roi})


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "calibrate": cmd_calibrate,
    "run": cmd_run,
    "replay": cmd_replay,
    "probe-gspro": cmd_probe,
    "snapshot": cmd_snapshot,
    "validate": cmd_validate,
}


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return int(COMMANDS[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
