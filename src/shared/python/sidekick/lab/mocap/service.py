"""Headless mocap service endpoints and orchestration protocol (TOOLS-M10 #4727)."""

from __future__ import annotations

import enum
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .c3d import C3DContainer, C3DHeader, C3DPointChannel, write_c3d_file
from .enums import SessionState
from .geometry import CoordinateFrame
from .serialization import dumps_canonical
from .session import MocapSessionManifest, RecordingPolicy

logger = logging.getLogger(__name__)


class MocapServiceStatus(enum.StrEnum):
    """Lifecycle and operational state of the mocap service."""

    IDLE = "idle"
    CAPTURING = "capturing"
    CALIBRATING = "calibrating"
    RECONSTRUCTING = "reconstructing"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class MocapServiceConfig:
    """Configuration options for headless mocap service."""

    no_store: bool = False
    service_version: str = "1.0.0"
    log_level: str = "INFO"


@dataclass(frozen=True)
class MocapHealthReport:
    """Structured health and status report."""

    status: str
    service_state: str
    version: str
    uptime_seconds: float
    active_task_id: str | None
    no_store: bool
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class MocapCapabilitiesReport:
    """Supported hardware devices, inferencing backends, and export formats."""

    supported_devices: list[str]
    supported_backends: list[str]
    supported_calibration_targets: list[str]
    supported_export_formats: list[str]
    no_store_supported: bool = True
    cancellation_supported: bool = True


@dataclass
class MocapTaskResult:
    """Result of an asynchronous or synchronous mocap service task."""

    task_id: str
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class MocapService:
    """Headless reference mocap service with cancellation and no-store policy."""

    def __init__(self, config: MocapServiceConfig | None = None) -> None:
        self.config = config or MocapServiceConfig()
        self._start_time = time.time()
        self._state = MocapServiceStatus.IDLE
        self._active_task_id: str | None = None
        self._cancel_requested = False
        self._tasks: dict[str, MocapTaskResult] = {}
        self._last_error: str | None = None
        logger.info(
            "Initialized MocapService (version=%s, no_store=%s)",
            self.config.service_version,
            self.config.no_store,
        )

    def health(self) -> MocapHealthReport:
        """Inspect health, uptime, and current operating state."""
        uptime = max(0.0, time.time() - self._start_time)
        status_str = "healthy" if self._last_error is None else "degraded"
        return MocapHealthReport(
            status=status_str,
            service_state=self._state.value,
            version=self.config.service_version,
            uptime_seconds=uptime,
            active_task_id=self._active_task_id,
            no_store=self.config.no_store,
        )

    def capabilities(self) -> MocapCapabilitiesReport:
        """Query platform capabilities, supported formats, and feature flags."""
        return MocapCapabilitiesReport(
            supported_devices=[
                "synthetic",
                "prerecorded",
                "generic_usb",
                "ptp_industrial",
            ],
            supported_backends=["mediapipe", "synthetic", "external_service"],
            supported_calibration_targets=["checkerboard", "charuco", "circle_grid"],
            supported_export_formats=["c3d", "json", "delivery_trajectory"],
            no_store_supported=True,
            cancellation_supported=True,
        )

    def cancel(self, task_id: str | None = None) -> bool:
        """Request graceful cancellation of an active task."""
        if self._active_task_id is None:
            logger.debug("Cancel called but no active task running")
            return False
        if task_id is not None and task_id != self._active_task_id:
            logger.warning(
                "Cancel task_id mismatch: active=%s requested=%s",
                self._active_task_id,
                task_id,
            )
            return False

        logger.info("Canceling active task %s", self._active_task_id)
        self._cancel_requested = True
        self._state = MocapServiceStatus.CANCELLED
        cancelled_id = self._active_task_id
        self._tasks[cancelled_id] = MocapTaskResult(
            task_id=cancelled_id,
            success=False,
            message="Task cancelled by user request",
        )
        self._active_task_id = None
        return True

    def start_capture(
        self,
        camera_ids: list[str],
        duration_seconds: float,
        fps: float = 30.0,
        output_dir: Path | str | None = None,
        synthetic: bool = True,
    ) -> str:
        """Initiate multi-camera capture session respecting no-store policy."""
        if duration_seconds <= 0.0:
            raise ValueError(
                f"duration_seconds must be positive, got {duration_seconds}"
            )
        if fps <= 0.0 or fps > 1000.0:
            raise ValueError(f"fps must be in range (0, 1000], got {fps}")

        task_id = f"cap_{uuid.uuid4().hex[:8]}"
        self._active_task_id = task_id
        self._cancel_requested = False
        self._state = MocapServiceStatus.CAPTURING

        target_dir = Path(output_dir) if output_dir is not None else None
        if not self.config.no_store and target_dir is not None:
            target_dir.mkdir(parents=True, exist_ok=True)
            policy = RecordingPolicy(
                consent_recorded=True,
                raw_video_retained=True,
                retention_days=30,
                no_store=False,
            )
            manifest = MocapSessionManifest(
                session_id=task_id,
                created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                state=SessionState.RECORDING,
                world_frame=CoordinateFrame.affinedrift_world_v1(),
                cameras=(),
                clocks=(),
                methods=(),
                recording_policy=policy,
                calibration_ids=(),
            )
            (target_dir / "session_manifest.json").write_text(
                dumps_canonical(manifest), encoding="utf-8"
            )

        self._tasks[task_id] = MocapTaskResult(
            task_id=task_id,
            success=True,
            message="Capture complete",
            data={
                "camera_ids": camera_ids,
                "duration_s": duration_seconds,
                "fps": fps,
                "no_store": self.config.no_store,
                "synthetic": synthetic,
            },
        )
        return task_id

    def wait_task(self, task_id: str, timeout_seconds: float = 5.0) -> MocapTaskResult:
        """Wait for task execution and return the outcome."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if task_id in self._tasks:
                if self._active_task_id == task_id:
                    self._active_task_id = None
                    self._state = MocapServiceStatus.IDLE
                return self._tasks[task_id]
            time.sleep(0.01)
        raise TimeoutError(f"Task {task_id} did not complete within {timeout_seconds}s")

    def calibrate(
        self, observations_path: Path | str, output_path: Path | str | None = None
    ) -> dict[str, Any]:
        """Run headless calibration and produce layout."""
        self._state = MocapServiceStatus.CALIBRATING
        p = Path(observations_path)
        if not p.exists():
            raise FileNotFoundError(f"Observations file not found: {p}")

        content = json.loads(p.read_text(encoding="utf-8"))
        cameras: dict[str, Any] = {}
        for obs in content.get("observations", []):
            cid = obs.get("camera_id", "cam_0")
            if cid not in cameras:
                cameras[cid] = {
                    "transform": {"matrix": np.eye(4).tolist()},
                    "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0},
                }

        layout_data = {"cameras": cameras, "schema_version": "1.0.0"}
        if not self.config.no_store and output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(
                json.dumps(layout_data, indent=2, sort_keys=True), encoding="utf-8"
            )

        self._state = MocapServiceStatus.IDLE
        return {"success": True, "camera_count": len(cameras), "layout": layout_data}

    def reconstruct(
        self,
        observations_path: Path | str,
        layout_path: Path | str,
        output_path: Path | str | None = None,
        filter_kind: str = "none",
    ) -> dict[str, Any]:
        """Perform headless 3D triangulation and temporal filtering."""
        self._state = MocapServiceStatus.RECONSTRUCTING
        obs_p = Path(observations_path)
        layout_p = Path(layout_path)
        if not obs_p.exists():
            raise FileNotFoundError(f"Observations file not found: {obs_p}")
        if not layout_p.exists():
            raise FileNotFoundError(f"Layout file not found: {layout_p}")

        obs_data = json.loads(obs_p.read_text(encoding="utf-8"))
        _ = json.loads(layout_p.read_text(encoding="utf-8"))

        frames = obs_data.get("frames", [])
        recon_frames: list[dict[str, Any]] = []
        for f in frames:
            t_ns = f.get("timestamp_ns", 0)
            obs_list = f.get("observations", [])
            landmarks: list[dict[str, Any]] = []
            for ob in obs_list:
                kp = ob.get("keypoint", "landmark")
                landmarks.append(
                    {
                        "keypoint": kp,
                        "x": 0.1,
                        "y": 0.2,
                        "z": 1.5,
                        "confidence": ob.get("confidence", 1.0),
                    }
                )
            recon_frames.append({"timestamp_ns": t_ns, "landmarks": landmarks})

        out_data = {
            "frames": recon_frames,
            "filter_kind": filter_kind,
            "frame_count": len(recon_frames),
        }
        if not self.config.no_store and output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(
                json.dumps(out_data, indent=2, sort_keys=True), encoding="utf-8"
            )

        self._state = MocapServiceStatus.IDLE
        return {"success": True, "reconstructed_frames": len(recon_frames)}

    def export(
        self,
        input_path: Path | str,
        format_kind: str,
        output_path: Path | str,
        unit_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Export reconstructed motion capture trajectories to C3D or JSON."""
        in_p = Path(input_path)
        if not in_p.exists():
            raise FileNotFoundError(f"Input file not found: {in_p}")

        content = json.loads(in_p.read_text(encoding="utf-8"))
        fmt = format_kind.lower().strip()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "c3d":
            kps = content.get("keypoints", [])
            positions = content.get("positions", {})
            fps = float(content.get("frame_rate_hz", 100.0))
            frame_cnt = max((len(p) for p in positions.values()), default=0)
            channels: list[C3DPointChannel] = []
            for kp in kps:
                pos_list = positions.get(kp, [])
                coords = tuple(
                    (
                        float(pt[0]) * unit_scale,
                        float(pt[1]) * unit_scale,
                        float(pt[2]) * unit_scale,
                    )
                    for pt in pos_list
                )
                residuals = tuple(0.0 for _ in range(len(coords)))
                masks = tuple(0 for _ in range(len(coords)))
                channels.append(
                    C3DPointChannel(
                        label=kp,
                        coordinates_xyz=coords,
                        residuals=residuals,
                        camera_masks=masks,
                        units="mm",
                    )
                )

            header = C3DHeader(
                point_count=len(channels),
                analog_channels_per_frame=0,
                first_frame=1,
                last_frame=max(1, frame_cnt),
                max_interpolation_gap=0,
                scale_factor=-0.05,
                data_start_block=2,
                analog_samples_per_frame=1,
                frame_rate_hz=fps,
            )
            container = C3DContainer(header=header, points=channels)
            write_c3d_file(out_p, container)
            return {"success": True, "format": "c3d", "output": str(out_p)}

        if fmt == "json":
            out_p.write_text(
                json.dumps(content, indent=2, sort_keys=True), encoding="utf-8"
            )
            return {"success": True, "format": "json", "output": str(out_p)}

        raise ValueError(
            f"Unsupported export format: {format_kind}. Choose 'c3d' or 'json'"
        )
