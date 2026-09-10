"""Contract tests for mocap reference CLI and service (TOOLS-M10 #4727)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sidekick.lab.mocap.c3d import parse_c3d_header
from sidekick.lab.mocap.cli import cli_main
from sidekick.lab.mocap.service import (
    MocapCapabilitiesReport,
    MocapHealthReport,
    MocapService,
    MocapServiceConfig,
    MocapServiceStatus,
)


def test_service_health_endpoint() -> None:
    service = MocapService(MocapServiceConfig(no_store=False))
    health = service.health()
    assert isinstance(health, MocapHealthReport)
    assert health.status == "healthy"
    assert health.service_state == MocapServiceStatus.IDLE.value
    assert health.uptime_seconds >= 0.0
    assert health.no_store is False
    assert health.active_task_id is None


def test_service_capabilities_endpoint() -> None:
    service = MocapService()
    caps = service.capabilities()
    assert isinstance(caps, MocapCapabilitiesReport)
    assert "synthetic" in caps.supported_devices
    assert "mediapipe" in caps.supported_backends
    assert "c3d" in caps.supported_export_formats
    assert caps.no_store_supported is True
    assert caps.cancellation_supported is True


def test_service_cancellation() -> None:
    service = MocapService()
    # Cancel when idle should return False
    assert service.cancel() is False

    # Start a mock long-running task and cancel it
    task_id = service.start_capture(
        camera_ids=["cam_0", "cam_1"],
        duration_seconds=10.0,
        synthetic=True,
    )
    assert service.health().service_state == MocapServiceStatus.CAPTURING.value
    assert service.cancel(task_id) is True
    assert service.health().service_state == MocapServiceStatus.CANCELLED.value


def test_service_no_store_policy(tmp_path: Path) -> None:
    # Under no_store mode, capture should not persist raw recording files to disk
    service = MocapService(MocapServiceConfig(no_store=True))
    task_id = service.start_capture(
        camera_ids=["cam_0"],
        duration_seconds=0.1,
        synthetic=True,
        output_dir=tmp_path,
    )
    result = service.wait_task(task_id, timeout_seconds=2.0)
    assert result.success is True
    # Verify no raw video or frame files are written to tmp_path
    disk_files = list(tmp_path.glob("**/*.*"))
    assert len(disk_files) == 0


def test_cli_discover_text(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(["discover"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Available Cameras / Capture Devices" in captured.out
    assert "synthetic" in captured.out


def test_cli_discover_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(["discover", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "devices" in data
    assert len(data["devices"]) > 0


def test_cli_capture_synthetic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "capture_session"
    code = cli_main(
        [
            "capture",
            "--synthetic",
            "--duration",
            "0.2",
            "--fps",
            "30",
            "--output-dir",
            str(out_dir),
            "--json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is True
    assert (out_dir / "session_manifest.json").exists()


def test_cli_capture_no_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "ephemeral_session"
    code = cli_main(
        [
            "capture",
            "--synthetic",
            "--duration",
            "0.1",
            "--output-dir",
            str(out_dir),
            "--no-store",
            "--json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["no_store"] is True
    assert not out_dir.exists() or len(list(out_dir.glob("*"))) == 0


def test_cli_calibrate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    obs_file = tmp_path / "obs.json"
    # Write synthetic observations
    obs_data = {
        "observations": [
            {
                "camera_id": "cam_0",
                "point_id": "p0",
                "u": 320.0,
                "v": 240.0,
                "confidence": 1.0,
            },
            {
                "camera_id": "cam_1",
                "point_id": "p0",
                "u": 350.0,
                "v": 240.0,
                "confidence": 1.0,
            },
        ],
        "world_points": {
            "p0": [0.0, 0.0, 2.0],
        },
    }
    obs_file.write_text(json.dumps(obs_data), encoding="utf-8")
    out_layout = tmp_path / "calib_layout.json"

    code = cli_main(
        [
            "calibrate",
            "--observations",
            str(obs_file),
            "--output",
            str(out_layout),
            "--json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is True
    assert out_layout.exists()


def test_cli_reconstruct(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    obs_file = tmp_path / "obs_recon.json"
    layout_file = tmp_path / "layout_recon.json"

    # Layout with 2 cameras
    layout_data = {
        "cameras": {
            "cam_0": {
                "transform": {"matrix": np.eye(4).tolist()},
                "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0},
            },
            "cam_1": {
                "transform": {
                    "matrix": [
                        [1.0, 0.0, 0.0, -0.5],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
                "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0},
            },
        }
    }
    layout_file.write_text(json.dumps(layout_data), encoding="utf-8")

    # Observations across 5 frames
    obs_frames = {
        "frames": [
            {
                "timestamp_ns": i * 10_000_000,
                "observations": [
                    {
                        "camera_id": "cam_0",
                        "keypoint": "nose",
                        "u": 320.0,
                        "v": 240.0,
                        "confidence": 0.95,
                    },
                    {
                        "camera_id": "cam_1",
                        "keypoint": "nose",
                        "u": 120.0,
                        "v": 240.0,
                        "confidence": 0.95,
                    },
                ],
            }
            for i in range(5)
        ]
    }
    obs_file.write_text(json.dumps(obs_frames), encoding="utf-8")
    out_traj = tmp_path / "reconstructed_traj.json"

    code = cli_main(
        [
            "reconstruct",
            "--observations",
            str(obs_file),
            "--layout",
            str(layout_file),
            "--output",
            str(out_traj),
            "--filter",
            "savgol",
            "--json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is True
    assert out_traj.exists()


def test_cli_export_c3d(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Create input trajectory
    traj_file = tmp_path / "traj_input.json"
    traj_data = {
        "keypoints": ["R_ANKLE", "L_ANKLE"],
        "timestamps_ns": [0, 10_000_000, 20_000_000],
        "positions": {
            "R_ANKLE": [[0.1, 0.2, 0.3], [0.11, 0.21, 0.31], [0.12, 0.22, 0.32]],
            "L_ANKLE": [[-0.1, 0.2, 0.3], [-0.11, 0.21, 0.31], [-0.12, 0.22, 0.32]],
        },
        "frame_rate_hz": 100.0,
    }
    traj_file.write_text(json.dumps(traj_data), encoding="utf-8")
    out_c3d = tmp_path / "exported.c3d"

    code = cli_main(
        [
            "export",
            "--input",
            str(traj_file),
            "--format",
            "c3d",
            "--output",
            str(out_c3d),
            "--json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is True
    assert out_c3d.exists()
    assert out_c3d.stat().st_size >= 512

    # Verify header via parser
    header = parse_c3d_header(out_c3d.read_bytes())
    assert header.point_count == 2
    assert header.frame_count == 3


def test_cli_missing_args_fails() -> None:
    code = cli_main([])
    assert code == 2

    code = cli_main(["export"])
    assert code == 2
