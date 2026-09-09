"""Contract tests for C3D biomechanical exchange (TOOLS-M9 #4716)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sidekick.lab.mocap.c3d import (
    C3DAnalogChannel,
    C3DContainer,
    C3DEvent,
    C3DHeader,
    C3DPointChannel,
    compute_center_of_pressure,
    parse_c3d_header,
    serialize_c3d_header,
    unit_scale_factor,
    validate_c3d_header_magic,
    write_c3d_file,
)


def test_c3d_header_contract() -> None:
    header = C3DHeader(
        point_count=10,
        analog_channels_per_frame=4,
        first_frame=1,
        last_frame=100,
        max_interpolation_gap=10,
        scale_factor=-0.05,
        data_start_block=2,
        analog_samples_per_frame=10,
        frame_rate_hz=120.0,
    )
    assert header.frame_count == 100
    assert header.analog_sample_rate_hz == 1200.0


def test_c3d_header_bounds_validation() -> None:
    with pytest.raises(ValueError, match="last_frame"):
        C3DHeader(
            point_count=10,
            analog_channels_per_frame=4,
            first_frame=50,
            last_frame=10,
            max_interpolation_gap=0,
            scale_factor=1.0,
            data_start_block=2,
            analog_samples_per_frame=1,
            frame_rate_hz=100.0,
        )

    with pytest.raises(ValueError, match="scale_factor"):
        C3DHeader(
            point_count=10,
            analog_channels_per_frame=4,
            first_frame=1,
            last_frame=100,
            max_interpolation_gap=0,
            scale_factor=0.0,
            data_start_block=2,
            analog_samples_per_frame=1,
            frame_rate_hz=100.0,
        )


def test_c3d_header_serialization_roundtrip() -> None:
    orig = C3DHeader(
        point_count=15,
        analog_channels_per_frame=6,
        first_frame=1,
        last_frame=250,
        max_interpolation_gap=5,
        scale_factor=-0.01,
        data_start_block=3,
        analog_samples_per_frame=8,
        frame_rate_hz=240.0,
    )
    raw_bytes = serialize_c3d_header(orig)
    assert len(raw_bytes) == 512
    parsed = parse_c3d_header(raw_bytes)
    assert parsed.point_count == orig.point_count
    assert parsed.analog_channels_per_frame == orig.analog_channels_per_frame
    assert parsed.first_frame == orig.first_frame
    assert parsed.last_frame == orig.last_frame
    assert pytest.approx(parsed.scale_factor, abs=1e-5) == orig.scale_factor
    assert pytest.approx(parsed.frame_rate_hz, abs=1e-4) == orig.frame_rate_hz


def test_c3d_file_write_and_header_validation(tmp_path: Path) -> None:
    header = C3DHeader(
        point_count=2,
        analog_channels_per_frame=1,
        first_frame=1,
        last_frame=5,
        max_interpolation_gap=0,
        scale_factor=-1.0,
        data_start_block=2,
        analog_samples_per_frame=1,
        frame_rate_hz=60.0,
    )
    pt1 = C3DPointChannel(
        label="R_HEEL",
        coordinates_xyz=(
            (0.1, 0.2, 0.3),
            (0.1, 0.2, 0.31),
            (0.1, 0.2, 0.32),
            (0.1, 0.2, 0.33),
            (0.1, 0.2, 0.34),
        ),
        residuals=(0.0, 0.0, 0.0, 0.0, 0.0),
        camera_masks=(3, 3, 3, 3, 3),
    )
    pt2 = C3DPointChannel(
        label="L_HEEL",
        coordinates_xyz=(
            (0.4, 0.5, 0.6),
            (0.4, 0.5, 0.61),
            (0.4, 0.5, 0.62),
            (0.4, 0.5, 0.63),
            (0.4, 0.5, 0.64),
        ),
        residuals=(0.0, 0.0, 0.0, 0.0, 0.0),
        camera_masks=(3, 3, 3, 3, 3),
    )
    analog1 = C3DAnalogChannel(
        label="EMG1",
        values=(0.01, 0.02, 0.05, 0.01, 0.0),
    )
    evt = C3DEvent(label="HEEL_STRIKE", time_s=0.033)

    container = C3DContainer(
        header=header,
        points=(pt1, pt2),
        analogs=(analog1,),
        events=(evt,),
    )

    out_file = tmp_path / "test_export.c3d"
    write_c3d_file(out_file, container)

    assert out_file.exists()
    assert out_file.stat().st_size >= 512 * 2
    validate_c3d_header_magic(out_file)

    assert container.get_point("R_HEEL") is pt1
    assert container.get_point("UNKNOWN") is None
    assert container.get_analog("EMG1") is analog1


def test_unit_scaling_factors() -> None:
    assert pytest.approx(unit_scale_factor("mm", "m"), abs=1e-6) == 0.001
    assert pytest.approx(unit_scale_factor("m", "mm"), abs=1e-6) == 1000.0
    assert pytest.approx(unit_scale_factor("in", "m"), abs=1e-6) == 0.0254
    assert pytest.approx(unit_scale_factor("m", "m"), abs=1e-6) == 1.0


def test_cop_calculation() -> None:
    fz = np.array([0.0, 500.0, 1000.0])
    mx = np.array([0.0, 50.0, 100.0])
    my = np.array([0.0, -100.0, -200.0])

    cop_x, cop_y, cop_z = compute_center_of_pressure(
        fz, mx, my, min_force_threshold_n=10.0, ground_height_m=0.0
    )

    # When fz == 0.0 (below threshold), COP should be NaN
    assert np.isnan(cop_x[0])
    assert np.isnan(cop_y[0])

    # cop_x = -my / fz = -(-100) / 500 = 0.2
    assert pytest.approx(cop_x[1], abs=1e-4) == 0.2
    # cop_y = mx / fz = 50 / 500 = 0.1
    assert pytest.approx(cop_y[1], abs=1e-4) == 0.1
    assert cop_z[1] == 0.0
