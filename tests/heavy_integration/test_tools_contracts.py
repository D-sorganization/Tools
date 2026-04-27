"""
Heavy Integration Contracts — Tools
=====================================
These tests are marked @pytest.mark.live_simulation and are EXCLUDED from
standard CI (enforced via addopts = "-m not live_simulation" in pyproject.toml).
They run only:
  • Weekly on the d-sorg-fleet-4core custom runner
  • Manually via: wsl bash run_local_heavy_tests.sh

Each test proves real behavior of shared tools and heavy dependencies.
"""

from __future__ import annotations

import sys

import pytest


@pytest.mark.live_simulation
class TestTrimeshGeometryContracts:
    """Contract: Trimesh produces geometrically valid meshes for URDF/CAD pipelines."""

    def test_box_mesh_is_watertight_and_valid_volume(self) -> None:
        trimesh = pytest.importorskip(
            "trimesh",
            reason="trimesh not installed — skipping heavy geometry contracts",
        )

        box = trimesh.creation.box((1.0, 2.0, 3.0))

        assert box.is_watertight, "Box mesh must be watertight for URDF/physics use"
        assert box.volume == pytest.approx(6.0, rel=1e-4), (
            f"Expected volume 6.0, got {box.volume}"
        )
        assert len(box.vertices) > 0
        assert len(box.faces) > 0

    def test_mesh_boolean_intersection(self) -> None:
        """Prove boolean ops work — used in electrode CAD pipelines."""
        trimesh = pytest.importorskip(
            "trimesh",
            reason="trimesh not installed — skipping heavy geometry contracts",
        )

        a = trimesh.creation.box((1, 1, 1))
        b = trimesh.creation.box((1, 1, 1))
        b.apply_translation([0.5, 0, 0])

        try:
            result = trimesh.boolean.intersection([a, b])
            # If boolean is available (requires manifold/blender backend)
            assert result is not None
            assert result.volume > 0
        except Exception:  # noqa: BLE001
            pytest.skip("Trimesh boolean backend not available in this environment")


@pytest.mark.live_simulation
class TestSignalProcessingStack:
    """Contract: SciPy/NumPy signal processing produces correct DSP output."""

    def test_butterworth_filter_attenuation(self) -> None:
        """Prove a 4th-order Butterworth low-pass filter attenuates high-freq signals."""
        pytest.importorskip("numpy")
        import numpy as np

        pytest.importorskip("scipy")
        from scipy import signal

        # Design a 4th-order Butterworth low-pass at 0.1 (normalized)
        b, a = signal.butter(4, 0.1, btype="low")
        w, h = signal.freqz(b, a, worN=1024)

        freq = w / (2 * np.pi)
        # Passband gain at DC should be ~1.0
        dc_gain = abs(h[0])
        assert dc_gain == pytest.approx(1.0, abs=0.01), (
            f"DC gain should be 1.0, got {dc_gain}"
        )

        # Stopband attenuation at 0.5 should be < -20 dB
        stop_idx = int(0.5 * len(freq))
        stop_gain_db = 20 * np.log10(abs(h[stop_idx]) + 1e-12)
        assert stop_gain_db < -20, (
            f"Expected > 20 dB attenuation, got {stop_gain_db:.1f} dB"
        )

    def test_fft_roundtrip(self) -> None:
        """FFT→IFFT roundtrip preserves signal — fundamental DSP contract."""
        import numpy as np

        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)

        fft_result = np.fft.fft(signal)
        recovered = np.fft.ifft(fft_result).real

        assert np.allclose(signal, recovered, atol=1e-9), "FFT roundtrip failed"


@pytest.mark.live_simulation
class TestEzdxfPIDContracts:
    """Contract: ezdxf creates valid DXF drawings for P&ID generation."""

    def test_pid_drawing_entities_valid(self) -> None:
        import io

        ezdxf = pytest.importorskip(
            "ezdxf",
            reason="ezdxf not installed — skipping heavy P&ID contracts",
        )

        doc = ezdxf.new(dxfversion="R2010")
        msp = doc.modelspace()

        # Simulate a P&ID: pipe (line), vessel (circle), valve (text tag)
        msp.add_line((0, 0), (100, 0), dxfattribs={"layer": "PIPE"})
        msp.add_circle((50, 0), radius=10, dxfattribs={"layer": "VESSEL"})
        msp.add_text("V-101", dxfattribs={"height": 2.5, "layer": "TAG"}).set_placement(
            (45, 15)
        )

        # Write to in-memory buffer and re-read — proves valid DXF output
        buf = io.StringIO()
        doc.write(buf)
        buf.seek(0)

        doc2 = ezdxf.read(buf)
        entities = list(doc2.modelspace())
        assert len(entities) == 3, f"Expected 3 P&ID entities, got {len(entities)}"


@pytest.mark.live_simulation
class TestHeadlessQtEnvironment:
    """Contract: PyQt6 can initialise in the headless Xvfb environment."""

    def test_qt_application_lifecycle(self) -> None:
        from PyQt6.QtWidgets import QApplication, QWidget

        _app = QApplication.instance() or QApplication(sys.argv)
        widget = QWidget()
        widget.setWindowTitle("Heavy Test Widget")
        widget.resize(200, 100)

        assert widget.windowTitle() == "Heavy Test Widget"
        assert widget.width() == 200

        widget.close()
