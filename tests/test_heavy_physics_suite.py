"""
Comprehensive Heavy & Integration Test Suite for Tools
Designed for execution on the fleet-custom-runner or equivalent local Docker image.

This suite ensures the shared capabilities installed in the heavy image
(GUI, MuJoCo, Trimesh, CAD, Signal Processing, etc.) load and execute properly inside
the common architecture provided by the ud_tools repository.
"""

import sys

import pytest


@pytest.mark.live_simulation
class TestHeavyToolsIntegrations:
    """Rigorous tests forcing actual simulation tools to initialize."""

    def test_mujoco_api_readiness(self):
        """Verify MuJoCo loads and can parse basic XML for general tools."""
        import mujoco

        xml = """
        <mujoco>
            <worlddir>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                <body pos="0 0 1">
                    <joint type="free"/>
                    <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
                </body>
            </worlddir>
        </mujoco>
        """
        try:
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)

            initial_z = data.qpos[2]
            for _ in range(5):
                mujoco.mj_step(model, data)

            assert data.qpos[2] < initial_z
        except Exception as e:
            pytest.fail(f"MuJoCo integration failed in Tools: {e}")

    def test_trimesh_and_cad_integration(self):
        """Verify Trimesh library can be used for geometric processing."""
        import trimesh

        # Create a basic sphere mesh
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

        # Verify basic geometry queries
        assert len(mesh.vertices) > 10
        assert len(mesh.faces) > 10
        assert mesh.volume > 0

    def test_ezdxf_pid_generation_capabilities(self):
        """Verify EZDXF can build drawings for P&ID architecture."""
        try:
            import ezdxf

            # Create a new DXF document.
            doc = ezdxf.new(dxfversion="R2010")
            msp = doc.modelspace()

            # Add a line and verify it was added
            line = msp.add_line((0, 0), (10, 0))
            assert line is not None
        except ImportError:
            pytest.skip("EZDXF library not installed or failed to import")

    def test_signal_processing_stack(self):
        """Verify scipy and numpy correctly integrate for signal-toolkit logic."""
        import numpy as np
        from scipy import signal

        # Create a 10 Hz sine wave
        t = np.linspace(0, 1, 1000, endpoint=False)
        sig = np.sin(2 * np.pi * 10 * t)

        # Apply a Butterworth filter
        sos = signal.butter(10, 15, "lp", fs=1000, output="sos")
        filtered = signal.sosfilt(sos, sig)

        # Ensure array was correctly processed
        assert len(filtered) == 1000

    def test_qt_integration_headless(self):
        """Verify proper headless display capability via xvfb for GUI testing."""
        from PyQt6.QtWidgets import QApplication, QMainWindow

        # This will crash if xvfb is not running natively on the custom runner
        _app = QApplication.instance() or QApplication(sys.argv)

        window = QMainWindow()
        assert window is not None
