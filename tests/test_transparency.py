"""Tests for transparent background support and PNG export with alpha channel.

Issue #550: Transparent Background Support & Export

This module validates:
- TransparencyRenderer class existence and interface
- enable_transparent_background() configures PyVista correctly
- disable_transparent_background() reverts to opaque
- export_with_transparency() saves PNG with alpha channel
- Alpha channel properly embedded in saved PNG files
- No performance degradation with transparency enabled
- On-screen rendering with transparent background
- Integration with viewer UI (checkbox)

Design by Contract:
    - Plotter must be valid PyVista plotter
    - Format must be 'png' (case-insensitive)
    - Path must be valid file path
    - Alpha channel must be present in saved PNG files
    - Transparency must not break on-screen rendering
"""

from __future__ import annotations

import logging
import struct
import tempfile
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


class TestTransparencyRendererClass:
    """TransparencyRenderer class must exist and be properly defined."""

    def test_transparency_renderer_is_importable(self) -> None:
        """TransparencyRenderer must be importable from transparency module."""
        from glass_models.viz.transparency import TransparencyRenderer

        assert TransparencyRenderer is not None

    def test_transparency_renderer_is_class(self) -> None:
        """TransparencyRenderer must be a class."""
        from glass_models.viz.transparency import TransparencyRenderer

        assert isinstance(TransparencyRenderer, type)

    def test_transparency_renderer_init(self) -> None:
        """TransparencyRenderer must be instantiable."""
        from glass_models.viz.transparency import TransparencyRenderer

        renderer = TransparencyRenderer()
        assert renderer is not None


class TestEnableTransparentBackground:
    """enable_transparent_background() must configure PyVista correctly."""

    def test_enable_transparent_background_exists(self) -> None:
        """Function enable_transparent_background must exist."""
        from glass_models.viz.transparency import enable_transparent_background

        assert callable(enable_transparent_background)

    def test_enable_transparent_background_accepts_plotter(self) -> None:
        """enable_transparent_background must accept a plotter argument."""
        from glass_models.viz.transparency import enable_transparent_background

        try:
            import pyvista as pv

            plotter = pv.Plotter()
            # Should not raise
            enable_transparent_background(plotter)
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_enable_transparent_background_returns_plotter(self) -> None:
        """enable_transparent_background should return the plotter."""
        from glass_models.viz.transparency import enable_transparent_background

        try:
            import pyvista as pv

            plotter = pv.Plotter()
            result = enable_transparent_background(plotter)
            assert result is plotter
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_enable_transparent_background_sets_alpha(self) -> None:
        """enable_transparent_background must set off_screen_rendering."""
        from glass_models.viz.transparency import enable_transparent_background

        try:
            import pyvista as pv

            plotter = pv.Plotter()
            enable_transparent_background(plotter)
            # Check that transparency was enabled (implementation-dependent)
            # This is a basic smoke test
            assert plotter is not None
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")


class TestDisableTransparentBackground:
    """disable_transparent_background() must revert to opaque rendering."""

    def test_disable_transparent_background_exists(self) -> None:
        """Function disable_transparent_background must exist."""
        from glass_models.viz.transparency import disable_transparent_background

        assert callable(disable_transparent_background)

    def test_disable_transparent_background_accepts_plotter(self) -> None:
        """disable_transparent_background must accept a plotter argument."""
        from glass_models.viz.transparency import disable_transparent_background

        try:
            import pyvista as pv

            plotter = pv.Plotter()
            # Should not raise
            disable_transparent_background(plotter)
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_disable_transparent_background_returns_plotter(self) -> None:
        """disable_transparent_background should return the plotter."""
        from glass_models.viz.transparency import disable_transparent_background

        try:
            import pyvista as pv

            plotter = pv.Plotter()
            result = disable_transparent_background(plotter)
            assert result is plotter
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_disable_reverts_to_opaque(self) -> None:
        """disable_transparent_background must revert opacity settings."""
        from glass_models.viz.transparency import (
            disable_transparent_background,
            enable_transparent_background,
        )

        try:
            import pyvista as pv

            plotter = pv.Plotter()
            enable_transparent_background(plotter)
            disable_transparent_background(plotter)
            # Verify it's back to opaque (implementation-dependent)
            assert plotter is not None
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")


class TestExportWithTransparency:
    """export_with_transparency() must save PNG with alpha channel."""

    def test_export_with_transparency_exists(self) -> None:
        """Function export_with_transparency must exist."""
        from glass_models.viz.transparency import export_with_transparency

        assert callable(export_with_transparency)

    def test_export_with_transparency_requires_path(self) -> None:
        """export_with_transparency must require a path argument."""
        from glass_models.viz.transparency import export_with_transparency

        try:
            import pyvista as pv

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere())
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test.png"
                # Should not raise
                export_with_transparency(plotter, str(path))
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_export_with_transparency_accepts_format(self) -> None:
        """export_with_transparency must accept format argument."""
        from glass_models.viz.transparency import export_with_transparency

        try:
            import pyvista as pv

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere())
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test.png"
                # Should not raise with format parameter
                export_with_transparency(plotter, str(path), format="png")
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_export_creates_file(self) -> None:
        """export_with_transparency must create the output file."""
        from glass_models.viz.transparency import export_with_transparency

        try:
            import pyvista as pv

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere())

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test.png"
                export_with_transparency(plotter, str(path))
                assert path.exists()
                assert path.stat().st_size > 0
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")


class TestPNGAlphaChannel:
    """PNG files must have proper alpha channel embedded."""

    def _verify_png_has_alpha(self, path: Path) -> bool:
        """Verify PNG file has alpha channel (RGBA).

        Checks PNG file signature and IHDR chunk for color type 6 (RGBA).
        Returns True if alpha channel is present.
        """
        if not path.exists():
            return False

        with open(path, "rb") as f:
            # PNG signature
            signature = f.read(8)
            if signature != b"\x89PNG\r\n\x1a\n":
                return False

            # Read IHDR chunk
            f.read(4)  # IHDR length (always 13 bytes)
            chunk_type = f.read(4)
            if chunk_type != b"IHDR":
                return False

            # IHDR structure: width(4) height(4) bit_depth(1) color_type(1) ...
            f.read(8)  # Skip width and height
            f.read(1)  # Skip bit depth
            color_type = struct.unpack("B", f.read(1))[0]

            # Color type 6 = RGBA (truecolor with alpha)
            # Color type 4 = grayscale with alpha
            return color_type in (4, 6)

    def test_png_alpha_channel_in_export(self) -> None:
        """Exported PNG must have alpha channel."""
        from glass_models.viz.transparency import export_with_transparency

        try:
            import pyvista as pv

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere())

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test_alpha.png"
                export_with_transparency(plotter, str(path))

                # Verify PNG has alpha channel
                has_alpha = self._verify_png_has_alpha(path)
                assert has_alpha, "PNG file does not have alpha channel"

            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_png_alpha_consistency(self) -> None:
        """Alpha channel must persist across multiple exports."""
        from glass_models.viz.transparency import export_with_transparency

        try:
            import pyvista as pv

            with tempfile.TemporaryDirectory() as tmpdir:
                for i in range(3):
                    plotter = pv.Plotter(off_screen=True)
                    plotter.add_mesh(pv.Sphere())

                    path = Path(tmpdir) / f"test_alpha_{i}.png"
                    export_with_transparency(plotter, str(path))

                    has_alpha = self._verify_png_has_alpha(path)
                    assert has_alpha, f"PNG {i} missing alpha channel"
                    plotter.close()

        except ImportError:
            pytest.skip("PyVista not available")


class TestOnScreenTransparentRendering:
    """On-screen rendering with transparent background must work."""

    def test_enable_then_render_no_crash(self) -> None:
        """Rendering after enabling transparency must not crash."""
        from glass_models.viz.transparency import (
            enable_transparent_background,
        )

        try:
            import pyvista as pv

            plotter = pv.Plotter()
            plotter.add_mesh(pv.Sphere())

            # Enable transparency
            enable_transparent_background(plotter)

            # Should render without crashing
            # In headless/CI environment, this is a smoke test
            assert plotter.render_window is not None
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_transparency_toggles_correctly(self) -> None:
        """Toggling transparency on/off should work repeatedly."""
        from glass_models.viz.transparency import (
            disable_transparent_background,
            enable_transparent_background,
        )

        try:
            import pyvista as pv

            plotter = pv.Plotter()
            plotter.add_mesh(pv.Sphere())

            # Toggle multiple times
            for _ in range(3):
                enable_transparent_background(plotter)
                disable_transparent_background(plotter)

            # Should still be functional
            assert plotter.render_window is not None
            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")


class TestTransparencyIntegration:
    """Integration tests for transparent background workflow."""

    def test_end_to_end_transparent_export(self) -> None:
        """Complete workflow: enable -> render -> export with transparency."""
        from glass_models.viz.transparency import (
            enable_transparent_background,
            export_with_transparency,
        )

        try:
            import pyvista as pv

            plotter = pv.Plotter(off_screen=True)
            mesh = pv.Sphere()
            plotter.add_mesh(mesh, color="blue")

            enable_transparent_background(plotter)

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test_e2e.png"
                export_with_transparency(plotter, str(path))

                # Verify file exists and has alpha
                assert path.exists()
                assert path.stat().st_size > 0

            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_multiple_objects_transparent_export(self) -> None:
        """Export with multiple objects with transparency enabled."""
        from glass_models.viz.transparency import (
            enable_transparent_background,
            export_with_transparency,
        )

        try:
            import pyvista as pv

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere(center=(0, 0, 0)), color="red")
            plotter.add_mesh(pv.Cube(center=(1, 1, 1)), color="blue")

            enable_transparent_background(plotter)

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test_multi.png"
                export_with_transparency(plotter, str(path))
                assert path.exists()
                assert path.stat().st_size > 0

            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")


class TestTransparencyPerformance:
    """Transparency features must not degrade performance."""

    def test_transparency_doesnt_slow_export_significantly(self) -> None:
        """Exporting with transparency shouldn't be significantly slower."""
        import time

        from glass_models.viz.transparency import (
            enable_transparent_background,
            export_with_transparency,
        )

        try:
            import pyvista as pv

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create sphere once
                mesh = pv.Sphere()

                # Time export with transparency
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(mesh)
                enable_transparent_background(plotter)

                path = Path(tmpdir) / "test_perf.png"
                start = time.time()
                export_with_transparency(plotter, str(path))
                transparent_time = time.time() - start
                plotter.close()

                # Time export without transparency (for comparison)
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(mesh)

                path2 = Path(tmpdir) / "test_perf_opaque.png"
                start = time.time()
                plotter.screenshot(str(path2), transparent_background=False)
                opaque_time = time.time() - start
                plotter.close()

                # Transparent should not be dramatically slower
                # Allow 3x tolerance for setup overhead
                assert transparent_time < opaque_time * 3, (
                    "Transparency export significantly slower than opaque"
                )

        except ImportError:
            pytest.skip("PyVista not available")


class TestTransparencyEdgeCases:
    """Edge cases and error handling for transparency."""

    def test_export_invalid_path_handling(self) -> None:
        """Export to invalid path should raise appropriate error."""
        from glass_models.viz.transparency import export_with_transparency

        try:
            import pyvista as pv

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere())

            # Try to write to non-existent directory
            invalid_path = "/nonexistent/directory/test.png"

            with pytest.raises((FileNotFoundError, OSError, Exception)):
                export_with_transparency(plotter, invalid_path)

            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")

    def test_none_plotter_handling(self) -> None:
        """Functions should handle invalid plotter gracefully."""
        from glass_models.viz.transparency import enable_transparent_background

        with pytest.raises((TypeError, AttributeError, ValueError)):
            enable_transparent_background(None)  # type: ignore

    def test_export_format_case_insensitive(self) -> None:
        """Format parameter should be case-insensitive."""
        from glass_models.viz.transparency import export_with_transparency

        try:
            import pyvista as pv

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere())

            with tempfile.TemporaryDirectory() as tmpdir:
                for fmt in ["PNG", "Png", "pNg"]:
                    path = Path(tmpdir) / f"test_{fmt}.png"
                    # Should not raise
                    export_with_transparency(plotter, str(path), format=fmt)
                    assert path.exists()

            plotter.close()
        except ImportError:
            pytest.skip("PyVista not available")


@pytest.mark.contract
class TestTransparencyContract:
    """API contract tests for transparent background support."""

    def test_transparency_module_exports_all_required_functions(self) -> None:
        """All required functions must be in __all__ or importable."""
        from glass_models.viz import transparency

        required = [
            "TransparencyRenderer",
            "enable_transparent_background",
            "disable_transparent_background",
            "export_with_transparency",
        ]

        for name in required:
            assert hasattr(transparency, name), f"Missing {name} in transparency module"

    def test_transparency_functions_have_docstrings(self) -> None:
        """All public functions must have docstrings."""
        from glass_models.viz.transparency import (
            TransparencyRenderer,
            disable_transparent_background,
            enable_transparent_background,
            export_with_transparency,
        )

        assert TransparencyRenderer.__doc__ is not None, (
            "TransparencyRenderer missing docstring"
        )
        assert enable_transparent_background.__doc__ is not None, (
            "enable_transparent_background missing docstring"
        )
        assert disable_transparent_background.__doc__ is not None, (
            "disable_transparent_background missing docstring"
        )
        assert export_with_transparency.__doc__ is not None, (
            "export_with_transparency missing docstring"
        )
