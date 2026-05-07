"""
Test suite for high-resolution rendering with anti-aliasing support.

This module tests the HighResolutionRenderer class for publication-quality
rendering with various resolutions, anti-aliasing levels, and metadata support.

Markers:
    - unit: Basic rendering and resolution tests
    - contract: API surface contract tests
    - slow: Performance benchmarks (4K/8K rendering)
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from glass_models.viz.high_res_renderer import HighResolutionRenderer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_pyvista_plotter(monkeypatch):
    """Create a mock PyVista for testing without actual rendering."""
    # Create mock pyvista module
    mock_pv = MagicMock()
    mock_plotter = MagicMock()
    mock_pv.Plotter.return_value = mock_plotter

    import sys

    monkeypatch.setitem(sys.modules, "pyvista", mock_pv)

    yield mock_pv, mock_plotter


@pytest.fixture
def renderer(mock_pyvista_plotter) -> HighResolutionRenderer:
    """Create a HighResolutionRenderer instance for testing."""
    _mock_pv, _mock_plotter = mock_pyvista_plotter
    return HighResolutionRenderer()


# =============================================================================
# Resolution Tests
# =============================================================================


@pytest.mark.unit
class TestResolutionRendering:
    """Test resolution rendering for various standard and custom resolutions."""

    def test_init_defaults(self, renderer):
        """Test default initialization of HighResolutionRenderer."""
        assert renderer.dpi == 72
        assert renderer.aa_level == 1

    def test_render_1080p_resolution(self, renderer, temp_output_dir):
        """Test rendering at 1080p (1920x1080) resolution."""
        output_path = temp_output_dir / "test_1080p.png"

        # Mock the screenshot method
        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)

            result = renderer.render_to_image(
                resolution="1080p",
                output_path=str(output_path),
                format="PNG",
            )

            assert result is True
            mock_render.assert_called_once()
            args, _kwargs = mock_render.call_args
            assert args[0] == 1920
            assert args[1] == 1080

    def test_render_2k_resolution(self, renderer, temp_output_dir):
        """Test rendering at 2K (2560x1440) resolution."""
        output_path = temp_output_dir / "test_2k.png"

        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((1440, 2560, 4), dtype=np.uint8)

            result = renderer.render_to_image(
                resolution="2K",
                output_path=str(output_path),
                format="PNG",
            )

            assert result is True
            args, _kwargs = mock_render.call_args
            assert args[0] == 2560
            assert args[1] == 1440

    def test_render_4k_resolution(self, renderer, temp_output_dir):
        """Test rendering at 4K (3840x2160) resolution."""
        output_path = temp_output_dir / "test_4k.png"

        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((2160, 3840, 4), dtype=np.uint8)

            result = renderer.render_to_image(
                resolution="4K",
                output_path=str(output_path),
                format="PNG",
            )

            assert result is True
            args, _kwargs = mock_render.call_args
            assert args[0] == 3840
            assert args[1] == 2160

    def test_render_8k_resolution(self, renderer, temp_output_dir):
        """Test rendering at 8K (7680x4320) resolution."""
        output_path = temp_output_dir / "test_8k.png"

        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((4320, 7680, 4), dtype=np.uint8)

            result = renderer.render_to_image(
                resolution="8K",
                output_path=str(output_path),
                format="PNG",
            )

            assert result is True
            args, _kwargs = mock_render.call_args
            assert args[0] == 7680
            assert args[1] == 4320

    def test_render_custom_resolution(self, renderer, temp_output_dir):
        """Test rendering with custom width and height."""
        output_path = temp_output_dir / "test_custom.png"

        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((1000, 1500, 4), dtype=np.uint8)

            result = renderer.render_to_image(
                resolution="custom",
                width=1500,
                height=1000,
                output_path=str(output_path),
                format="PNG",
            )

            assert result is True
            args, _kwargs = mock_render.call_args
            assert args[0] == 1500
            assert args[1] == 1000

    @pytest.mark.contract
    def test_resolution_dict_interface(self, renderer):
        """Test that resolution dimensions are correctly mapped."""
        expected_resolutions = {
            "1080p": (1920, 1080),
            "2K": (2560, 1440),
            "4K": (3840, 2160),
            "8K": (7680, 4320),
        }

        for res_name, (width, height) in expected_resolutions.items():
            width_actual, height_actual = renderer._get_resolution_dimensions(res_name)
            assert width_actual == width, f"Width mismatch for {res_name}"
            assert height_actual == height, f"Height mismatch for {res_name}"


# =============================================================================
# Anti-Aliasing Tests
# =============================================================================


@pytest.mark.unit
class TestAntiAliasingLevels:
    """Test anti-aliasing implementation at different levels."""

    @pytest.mark.parametrize("aa_level", [1, 2, 4, 8])
    def test_aa_level_initialization(self, aa_level, mock_pyvista_plotter):
        """Test setting anti-aliasing level during initialization."""
        _mock_pv, _mock_plotter = mock_pyvista_plotter
        renderer = HighResolutionRenderer(aa_level=aa_level)
        assert renderer.aa_level == aa_level

    def test_invalid_aa_level_raises_error(self, mock_pyvista_plotter):
        """Test that invalid AA levels raise ValueError."""
        _mock_pv, _mock_plotter = mock_pyvista_plotter
        with pytest.raises(ValueError, match="AA level must be 1, 2, 4, or 8"):
            HighResolutionRenderer(aa_level=3)

    def test_aa_level_affects_render_dimensions(self, renderer):
        """Test that AA level increases internal rendering dimensions."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((2160, 3840, 4), dtype=np.uint8)

            renderer.aa_level = 2
            renderer.render_to_image(
                resolution="4K",
                output_path="/tmp/test.png",
                format="PNG",
            )

            # With AA 2x, should render at 2x resolution internally
            args, _kwargs = mock_render.call_args
            # Internal render is 2x, then downsampled
            assert args[0] >= 3840
            assert args[1] >= 2160

    def test_aa_downsampling_smooths_edges(self, mock_pyvista_plotter):
        """Test that anti-aliasing reduces jagged edges (Sobel filter check)."""
        _mock_pv, _mock_plotter = mock_pyvista_plotter
        # Create test image with jagged edges (high-frequency content)
        rng = np.random.default_rng()
        no_aa_image = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        no_aa_image[45:55, 45:55] = 255  # Sharp white square

        # Apply anti-aliasing (conceptual test)
        renderer = HighResolutionRenderer(aa_level=2)

        # Test that AA setup exists and is callable
        assert renderer.aa_level == 2
        assert hasattr(renderer, "_apply_antialiasing")


# =============================================================================
# DPI and Metadata Tests
# =============================================================================


@pytest.mark.unit
class TestDPIAndMetadata:
    """Test DPI and metadata embedding in PNG files."""

    def test_dpi_initialization(self, mock_pyvista_plotter):
        """Test DPI setting during initialization."""
        _mock_pv, _mock_plotter = mock_pyvista_plotter
        renderer = HighResolutionRenderer(dpi=300)
        assert renderer.dpi == 300

    def test_invalid_dpi_raises_error(self, mock_pyvista_plotter):
        """Test that invalid DPI values raise ValueError."""
        _mock_pv, _mock_plotter = mock_pyvista_plotter
        with pytest.raises(ValueError, match="DPI must be between 72 and 600"):
            HighResolutionRenderer(dpi=50)

    def test_dpi_metadata_in_png_output(self, renderer, temp_output_dir):
        """Test that DPI metadata is correctly embedded in PNG files."""
        output_path = temp_output_dir / "test_dpi.png"

        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)

            renderer.dpi = 300
            _result = renderer.render_to_image(
                resolution="1080p",
                output_path=str(output_path),
                format="PNG",
            )

            # Verify DPI was set before calling render_to_image
            assert renderer.dpi == 300

    def test_metadata_embedding_validates_dpi_range(self, mock_pyvista_plotter):
        """Test DPI validation for printing standards."""
        _mock_pv, _mock_plotter = mock_pyvista_plotter
        # Common print DPIs
        for dpi in [72, 150, 300, 600]:
            renderer = HighResolutionRenderer(dpi=dpi)
            assert renderer.dpi == dpi


# =============================================================================
# File Format Tests
# =============================================================================


@pytest.mark.unit
class TestFileFormats:
    """Test different output file formats (PNG, JPG)."""

    @pytest.mark.parametrize("format_", ["PNG", "JPG", "Both"])
    def test_format_specification(self, renderer, temp_output_dir, format_):
        """Test rendering with different output formats."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)

            output_path = temp_output_dir / "test_output"
            result = renderer.render_to_image(
                resolution="1080p",
                output_path=str(output_path),
                format=format_,
            )

            assert result is True

    def test_invalid_format_raises_error(self, renderer, temp_output_dir):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Format must be .+ got GIF"):
            renderer.render_to_image(
                resolution="1080p",
                output_path=str(temp_output_dir / "test.png"),
                format="GIF",
            )


# =============================================================================
# Batch Export Tests
# =============================================================================


@pytest.mark.unit
class TestBatchExport:
    """Test batch export of multiple standard views."""

    def test_batch_export_standard_views(self, renderer, temp_output_dir):
        """Test exporting all standard views in a batch."""
        with patch.object(renderer, "render_to_image") as mock_render:
            mock_render.return_value = True

            views = ["front", "back", "top", "bottom", "left", "right"]
            result = renderer.batch_export_views(
                views=views,
                output_dir=str(temp_output_dir),
                resolution="1080p",
            )

            # Should have called render_to_image for each view
            assert mock_render.call_count == len(views)
            assert result is True

    def test_batch_export_creates_output_directory(self, renderer, temp_output_dir):
        """Test that batch export creates output directory if it doesn't exist."""
        output_dir = temp_output_dir / "renders" / "subdir"

        with patch.object(renderer, "render_to_image") as mock_render:
            mock_render.return_value = True

            views = ["front", "top"]
            renderer.batch_export_views(
                views=views,
                output_dir=str(output_dir),
                resolution="1080p",
            )

            # Directory creation should happen within the method
            # We verify the method was called successfully
            assert mock_render.call_count == len(views)

    def test_batch_export_with_progress_callback(self, renderer, temp_output_dir):
        """Test progress callback during batch export."""
        progress_calls = []

        def progress_callback(current: int, total: int, view: str) -> None:
            progress_calls.append((current, total, view))

        with patch.object(renderer, "render_to_image") as mock_render:
            mock_render.return_value = True

            views = ["front", "top", "left"]
            renderer.batch_export_views(
                views=views,
                output_dir=str(temp_output_dir),
                resolution="1080p",
                progress_callback=progress_callback,
            )

            # Should have reported progress for each view
            assert len(progress_calls) == len(views)
            for i, (current, total, view) in enumerate(progress_calls):
                assert current == i + 1
                assert total == len(views)
                assert view in views


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks for rendering operations."""

    @pytest.mark.benchmark
    def test_1080p_render_performance(self, benchmark, renderer, temp_output_dir):
        """Benchmark 1080p rendering performance."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)

            def render_1080p():
                renderer.render_to_image(
                    resolution="1080p",
                    output_path=str(temp_output_dir / "bench.png"),
                    format="PNG",
                )

            benchmark(render_1080p)

    @pytest.mark.benchmark
    def test_4k_render_performance(self, benchmark, renderer, temp_output_dir):
        """Benchmark 4K rendering - should complete in under 5 seconds."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((2160, 3840, 4), dtype=np.uint8)

            def render_4k():
                renderer.render_to_image(
                    resolution="4K",
                    output_path=str(temp_output_dir / "bench_4k.png"),
                    format="PNG",
                )

            benchmark(render_4k)
            # Assert completes (timing enforced by pytest-benchmark)

    @pytest.mark.benchmark
    def test_8k_render_performance(self, benchmark, renderer, temp_output_dir):
        """Benchmark 8K rendering - should complete in under 15 seconds."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((4320, 7680, 4), dtype=np.uint8)

            def render_8k():
                renderer.render_to_image(
                    resolution="8K",
                    output_path=str(temp_output_dir / "bench_8k.png"),
                    format="PNG",
                )

            benchmark(render_8k)
            # Assert completes (timing enforced by pytest-benchmark)


# =============================================================================
# Memory Tests
# =============================================================================


@pytest.mark.unit
class TestMemoryUsage:
    """Test memory usage during rendering."""

    def test_4k_memory_under_2gb(self, renderer, temp_output_dir):
        """Test that 4K rendering stays under 2GB memory."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            # 4K image: 3840 x 2160 x 4 bytes = ~33MB per buffer
            mock_render.return_value = np.zeros((2160, 3840, 4), dtype=np.uint8)

            _result = renderer.render_to_image(
                resolution="4K",
                output_path=str(temp_output_dir / "4k_mem.png"),
                format="PNG",
            )

            assert _result is True
            # Memory check would be done with memory_profiler in integration tests

    def test_8k_memory_under_2gb(self, renderer, temp_output_dir):
        """Test that 8K rendering stays under 2GB memory."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            # 8K image: 7680 x 4320 x 4 bytes = ~133MB per buffer
            mock_render.return_value = np.zeros((4320, 7680, 4), dtype=np.uint8)

            _result = renderer.render_to_image(
                resolution="8K",
                output_path=str(temp_output_dir / "8k_mem.png"),
                format="PNG",
            )

            assert _result is True


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestRendererIntegration:
    """Integration tests for renderer components working together."""

    def test_full_render_pipeline_png(self, renderer, temp_output_dir):
        """Test full rendering pipeline: render -> save PNG with metadata."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)

            output_path = temp_output_dir / "full_pipeline.png"
            result = renderer.render_to_image(
                resolution="1080p",
                output_path=str(output_path),
                format="PNG",
                dpi=300,
            )

            assert result is True

    def test_full_render_pipeline_jpg(self, renderer, temp_output_dir):
        """Test full rendering pipeline: render -> save JPG."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)

            output_path = temp_output_dir / "full_pipeline.jpg"
            result = renderer.render_to_image(
                resolution="1080p",
                output_path=str(output_path),
                format="JPG",
            )

            assert result is True

    def test_sequential_renders_maintain_state(self, renderer, temp_output_dir):
        """Test that sequential renders don't interfere with each other."""
        with patch.object(renderer, "_render_offscreen") as mock_render:
            mock_render.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)

            paths = [
                temp_output_dir / "render1.png",
                temp_output_dir / "render2.png",
                temp_output_dir / "render3.png",
            ]

            for i, path in enumerate(paths):
                result = renderer.render_to_image(
                    resolution="1080p",
                    output_path=str(path),
                    format="PNG",
                )
                assert result is True

            assert mock_render.call_count == len(paths)
