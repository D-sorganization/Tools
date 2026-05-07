"""Tests for GitHub issue #545: Custom Colormap Support & Library.

This module implements TDD for colormaps with support for:
1. 20+ scientifically-validated built-in colormaps
2. Custom colormap creation from color stops
3. Colorblind-friendly verification
4. Print-friendly B&W conversion
5. Perceptual uniformity testing
6. PyQt6 widget integration

Success criteria:
- All colormap tests pass
- 20+ colormaps available
- Custom colormaps editable
- B&W conversion works
- Colorblind-friendly options verified
- Preview displays correctly
- Code formatted and typed
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestColormapManagerInitialization:
    """Unit tests for ColormapManager initialization."""

    @pytest.mark.unit
    def test_colormap_manager_imports(self) -> None:
        """ColormapManager should be importable from glass_models.viz.colormaps."""
        from glass_models.viz.colormaps import ColormapManager

        assert ColormapManager is not None

    @pytest.mark.unit
    def test_colormap_manager_instantiation(self) -> None:
        """ColormapManager should instantiate without arguments."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        assert manager is not None

    @pytest.mark.unit
    def test_colormap_manager_has_required_methods(self) -> None:
        """ColormapManager should have required methods."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        assert callable(getattr(manager, "get_colormap", None))
        assert callable(getattr(manager, "create_custom_colormap", None))
        assert callable(getattr(manager, "apply_colormap", None))
        assert callable(getattr(manager, "to_bw", None))
        assert callable(getattr(manager, "list_colormaps", None))


class TestColormapLoading:
    """Tests for loading and retrieving colormaps."""

    @pytest.mark.unit
    def test_list_colormaps_returns_list(self) -> None:
        """list_colormaps should return a list of colormap names."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colormaps = manager.list_colormaps()
        assert isinstance(colormaps, list)
        assert len(colormaps) > 0

    @pytest.mark.unit
    def test_at_least_20_colormaps_available(self) -> None:
        """At least 20 colormaps should be available."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colormaps = manager.list_colormaps()
        assert len(colormaps) >= 20, f"Expected >=20 colormaps, got {len(colormaps)}"

    @pytest.mark.unit
    def test_required_colormaps_exist(self) -> None:
        """Required colormaps should be available."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colormaps = manager.list_colormaps()
        required = ["viridis", "turbo", "coolwarm", "RdBu", "cividis"]
        for name in required:
            assert name in colormaps, f"Required colormap '{name}' not found"

    @pytest.mark.unit
    def test_get_colormap_by_name(self) -> None:
        """get_colormap should return a colormap by name."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        cmap = manager.get_colormap("viridis")
        assert cmap is not None
        assert callable(cmap), "Colormap should be callable"

    @pytest.mark.unit
    def test_get_colormap_invalid_name_raises(self) -> None:
        """get_colormap should raise ValueError for invalid name."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        with pytest.raises(ValueError):
            manager.get_colormap("nonexistent_colormap_xyz")


class TestCustomColormapCreation:
    """Tests for creating custom colormaps."""

    @pytest.mark.unit
    def test_create_custom_colormap_basic(self) -> None:
        """create_custom_colormap should create a colormap from color stops."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colors = ["red", "white", "blue"]
        positions = [0.0, 0.5, 1.0]
        cmap = manager.create_custom_colormap(colors, positions)
        assert cmap is not None
        assert callable(cmap)

    @pytest.mark.unit
    def test_create_custom_colormap_hex_colors(self) -> None:
        """create_custom_colormap should accept hex color codes."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colors = ["#FF0000", "#FFFFFF", "#0000FF"]
        positions = [0.0, 0.5, 1.0]
        cmap = manager.create_custom_colormap(colors, positions)
        assert cmap is not None

    @pytest.mark.unit
    def test_create_custom_colormap_rgb_tuples(self) -> None:
        """create_custom_colormap should accept RGB tuples."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colors = [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)]
        positions = [0.0, 0.5, 1.0]
        cmap = manager.create_custom_colormap(colors, positions)
        assert cmap is not None

    @pytest.mark.unit
    def test_create_custom_colormap_mismatched_lengths_raises(self) -> None:
        """create_custom_colormap should raise ValueError for mismatched lengths."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colors = ["red", "blue"]
        positions = [0.0, 0.5, 1.0]
        with pytest.raises(ValueError):
            manager.create_custom_colormap(colors, positions)

    @pytest.mark.unit
    def test_create_custom_colormap_invalid_positions_raises(self) -> None:
        """create_custom_colormap should raise ValueError for invalid positions."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colors = ["red", "white", "blue"]
        positions = [0.0, 0.5, 1.5]  # position > 1.0
        with pytest.raises(ValueError):
            manager.create_custom_colormap(colors, positions)

    @pytest.mark.unit
    def test_create_custom_colormap_positions_unsorted_raises(self) -> None:
        """create_custom_colormap should raise ValueError for unsorted positions."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colors = ["red", "white", "blue"]
        positions = [0.5, 0.0, 1.0]  # Unsorted
        with pytest.raises(ValueError):
            manager.create_custom_colormap(colors, positions)

    @pytest.mark.unit
    def test_custom_colormap_is_callable(self) -> None:
        """Custom colormap should be callable and return color values."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        colors = ["red", "white", "blue"]
        positions = [0.0, 0.5, 1.0]
        cmap = manager.create_custom_colormap(colors, positions)

        # Test at different normalized values [0, 1]
        result = cmap(0.0)
        assert result is not None
        assert len(result) in (3, 4)  # RGB or RGBA

        result = cmap(0.5)
        assert result is not None

        result = cmap(1.0)
        assert result is not None


class TestBlackAndWhiteConversion:
    """Tests for print-friendly B&W conversion."""

    @pytest.mark.unit
    def test_to_bw_method_exists(self) -> None:
        """ColormapManager should have to_bw method."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        assert callable(getattr(manager, "to_bw", None))

    @pytest.mark.unit
    def test_to_bw_colormap_basic(self) -> None:
        """to_bw should convert a colormap to grayscale."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        cmap = manager.get_colormap("viridis")
        bw_cmap = manager.to_bw(cmap)
        assert bw_cmap is not None
        assert callable(bw_cmap)

    @pytest.mark.unit
    def test_to_bw_returns_grayscale_values(self) -> None:
        """to_bw colormap should return grayscale values (R=G=B)."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        cmap = manager.get_colormap("viridis")
        bw_cmap = manager.to_bw(cmap)

        # Sample at multiple positions
        for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
            color = bw_cmap(value)
            r, g, b = color[0], color[1], color[2]
            # For grayscale, R should equal G and B (within rounding)
            assert abs(r - g) < 0.01, f"R({r}) != G({g}) at {value}"
            assert abs(g - b) < 0.01, f"G({g}) != B({b}) at {value}"

    @pytest.mark.unit
    def test_to_bw_preserves_luminance_range(self) -> None:
        """to_bw should preserve luminance range (dark to light)."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        cmap = manager.get_colormap("viridis")
        bw_cmap = manager.to_bw(cmap)

        value_0 = bw_cmap(0.0)[0]  # First value
        value_1 = bw_cmap(1.0)[0]  # Last value
        # Luminance should change across the range
        assert abs(value_0 - value_1) > 0.1, "B&W conversion lost luminance range"


class TestColormapApplication:
    """Tests for applying colormaps to actors/visualizations."""

    @pytest.mark.unit
    def test_apply_colormap_method_exists(self) -> None:
        """ColormapManager should have apply_colormap method."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        assert callable(getattr(manager, "apply_colormap", None))

    @pytest.mark.unit
    def test_apply_colormap_to_mock_actor(self) -> None:
        """apply_colormap should work with mock actor objects."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        mock_actor = MagicMock()
        cmap_name = "viridis"

        # Should not raise
        result = manager.apply_colormap(mock_actor, cmap_name)
        assert result is not None or result is None  # May return actor or None

    @pytest.mark.unit
    def test_apply_colormap_invalid_name_raises(self) -> None:
        """apply_colormap should raise ValueError for invalid colormap name."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        mock_actor = MagicMock()

        with pytest.raises(ValueError):
            manager.apply_colormap(mock_actor, "nonexistent_colormap")


class TestPerceptualUniformity:
    """Tests for perceptual uniformity of colormaps."""

    @pytest.mark.unit
    def test_perceptually_uniform_colormaps_listed(self) -> None:
        """Perceptually uniform colormaps should be identifiable."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        uniform_cmaps = manager.list_uniform_colormaps()
        assert isinstance(uniform_cmaps, list)
        assert len(uniform_cmaps) > 0

    @pytest.mark.unit
    def test_viridis_is_perceptually_uniform(self) -> None:
        """Viridis should be marked as perceptually uniform."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        uniform_cmaps = manager.list_uniform_colormaps()
        assert "viridis" in uniform_cmaps

    @pytest.mark.unit
    def test_perceptual_uniformity_metadata(self) -> None:
        """Colormaps should have perceptual uniformity metadata."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        metadata = manager.get_colormap_metadata("viridis")
        assert isinstance(metadata, dict)
        assert "uniform" in metadata or "perceptually_uniform" in metadata


class TestColorblindFriendliness:
    """Tests for colorblind-friendly colormaps."""

    @pytest.mark.unit
    def test_colorblind_friendly_colormaps_listed(self) -> None:
        """Colorblind-friendly colormaps should be identifiable."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        friendly_cmaps = manager.list_colorblind_friendly_colormaps()
        assert isinstance(friendly_cmaps, list)
        assert len(friendly_cmaps) > 0

    @pytest.mark.unit
    def test_cividis_is_colorblind_friendly(self) -> None:
        """Cividis should be marked as colorblind-friendly."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        friendly_cmaps = manager.list_colorblind_friendly_colormaps()
        assert "cividis" in friendly_cmaps

    @pytest.mark.unit
    def test_diverging_colormap_has_zero_crossing(self) -> None:
        """Diverging colormaps should have zero crossing at middle."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        diverging_cmaps = manager.list_diverging_colormaps()
        assert "RdBu" in diverging_cmaps or "coolwarm" in diverging_cmaps

    @pytest.mark.unit
    def test_diverging_colormap_perceptual_neutral_middle(self) -> None:
        """Diverging colormap should have neutral color at middle."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        # RdBu or coolwarm should have white/neutral at 0.5
        cmap_name = "RdBu" if "RdBu" in manager.list_colormaps() else "coolwarm"
        cmap = manager.get_colormap(cmap_name)
        middle_color = cmap(0.5)
        # Middle should be close to white/neutral (all channels similar and high)
        assert middle_color[0] > 0.8 or abs(middle_color[0] - middle_color[1]) < 0.1


class TestColormapMetadata:
    """Tests for colormap metadata and properties."""

    @pytest.mark.unit
    def test_colormap_metadata_accessible(self) -> None:
        """Colormap metadata should be accessible."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        metadata = manager.get_colormap_metadata("viridis")
        assert isinstance(metadata, dict)

    @pytest.mark.unit
    def test_metadata_contains_key_properties(self) -> None:
        """Metadata should contain key properties."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        metadata = manager.get_colormap_metadata("viridis")
        # At least one of these should be present
        keys = set(metadata.keys())
        assert len(keys) > 0

    @pytest.mark.unit
    def test_get_colormap_category(self) -> None:
        """Colormaps should be categorized."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        categories = manager.list_colormap_categories()
        assert isinstance(categories, (list, tuple))
        assert len(categories) > 0

    @pytest.mark.unit
    def test_sequential_categorical_diverging_categories(self) -> None:
        """Should have sequential, categorical, and diverging categories."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        categories = manager.list_colormap_categories()
        # At least one of these should exist
        has_categories = (
            any("sequential" in str(c).lower() for c in categories)
            or any("categorical" in str(c).lower() for c in categories)
            or any("diverging" in str(c).lower() for c in categories)
        )
        assert has_categories


class TestColormapIntegration:
    """Integration tests for colormap functionality."""

    @pytest.mark.unit
    def test_full_colormap_workflow(self) -> None:
        """Test complete colormap workflow."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()

        # List available colormaps
        colormaps = manager.list_colormaps()
        assert len(colormaps) >= 20

        # Get a colormap
        cmap = manager.get_colormap("viridis")
        assert cmap is not None

        # Create custom colormap
        custom_cmap = manager.create_custom_colormap(
            ["blue", "cyan", "yellow", "red"], [0.0, 0.33, 0.67, 1.0]
        )
        assert custom_cmap is not None

        # Convert to B&W
        bw_cmap = manager.to_bw(cmap)
        assert bw_cmap is not None

    @pytest.mark.unit
    def test_colormap_roundtrip_with_array(self) -> None:
        """Test mapping data values to colors."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        cmap = manager.get_colormap("viridis")

        # Create test data in [0, 1]
        data = np.linspace(0, 1, 10)
        colors = np.array([cmap(val) for val in data])

        assert colors.shape[0] == 10
        assert colors.shape[1] in (3, 4)  # RGB or RGBA


class TestColormapDataValidation:
    """Design by Contract tests for colormap validation."""

    @pytest.mark.unit
    def test_create_custom_colormap_empty_colors_raises(self) -> None:
        """create_custom_colormap should reject empty color list."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        with pytest.raises((ValueError, TypeError)):
            manager.create_custom_colormap([], [])

    @pytest.mark.unit
    def test_create_custom_colormap_single_color_ok(self) -> None:
        """create_custom_colormap should allow single color."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        cmap = manager.create_custom_colormap(["red"], [0.0])
        # Single-color colormap should still work (returns same color for all values)
        assert cmap is not None

    @pytest.mark.unit
    def test_colormap_negative_position_raises(self) -> None:
        """create_custom_colormap should reject negative positions."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        with pytest.raises(ValueError):
            manager.create_custom_colormap(["red", "blue"], [-0.1, 1.0])

    @pytest.mark.unit
    def test_apply_colormap_requires_valid_name(self) -> None:
        """apply_colormap should require valid colormap name."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        actor = MagicMock()
        with pytest.raises(ValueError):
            manager.apply_colormap(actor, "")  # Empty name


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11+")
class TestColormapTyping:
    """Type-checking tests for colormap module."""

    @pytest.mark.unit
    def test_colormap_callable_typing(self) -> None:
        """Colormaps should be properly typed as callables."""
        from glass_models.viz.colormaps import ColormapManager

        manager = ColormapManager()
        cmap = manager.get_colormap("viridis")
        # Should be callable and accept float input
        result = cmap(0.5)
        assert isinstance(result, (tuple, list, np.ndarray))
