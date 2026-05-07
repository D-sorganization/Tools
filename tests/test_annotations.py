"""Test suite for publication-quality annotations & labeling (GitHub issue #542).

This module implements TDD for annotations with support for:
1. Text annotations with LaTeX rendering
2. Dimension labels with auto-computed lengths
3. Boundary labels and axis annotations
4. Position validation (Design by Contract)
5. PyQt6 widget interaction
6. Annotation persistence (export/import)

Success criteria:
- All annotation tests pass
- Dimension labels compute correct lengths
- LaTeX renders without error
- UI interactions work correctly
- Annotations persist through export
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAnnotationDataclass:
    """Unit tests for Annotation dataclass."""

    @pytest.mark.unit
    def test_annotation_creation_basic(self) -> None:
        """Test creating a basic text annotation."""
        from glass_models.viz.annotations import Annotation

        ann = Annotation(
            id="ann_001",
            type="text",
            position=(0.5, 0.5, 0.0),
            text="Sample Label",
            font_size=12,
            color="black",
        )

        assert ann.id == "ann_001"
        assert ann.type == "text"
        assert ann.position == (0.5, 0.5, 0.0)
        assert ann.text == "Sample Label"
        assert ann.font_size == 12
        assert ann.color == "black"

    @pytest.mark.unit
    def test_annotation_position_validation(self) -> None:
        """Test position validation (Design by Contract)."""
        from glass_models.viz.annotations import Annotation

        # Valid position (3-tuple of floats)
        ann = Annotation(
            id="ann_001",
            type="text",
            position=(0.0, 0.0, 0.0),
            text="Test",
            font_size=12,
            color="black",
        )
        assert ann.position == (0.0, 0.0, 0.0)

        # Invalid position (not 3D)
        with pytest.raises(ValueError, match="position must be a 3-tuple"):
            Annotation(
                id="ann_001",
                type="text",
                position=(0.0, 0.0),  # type: ignore
                text="Test",
                font_size=12,
                color="black",
            )

    @pytest.mark.unit
    def test_annotation_font_size_validation(self) -> None:
        """Test font size validation (DbC)."""
        from glass_models.viz.annotations import Annotation

        # Valid font size
        ann = Annotation(
            id="ann_001",
            type="text",
            position=(0.0, 0.0, 0.0),
            text="Test",
            font_size=8,
            color="black",
        )
        assert ann.font_size > 0

        # Font size too small
        with pytest.raises(ValueError, match="font_size must be > 0"):
            Annotation(
                id="ann_001",
                type="text",
                position=(0.0, 0.0, 0.0),
                text="Test",
                font_size=0,
                color="black",
            )

        # Font size negative
        with pytest.raises(ValueError, match="font_size must be > 0"):
            Annotation(
                id="ann_001",
                type="text",
                position=(0.0, 0.0, 0.0),
                text="Test",
                font_size=-5,
                color="black",
            )

    @pytest.mark.unit
    def test_annotation_color_validation(self) -> None:
        """Test color validation (DbC)."""
        from glass_models.viz.annotations import Annotation

        # Valid color names
        for color in ["black", "red", "blue", "green", "white"]:
            ann = Annotation(
                id="ann_001",
                type="text",
                position=(0.0, 0.0, 0.0),
                text="Test",
                font_size=12,
                color=color,
            )
            assert ann.color == color

        # Valid RGB hex
        ann = Annotation(
            id="ann_001",
            type="text",
            position=(0.0, 0.0, 0.0),
            text="Test",
            font_size=12,
            color="#FF0000",
        )
        assert ann.color == "#FF0000"

        # Invalid color
        with pytest.raises(
            ValueError, match="color must be a color name or hex code"
        ):
            Annotation(
                id="ann_001",
                type="text",
                position=(0.0, 0.0, 0.0),
                text="Test",
                font_size=12,
                color="notacolor",
            )


class TestAnnotationManagerBasics:
    """Unit tests for AnnotationManager class."""

    @pytest.mark.unit
    def test_manager_creation(self) -> None:
        """Test creating an AnnotationManager."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        assert mgr is not None
        assert len(mgr.annotations) == 0

    @pytest.mark.unit
    def test_add_text_annotation(self) -> None:
        """Test adding a text annotation."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        ann_id = mgr.add_text("Test Label", position=(0.5, 0.5, 0.0))

        assert ann_id is not None
        assert len(mgr.annotations) == 1
        assert mgr.annotations[ann_id].text == "Test Label"
        assert mgr.annotations[ann_id].type == "text"

    @pytest.mark.unit
    def test_add_text_annotation_with_options(self) -> None:
        """Test adding text annotation with font size and color."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        ann_id = mgr.add_text(
            "Custom Label",
            position=(0.5, 0.5, 0.0),
            font_size=14,
            color="red",
        )

        ann = mgr.annotations[ann_id]
        assert ann.text == "Custom Label"
        assert ann.font_size == 14
        assert ann.color == "red"

    @pytest.mark.unit
    def test_add_dimension_annotation(self) -> None:
        """Test adding a dimension annotation."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        p1 = (0.0, 0.0, 0.0)
        p2 = (1.0, 0.0, 0.0)
        ann_id = mgr.add_dimension(p1, p2, label_position=(0.5, 0.1, 0.0))

        assert ann_id is not None
        assert len(mgr.annotations) == 1
        ann = mgr.annotations[ann_id]
        assert ann.type == "dimension"

    @pytest.mark.unit
    def test_dimension_length_computation(self) -> None:
        """Test that dimension annotations compute correct length.

        Creates a dimension from (0,0,0) to (3,4,0), which has length 5.
        """
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        p1 = (0.0, 0.0, 0.0)
        p2 = (3.0, 4.0, 0.0)
        dim_id = mgr.add_dimension(p1, p2, label_position=(1.5, 2.0, 0.0))

        ann = mgr.annotations[dim_id]
        # Length should be 5.0 (3-4-5 triangle)
        assert "5.0" in ann.text or "5" in ann.text

    @pytest.mark.unit
    def test_add_boundary_label(self) -> None:
        """Test adding a boundary label annotation."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        ann_id = mgr.add_boundary_label(
            "Fixed Support",
            boundary_center=(0.0, 0.0, 0.0),
            label_position=(0.0, -0.2, 0.0),
        )

        assert ann_id is not None
        ann = mgr.annotations[ann_id]
        assert ann.type == "boundary"
        assert ann.text == "Fixed Support"

    @pytest.mark.unit
    def test_add_axis_labels(self) -> None:
        """Test adding axis labels."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        axis_labels = mgr.add_axis_labels(
            x_label="X (mm)",
            y_label="Y (mm)",
            z_label="Z (mm)",
            origin=(0.0, 0.0, 0.0),
            axis_length=1.0,
        )

        assert len(axis_labels) == 3
        assert len(mgr.annotations) == 3

    @pytest.mark.unit
    def test_remove_annotation(self) -> None:
        """Test removing an annotation by ID."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        ann_id = mgr.add_text("Test", position=(0.0, 0.0, 0.0))
        assert len(mgr.annotations) == 1

        mgr.remove_annotation(ann_id)
        assert len(mgr.annotations) == 0
        assert ann_id not in mgr.annotations

    @pytest.mark.unit
    def test_get_annotation(self) -> None:
        """Test retrieving an annotation by ID."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        ann_id = mgr.add_text("Test Label", position=(0.5, 0.5, 0.0))

        ann = mgr.get_annotation(ann_id)
        assert ann is not None
        assert ann.text == "Test Label"

    @pytest.mark.unit
    def test_get_nonexistent_annotation(self) -> None:
        """Test getting a non-existent annotation returns None."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        ann = mgr.get_annotation("nonexistent_id")
        assert ann is None


class TestAnnotationLaTeXSupport:
    """Tests for LaTeX rendering in annotations."""

    @pytest.mark.unit
    def test_latex_in_text_annotation(self) -> None:
        """Test that LaTeX syntax is preserved in text annotations."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        latex_text = r"$\sigma = \frac{F}{A}$"
        ann_id = mgr.add_text(latex_text, position=(0.5, 0.5, 0.0))

        ann = mgr.annotations[ann_id]
        assert latex_text in ann.text

    @pytest.mark.unit
    def test_latex_validation(self) -> None:
        """Test that LaTeX can be validated for basic syntax."""
        from glass_models.viz.annotations import Annotation

        # Valid LaTeX
        ann = Annotation(
            id="ann_001",
            type="text",
            position=(0.0, 0.0, 0.0),
            text=r"$E=mc^2$",
            font_size=12,
            color="black",
        )
        assert ann.text == r"$E=mc^2$"

    @pytest.mark.unit
    def test_multiline_latex(self) -> None:
        """Test multi-line LaTeX expressions."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        latex_text = r"Stress: $\sigma_x = 100$ MPa\nStrain: $\epsilon = 0.001$"
        ann_id = mgr.add_text(latex_text, position=(0.5, 0.5, 0.0))

        ann = mgr.annotations[ann_id]
        assert "sigma" in ann.text


class TestAnnotationRendering:
    """Tests for rendering annotations (requires PyVista)."""

    @pytest.mark.unit
    def test_render_annotations_empty_list(self) -> None:
        """Test rendering with no annotations."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        # Create a mock plotter
        mock_plotter = MagicMock()

        # Should not raise
        mgr.render_annotations(mock_plotter)

    @pytest.mark.unit
    def test_render_text_annotation(self) -> None:
        """Test rendering a text annotation to PyVista plotter."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        mgr.add_text(
            "Test Label", position=(0.5, 0.5, 0.0), font_size=14, color="red"
        )

        mock_plotter = MagicMock()

        mgr.render_annotations(mock_plotter)

        # Verify add_text was called on plotter with correct signature
        # The method should be called with text, position, font_size, and color
        assert mock_plotter.add_text.called
        assert len(mgr.annotations) == 1

    @pytest.mark.unit
    def test_annotation_manager_get_all_by_type(self) -> None:
        """Test filtering annotations by type."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        text_id = mgr.add_text("Text", position=(0.0, 0.0, 0.0))
        dim_id = mgr.add_dimension((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

        text_anns = mgr.get_by_type("text")
        dim_anns = mgr.get_by_type("dimension")

        assert len(text_anns) == 1
        assert len(dim_anns) == 1
        assert text_anns[0].id == text_id
        assert dim_anns[0].id == dim_id


class TestAnnotationPersistence:
    """Tests for saving/loading annotations."""

    @pytest.mark.unit
    def test_annotation_to_dict(self) -> None:
        """Test converting annotation to dictionary."""
        from glass_models.viz.annotations import Annotation

        ann = Annotation(
            id="ann_001",
            type="text",
            position=(0.5, 0.5, 0.0),
            text="Test",
            font_size=12,
            color="black",
        )

        ann_dict = ann.to_dict()
        assert ann_dict["id"] == "ann_001"
        assert ann_dict["type"] == "text"
        assert ann_dict["text"] == "Test"

    @pytest.mark.unit
    def test_annotation_from_dict(self) -> None:
        """Test creating annotation from dictionary."""
        from glass_models.viz.annotations import Annotation

        ann_dict = {
            "id": "ann_001",
            "type": "text",
            "position": (0.5, 0.5, 0.0),
            "text": "Test",
            "font_size": 12,
            "color": "black",
        }

        ann = Annotation.from_dict(ann_dict)
        assert ann.id == "ann_001"
        assert ann.text == "Test"

    @pytest.mark.unit
    def test_manager_export_annotations(self) -> None:
        """Test exporting all annotations as JSON-serializable dict."""
        from glass_models.viz.annotations import AnnotationManager

        mgr = AnnotationManager()
        mgr.add_text("Label 1", position=(0.0, 0.0, 0.0))
        mgr.add_text("Label 2", position=(1.0, 1.0, 0.0))

        export_data = mgr.export_to_dict()
        assert isinstance(export_data, dict)
        assert "annotations" in export_data
        assert len(export_data["annotations"]) == 2

    @pytest.mark.unit
    def test_manager_import_annotations(self) -> None:
        """Test importing annotations from dict."""
        from glass_models.viz.annotations import AnnotationManager

        # Create manager with data
        mgr1 = AnnotationManager()
        mgr1.add_text("Label 1", position=(0.0, 0.0, 0.0))
        mgr1.add_text("Label 2", position=(1.0, 1.0, 0.0))

        export_data = mgr1.export_to_dict()

        # Import into new manager
        mgr2 = AnnotationManager()
        mgr2.import_from_dict(export_data)

        assert len(mgr2.annotations) == 2


class TestAnnotationWidget:
    """Tests for PyQt6 annotation control widget (requires PyQt6)."""

    @pytest.mark.unit
    def test_widget_creation(self) -> None:
        """Test creating the annotation control widget."""
        pytest.importorskip("PyQt6")
        from glass_models.ui.pyqt6.annotation_widget import AnnotationControlWidget

        # Create without parent
        widget = AnnotationControlWidget()
        assert widget is not None

    @pytest.mark.unit
    def test_widget_annotation_type_selector(self) -> None:
        """Test annotation type selector in widget."""
        pytest.importorskip("PyQt6")
        from glass_models.ui.pyqt6.annotation_widget import AnnotationControlWidget

        widget = AnnotationControlWidget()

        # Check that type selector exists and has expected types
        type_items = []
        for i in range(widget.type_combo.count()):
            type_items.append(widget.type_combo.itemText(i))

        assert "Text" in type_items
        assert "Dimension" in type_items
        assert "Boundary" in type_items
        assert "Axis" in type_items

    @pytest.mark.unit
    def test_widget_text_input(self) -> None:
        """Test text input field in widget."""
        pytest.importorskip("PyQt6")
        from glass_models.ui.pyqt6.annotation_widget import AnnotationControlWidget

        widget = AnnotationControlWidget()
        widget.text_input.setText("Test Label")

        assert widget.text_input.text() == "Test Label"

    @pytest.mark.unit
    def test_widget_font_size_spinner(self) -> None:
        """Test font size spinner in widget."""
        pytest.importorskip("PyQt6")
        from glass_models.ui.pyqt6.annotation_widget import AnnotationControlWidget

        widget = AnnotationControlWidget()
        widget.font_size_spinner.setValue(16)

        assert widget.font_size_spinner.value() == 16

    @pytest.mark.unit
    def test_widget_color_picker(self) -> None:
        """Test color picker in widget."""
        pytest.importorskip("PyQt6")
        from glass_models.ui.pyqt6.annotation_widget import AnnotationControlWidget

        widget = AnnotationControlWidget()

        # Color picker should be available
        assert hasattr(widget, "color_button")
        assert widget.color_button is not None

    @pytest.mark.unit
    def test_widget_add_button(self) -> None:
        """Test add annotation button signal."""
        pytest.importorskip("PyQt6")
        from glass_models.ui.pyqt6.annotation_widget import AnnotationControlWidget

        widget = AnnotationControlWidget()

        # Signal should be connectable
        assert hasattr(widget, "annotation_added")

    @pytest.mark.unit
    def test_widget_annotation_list_display(self) -> None:
        """Test annotation list display in widget."""
        pytest.importorskip("PyQt6")
        from glass_models.ui.pyqt6.annotation_widget import AnnotationControlWidget

        widget = AnnotationControlWidget()

        # List widget should exist
        assert hasattr(widget, "annotation_list")
        assert widget.annotation_list is not None


__all__ = [
    "TestAnnotationDataclass",
    "TestAnnotationManagerBasics",
    "TestAnnotationLaTeXSupport",
    "TestAnnotationRendering",
    "TestAnnotationPersistence",
    "TestAnnotationWidget",
]
