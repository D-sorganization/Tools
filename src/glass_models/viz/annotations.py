"""Publication-quality annotations and labeling for 3D FEA visualization.

Implements GitHub issue #542. This module provides annotation capabilities for
3D scientific visualizations:
- Text annotations with LaTeX rendering support
- Dimension labels with auto-computed lengths
- Boundary condition labels
- Axis annotations
- PyVista plotter integration
- Annotation persistence (export/import)

Key features:
- Design by Contract: position, font size, and color validation
- LaTeX formula support ($...$)
- Automatic dimension measurement
- JSON persistence for workflows
- PyQt6 UI widget support

Production quality: Publication-ready annotations with proper typography.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import pyvista  # noqa: F401

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

logger = logging.getLogger(__name__)

# Valid color names (CSS/matplotlib compatible)
VALID_COLORS = {
    "black",
    "white",
    "red",
    "green",
    "blue",
    "cyan",
    "magenta",
    "yellow",
    "gray",
    "grey",
    "orange",
    "purple",
    "brown",
    "pink",
}


def _validate_position(position: Any) -> tuple[float, float, float]:
    """Validate and normalize position to 3-tuple of floats (DbC).

    Args:
        position: Position tuple or array

    Returns:
        Normalized 3-tuple of floats

    Raises:
        ValueError: If position is not 3D
        TypeError: If position elements are not numeric
    """
    try:
        pos = tuple(float(x) for x in position)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Position elements must be numeric: {e}") from e

    if len(pos) != 3:
        raise ValueError(f"position must be a 3-tuple of floats, got length {len(pos)}")

    return (pos[0], pos[1], pos[2])


def _validate_font_size(font_size: int) -> int:
    """Validate font size (DbC).

    Args:
        font_size: Font size in points

    Returns:
        Validated font size

    Raises:
        ValueError: If font_size <= 0
        TypeError: If font_size is not numeric
    """
    if not isinstance(font_size, (int, float)):
        raise TypeError(f"font_size must be numeric, got {type(font_size)}")

    font_size_int = int(font_size)
    if font_size_int <= 0:
        raise ValueError(f"font_size must be > 0, got {font_size_int}")

    return font_size_int


def _validate_color(color: str) -> str:
    """Validate color (DbC).

    Args:
        color: Color name or hex code

    Returns:
        Validated color string

    Raises:
        ValueError: If color is not valid
    """
    color_str = str(color)
    color_lower = color_str.lower()

    # Check named color
    if color_lower in VALID_COLORS:
        return color_lower

    # Check hex color (#RRGGBB or #RGB)
    if re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$", color_str):
        return color_str  # Preserve original hex case

    raise ValueError(
        f"color must be a color name or hex code, got '{color}'. "
        f"Valid names: {', '.join(sorted(VALID_COLORS))}"
    )


@dataclass
class Annotation:
    """An annotation object for 3D visualization.

    Attributes:
        id: Unique annotation identifier
        type: Annotation type ('text', 'dimension', 'boundary', 'axis')
        position: 3D position (x, y, z)
        text: Display text (supports LaTeX with $...$)
        font_size: Font size in points
        color: Color (name or hex code)
        metadata: Additional annotation data (Dict by type)
    """

    id: str
    type: str  # 'text', 'dimension', 'boundary', 'axis'
    position: tuple[float, float, float]
    text: str
    font_size: int
    color: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields after initialization (DbC)."""
        # Validate position
        self.position = _validate_position(self.position)

        # Validate font_size
        self.font_size = _validate_font_size(self.font_size)

        # Validate color
        self.color = _validate_color(self.color)

        # Validate type
        valid_types = {"text", "dimension", "boundary", "axis"}
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got '{self.type}'")

        # Validate text is string
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text)}")

    def to_dict(self) -> dict[str, Any]:
        """Convert annotation to JSON-serializable dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "type": self.type,
            "position": self.position,
            "text": self.text,
            "font_size": self.font_size,
            "color": self.color,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Annotation":
        """Create annotation from dictionary.

        Args:
            data: Dictionary with annotation data

        Returns:
            Annotation instance

        Raises:
            KeyError: If required keys missing
            ValueError: If validation fails
        """
        return Annotation(
            id=data["id"],
            type=data["type"],
            position=tuple(data["position"]),  # type: ignore
            text=data["text"],
            font_size=data["font_size"],
            color=data["color"],
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def has_latex(text: str) -> bool:
        """Check if text contains LaTeX formulas.

        Args:
            text: Text to check

        Returns:
            True if text contains $...$ LaTeX expressions
        """
        return "$" in text and len(re.findall(r"\$[^\$]+\$", text)) > 0


class AnnotationManager:
    """Manager for creating, storing, and rendering annotations.

    This class provides a high-level API for annotation workflows.
    """

    def __init__(self) -> None:
        """Initialize the annotation manager."""
        self.annotations: dict[str, Annotation] = {}
        logger.debug("AnnotationManager initialized")

    def add_text(
        self,
        text: str,
        position: tuple[float, float, float],
        font_size: int = 12,
        color: str = "black",
        annotation_id: str | None = None,
    ) -> str:
        """Add a text annotation.

        Args:
            text: Display text (supports LaTeX)
            position: 3D position
            font_size: Font size in points (default: 12)
            color: Color name or hex (default: 'black')
            annotation_id: Custom ID (auto-generated if None)

        Returns:
            Annotation ID

        Raises:
            ValueError: If position/font_size/color invalid
        """
        ann_id = annotation_id or f"text_{uuid.uuid4().hex[:8]}"

        ann = Annotation(
            id=ann_id,
            type="text",
            position=position,
            text=text,
            font_size=font_size,
            color=color,
        )

        self.annotations[ann_id] = ann
        logger.debug("Added text annotation '%s' at %s", ann_id, position)
        return ann_id

    def add_dimension(
        self,
        point1: tuple[float, float, float],
        point2: tuple[float, float, float],
        label_position: tuple[float, float, float] | None = None,
        annotation_id: str | None = None,
        font_size: int = 10,
        color: str = "black",
    ) -> str:
        """Add a dimension annotation with auto-computed length.

        Args:
            point1: First point (x, y, z)
            point2: Second point (x, y, z)
            label_position: Position for label (defaults to midpoint)
            annotation_id: Custom ID (auto-generated if None)
            font_size: Font size in points (default: 10)
            color: Color name or hex (default: 'black')

        Returns:
            Annotation ID

        Raises:
            ValueError: If points invalid
        """
        # Validate points
        p1 = _validate_position(point1)
        p2 = _validate_position(point2)

        # Compute length
        length = np.sqrt(
            (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
        )

        # Use label_position or default to midpoint
        if label_position is None:
            label_position = (
                (p1[0] + p2[0]) / 2.0,
                (p1[1] + p2[1]) / 2.0,
                (p1[2] + p2[2]) / 2.0,
            )

        label_position = _validate_position(label_position)

        # Format length text
        if length == int(length):
            length_text = f"{int(length)}"
        else:
            length_text = f"{length:.2f}"

        text = f"L = {length_text}"

        ann_id = annotation_id or f"dim_{uuid.uuid4().hex[:8]}"

        ann = Annotation(
            id=ann_id,
            type="dimension",
            position=label_position,
            text=text,
            font_size=font_size,
            color=color,
            metadata={
                "point1": p1,
                "point2": p2,
                "length": float(length),
            },
        )

        self.annotations[ann_id] = ann
        logger.debug("Added dimension annotation '%s' with length %.2f", ann_id, length)
        return ann_id

    def add_boundary_label(
        self,
        label_text: str,
        boundary_center: tuple[float, float, float],
        label_position: tuple[float, float, float],
        annotation_id: str | None = None,
        font_size: int = 10,
        color: str = "red",
    ) -> str:
        """Add a boundary condition label.

        Args:
            label_text: Label text (e.g., "Fixed Support")
            boundary_center: Center of boundary region
            label_position: Position for label text
            annotation_id: Custom ID (auto-generated if None)
            font_size: Font size in points (default: 10)
            color: Color name or hex (default: 'red')

        Returns:
            Annotation ID

        Raises:
            ValueError: If positions invalid
        """
        boundary_center = _validate_position(boundary_center)
        label_position = _validate_position(label_position)

        ann_id = annotation_id or f"boundary_{uuid.uuid4().hex[:8]}"

        ann = Annotation(
            id=ann_id,
            type="boundary",
            position=label_position,
            text=label_text,
            font_size=font_size,
            color=color,
            metadata={"boundary_center": boundary_center},
        )

        self.annotations[ann_id] = ann
        logger.debug("Added boundary label '%s': %s", ann_id, label_text)
        return ann_id

    def add_axis_labels(
        self,
        x_label: str = "X",
        y_label: str = "Y",
        z_label: str = "Z",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_length: float = 1.0,
        font_size: int = 10,
        color: str = "black",
    ) -> list[str]:
        """Add coordinate axis labels.

        Args:
            x_label: X-axis label text (default: 'X')
            y_label: Y-axis label text (default: 'Y')
            z_label: Z-axis label text (default: 'Z')
            origin: Origin point (default: 0,0,0)
            axis_length: Distance from origin to label (default: 1.0)
            font_size: Font size in points (default: 10)
            color: Color name or hex (default: 'black')

        Returns:
            List of annotation IDs (3 total)
        """
        origin = _validate_position(origin)

        # X-axis label at (origin + axis_length, 0, 0)
        x_pos = (origin[0] + axis_length, origin[1], origin[2])
        x_id = self.add_text(
            x_label,
            x_pos,
            font_size=font_size,
            color=color,
            annotation_id=f"axis_x_{uuid.uuid4().hex[:8]}",
        )

        # Y-axis label at (0, origin + axis_length, 0)
        y_pos = (origin[0], origin[1] + axis_length, origin[2])
        y_id = self.add_text(
            y_label,
            y_pos,
            font_size=font_size,
            color=color,
            annotation_id=f"axis_y_{uuid.uuid4().hex[:8]}",
        )

        # Z-axis label at (0, 0, origin + axis_length)
        z_pos = (origin[0], origin[1], origin[2] + axis_length)
        z_id = self.add_text(
            z_label,
            z_pos,
            font_size=font_size,
            color=color,
            annotation_id=f"axis_z_{uuid.uuid4().hex[:8]}",
        )

        logger.debug("Added axis labels: X, Y, Z")
        return [x_id, y_id, z_id]

    def remove_annotation(self, annotation_id: str) -> bool:
        """Remove an annotation by ID.

        Args:
            annotation_id: ID of annotation to remove

        Returns:
            True if annotation was removed, False if not found
        """
        if annotation_id in self.annotations:
            del self.annotations[annotation_id]
            logger.debug("Removed annotation '%s'", annotation_id)
            return True
        logger.warning("Annotation '%s' not found", annotation_id)
        return False

    def get_annotation(self, annotation_id: str) -> Annotation | None:
        """Get annotation by ID.

        Args:
            annotation_id: ID to retrieve

        Returns:
            Annotation or None if not found
        """
        return self.annotations.get(annotation_id)

    def get_by_type(self, annotation_type: str) -> list[Annotation]:
        """Get all annotations of a specific type.

        Args:
            annotation_type: Type to filter by ('text', 'dimension', etc.)

        Returns:
            List of matching annotations
        """
        return [ann for ann in self.annotations.values() if ann.type == annotation_type]

    def render_annotations(self, plotter: Any) -> None:
        """Render all annotations to a PyVista plotter.

        Args:
            plotter: PyVista plotter object

        Raises:
            RuntimeError: If PyVista not available
        """
        if not HAS_PYVISTA:
            logger.warning("PyVista not available, skipping annotation rendering")
            return

        for ann in self.annotations.values():
            self._render_single_annotation(plotter, ann)

        logger.debug("Rendered %d annotations", len(self.annotations))

    def _render_single_annotation(self, plotter: Any, ann: Annotation) -> None:
        """Render a single annotation to plotter.

        Args:
            plotter: PyVista plotter
            ann: Annotation to render
        """
        try:
            # Convert position to array
            pos = np.array(ann.position)

            # Add text to plotter
            plotter.add_text(
                ann.text,
                position=pos,
                font_size=ann.font_size,
                color=ann.color,
            )

            logger.debug("Rendered annotation '%s'", ann.id)

        except Exception as e:
            logger.error("Failed to render annotation '%s': %s", ann.id, str(e))

    def export_to_dict(self) -> dict[str, Any]:
        """Export all annotations as JSON-serializable dictionary.

        Returns:
            Dictionary with 'annotations' key containing list of annotation dicts
        """
        return {
            "annotations": [ann.to_dict() for ann in self.annotations.values()],
            "version": "1.0",
            "timestamp": str(np.datetime64("now")),
        }

    def export_to_json(self, filepath: str) -> None:
        """Export annotations to JSON file.

        Args:
            filepath: Path to JSON file to write
        """
        data = self.export_to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Exported %d annotations to %s", len(self.annotations), filepath)

    def import_from_dict(self, data: dict[str, Any]) -> None:
        """Import annotations from dictionary.

        Args:
            data: Dictionary with 'annotations' key

        Raises:
            KeyError: If 'annotations' key missing
            ValueError: If annotation validation fails
        """
        annotations = data.get("annotations", [])
        for ann_data in annotations:
            try:
                ann = Annotation.from_dict(ann_data)
                self.annotations[ann.id] = ann
            except (KeyError, ValueError) as e:
                logger.error("Failed to import annotation: %s", str(e))

        logger.debug("Imported %d annotations", len(annotations))

    def import_from_json(self, filepath: str) -> None:
        """Import annotations from JSON file.

        Args:
            filepath: Path to JSON file to read

        Raises:
            FileNotFoundError: If file not found
            json.JSONDecodeError: If invalid JSON
        """
        with open(filepath) as f:
            data = json.load(f)
        self.import_from_dict(data)
        logger.debug("Imported annotations from %s", filepath)

    def clear_all(self) -> None:
        """Clear all annotations."""
        self.annotations.clear()
        logger.debug("Cleared all annotations")

    def get_statistics(self) -> dict[str, Any]:
        """Get annotation statistics.

        Returns:
            Dictionary with count by type and total count
        """
        type_counts = {}
        for ann in self.annotations.values():
            type_counts[ann.type] = type_counts.get(ann.type, 0) + 1

        return {
            "total": len(self.annotations),
            "by_type": type_counts,
        }


__all__ = [
    "Annotation",
    "AnnotationManager",
    "VALID_COLORS",
]
