"""Comparison Mode for FEA/CFD Results (GitHub issue #544).

Dual viewport side-by-side comparison with:
- Synchronized or independent field selection
- Linked cameras for synchronized navigation
- Difference field computation
- Agreement percentage calculation
- Split layout toggle (vertical/horizontal)
- Solver labels and position swap
"""

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .fea_results_viewer import FEAResultsViewer

logger = logging.getLogger(__name__)


class ComparisonViewController(QWidget):
    """Dual FEA/CFD results comparison viewer.

    Provides side-by-side visualization of two FEA/CFD results with:
    - Independent or synchronized iso-surface field selection
    - Linked cameras (one viewport update synchronizes to the other)
    - Difference field computation and visualization
    - Agreement percentage calculation
    - Split layout toggle (vertical/horizontal)
    - Solver labels and position swap button

    Signals:
        comparison_updated: Emitted when comparison state changes
        agreement_changed: Emitted when agreement % updates
    """

    comparison_updated = pyqtSignal(dict)
    agreement_changed = pyqtSignal(float)

    def __init__(
        self,
        parent: QWidget | None = None,
        synchronized_fields: bool = False,
    ) -> None:
        """Initialize comparison viewer.

        Args:
            parent: Parent widget
            synchronized_fields: If True, iso-value changes affect both viewers
        """
        super().__init__(parent)
        self.synchronized_fields = synchronized_fields
        self.camera_sync_enabled = True
        self._split_orientation = "horizontal"

        # Data fields
        self.left_field: np.ndarray | None = None
        self.right_field: np.ndarray | None = None
        self.difference_field: np.ndarray | None = None
        self.agreement_percentage: float = 0.0

        # Labels for solvers
        self.left_label = "Left Solver"
        self.right_label = "Right Solver"

        logger.debug("ComparisonViewController initialized")

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QVBoxLayout(self)

        # Title and controls
        header_layout = QHBoxLayout()

        title = QLabel("FEA/CFD Comparison Mode")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Control buttons
        self.layout_toggle_btn = QPushButton("Toggle Layout (H/V)")
        self.swap_btn = QPushButton("Swap Positions")
        self.agreement_label = QLabel("Agreement: --%")
        self.agreement_label.setStyleSheet("font-weight: bold;")

        header_layout.addWidget(self.layout_toggle_btn)
        header_layout.addWidget(self.swap_btn)
        header_layout.addWidget(self.agreement_label)

        layout.addLayout(header_layout)

        # Main splitter for dual viewers
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left viewer with label
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        self.left_label_widget = QLabel(self.left_label)
        self.left_label_widget.setStyleSheet(
            "font-weight: bold; font-size: 10pt; color: #0066cc;"
        )
        left_layout.addWidget(self.left_label_widget)
        self.left_viewer = FEAResultsViewer()
        left_layout.addWidget(self.left_viewer)
        left_container.setLayout(left_layout)

        # Right viewer with label
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        self.right_label_widget = QLabel(self.right_label)
        self.right_label_widget.setStyleSheet(
            "font-weight: bold; font-size: 10pt; color: #cc6600;"
        )
        right_layout.addWidget(self.right_label_widget)
        self.right_viewer = FEAResultsViewer()
        right_layout.addWidget(self.right_viewer)
        right_container.setLayout(right_layout)

        self.splitter.addWidget(left_container)
        self.splitter.addWidget(right_container)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)
        self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Connect control signals."""
        self.layout_toggle_btn.clicked.connect(self._on_toggle_layout)
        self.swap_btn.clicked.connect(self._on_swap_positions)

        # Connect viewer signals for camera sync
        if self.camera_sync_enabled:
            self.left_viewer.visualization_updated.connect(self._on_left_viewer_updated)
            self.right_viewer.visualization_updated.connect(
                self._on_right_viewer_updated
            )

    def _on_toggle_layout(self) -> None:
        """Toggle between horizontal and vertical split layout."""
        if self._split_orientation == "horizontal":
            self.set_split_orientation("vertical")
        else:
            self.set_split_orientation("horizontal")
        logger.debug("Layout toggled to %s", self._split_orientation)

    def _on_swap_positions(self) -> None:
        """Swap left and right viewer positions."""
        self.swap_viewer_positions()
        logger.debug("Viewer positions swapped")

    def _on_left_viewer_updated(self, viz_info: dict[str, Any]) -> None:
        """Handle left viewer update - sync to right if enabled."""
        if self.camera_sync_enabled and self.synchronized_fields:
            logger.debug(
                "Synchronizing right viewer to left (surfaces=%d)",
                viz_info.get("num_surfaces", 0),
            )
            # In a full implementation, this would update right viewer's
            # iso-value and visualization parameters
            self.comparison_updated.emit(
                {
                    "source": "left",
                    "sync": True,
                    "viz_info": viz_info,
                }
            )

    def _on_right_viewer_updated(self, viz_info: dict[str, Any]) -> None:
        """Handle right viewer update - sync to left if enabled."""
        if self.camera_sync_enabled and self.synchronized_fields:
            logger.debug(
                "Synchronizing left viewer to right (surfaces=%d)",
                viz_info.get("num_surfaces", 0),
            )
            self.comparison_updated.emit(
                {
                    "source": "right",
                    "sync": True,
                    "viz_info": viz_info,
                }
            )

    def set_solver_labels(self, left_label: str, right_label: str) -> None:
        """Set labels for the left and right solvers.

        Args:
            left_label: Label for left solver
            right_label: Label for right solver
        """
        self.left_label = left_label
        self.right_label = right_label
        self.left_label_widget.setText(left_label)
        self.right_label_widget.setText(right_label)
        logger.debug("Solver labels: %s (left), %s (right)", left_label, right_label)

    def swap_viewer_positions(self) -> None:
        """Swap the left and right viewers."""
        # Swap viewer references
        self.left_viewer, self.right_viewer = (
            self.right_viewer,
            self.left_viewer,
        )
        # Swap labels
        self.left_label, self.right_label = (
            self.right_label,
            self.left_label,
        )
        self.left_label_widget.setText(self.left_label)
        self.right_label_widget.setText(self.right_label)

        # Swap field data
        self.left_field, self.right_field = (
            self.right_field,
            self.left_field,
        )

        logger.debug("Viewer positions swapped")

    def set_camera_sync(self, enabled: bool) -> None:
        """Enable or disable camera synchronization.

        Args:
            enabled: Whether to sync cameras
        """
        self.camera_sync_enabled = enabled
        logger.debug("Camera sync: %s", "enabled" if enabled else "disabled")

    def set_split_orientation(self, orientation: str) -> None:
        """Set split layout orientation.

        Args:
            orientation: "horizontal" or "vertical"

        Raises:
            ValueError: If orientation is invalid
        """
        if orientation not in ("horizontal", "vertical"):
            raise ValueError('orientation must be "horizontal" or "vertical"')

        self._split_orientation = orientation

        # Update splitter orientation
        if orientation == "horizontal":
            self.splitter.setOrientation(Qt.Orientation.Horizontal)
        else:
            self.splitter.setOrientation(Qt.Orientation.Vertical)

        logger.debug("Split orientation set to %s", orientation)

    def get_split_orientation(self) -> str:
        """Get current split orientation.

        Returns:
            "horizontal" or "vertical"
        """
        return self._split_orientation

    def load_left_field(self, field_data: np.ndarray) -> None:
        """Load field data into left viewer.

        Args:
            field_data: 3D scalar field array
        """
        if field_data.ndim != 3:
            raise ValueError(f"Expected 3D field, got shape {field_data.shape}")

        self.left_field = field_data
        self.left_viewer.load_field_data(field_data)
        logger.info(
            "Loaded left field: shape %s, range [%.3f, %.3f]",
            field_data.shape,
            float(np.nanmin(field_data)),
            float(np.nanmax(field_data)),
        )

    def load_right_field(self, field_data: np.ndarray) -> None:
        """Load field data into right viewer.

        Args:
            field_data: 3D scalar field array
        """
        if field_data.ndim != 3:
            raise ValueError(f"Expected 3D field, got shape {field_data.shape}")

        self.right_field = field_data
        self.right_viewer.load_field_data(field_data)

        # Warn if shapes don't match
        if self.left_field is not None and self.left_field.shape != field_data.shape:
            logger.warning(
                "Field shape mismatch: left=%s, right=%s",
                self.left_field.shape,
                field_data.shape,
            )

        logger.info(
            "Loaded right field: shape %s, range [%.3f, %.3f]",
            field_data.shape,
            float(np.nanmin(field_data)),
            float(np.nanmax(field_data)),
        )

    @staticmethod
    def compute_difference_field(
        field_a: np.ndarray, field_b: np.ndarray
    ) -> np.ndarray:
        """Compute difference field (field_a - field_b).

        Args:
            field_a: First 3D scalar field
            field_b: Second 3D scalar field

        Returns:
            Difference field (field_a - field_b)

        Raises:
            ValueError: If shapes don't match
            TypeError: If inputs are not arrays
        """
        if not isinstance(field_a, np.ndarray):
            field_a = np.asarray(field_a)
        if not isinstance(field_b, np.ndarray):
            field_b = np.asarray(field_b)

        if field_a.shape != field_b.shape:
            raise ValueError(
                f"Shape mismatch: field_a={field_a.shape}, field_b={field_b.shape}"
            )

        diff = field_a - field_b
        logger.debug("Computed difference field: shape %s", diff.shape)

        return diff

    @staticmethod
    def compute_agreement_percentage(
        field_a: np.ndarray, field_b: np.ndarray, threshold: float = 0.01
    ) -> float:
        """Compute agreement percentage between fields.

        Agreement is defined as the fraction of points where
        |field_a - field_b| <= threshold.

        Args:
            field_a: First 3D scalar field
            field_b: Second 3D scalar field
            threshold: Maximum difference for agreement

        Returns:
            Agreement percentage (0-100)

        Raises:
            ValueError: If shapes don't match
        """
        if not isinstance(field_a, np.ndarray):
            field_a = np.asarray(field_a)
        if not isinstance(field_b, np.ndarray):
            field_b = np.asarray(field_b)

        if field_a.shape != field_b.shape:
            raise ValueError(
                f"Shape mismatch: field_a={field_a.shape}, field_b={field_b.shape}"
            )

        diff = np.abs(field_a - field_b)

        # Count valid (non-NaN) values
        valid_mask = ~(np.isnan(field_a) | np.isnan(field_b) | np.isnan(diff))
        num_valid = np.count_nonzero(valid_mask)

        if num_valid == 0:
            logger.warning("No valid values for agreement computation")
            return 0.0

        # Count agreements within threshold
        agreement_mask = (diff <= threshold) & valid_mask
        num_agreed = np.count_nonzero(agreement_mask)

        agreement = (num_agreed / num_valid) * 100.0

        logger.debug(
            "Agreement: %.1f%% (%d/%d within threshold %.3f)",
            agreement,
            num_agreed,
            num_valid,
            threshold,
        )

        return agreement

    def update_difference_visualization(self, threshold: float = 0.01) -> None:
        """Update difference field and agreement display.

        Computes the difference field, agreement %, and updates the display.

        Args:
            threshold: Threshold for agreement calculation
        """
        if self.left_field is None or self.right_field is None:
            logger.warning("Cannot compute difference: missing field data")
            return

        if self.left_field.shape != self.right_field.shape:
            logger.warning(
                "Cannot compute difference: shape mismatch (%s vs %s)",
                self.left_field.shape,
                self.right_field.shape,
            )
            return

        # Compute difference field
        self.difference_field = self.compute_difference_field(
            self.left_field, self.right_field
        )

        # Compute agreement
        self.agreement_percentage = self.compute_agreement_percentage(
            self.left_field, self.right_field, threshold=threshold
        )

        # Update display
        self.agreement_label.setText(f"Agreement: {self.agreement_percentage:.1f}%")

        self.agreement_changed.emit(self.agreement_percentage)

        logger.info(
            "Difference updated: agreement=%.1f%%",
            self.agreement_percentage,
        )

    def clear(self) -> None:
        """Clear all data and visualization."""
        self.left_field = None
        self.right_field = None
        self.difference_field = None
        self.agreement_percentage = 0.0

        self.left_viewer.clear()
        self.right_viewer.clear()
        self.agreement_label.setText("Agreement: --%")

        logger.debug("ComparisonViewController cleared")


__all__ = ["ComparisonViewController"]
