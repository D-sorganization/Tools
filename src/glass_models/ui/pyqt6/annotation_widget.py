"""PyQt6 widget for annotation control (GitHub issue #542).

Provides UI controls for creating and managing annotations:
- Annotation type selector (Text, Dimension, Boundary, Axis)
- Text input field for annotation content
- Font size spinner (8-72 points)
- Color picker with palette
- Add/remove/edit buttons
- Annotation list display
"""

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class AnnotationControlWidget(QWidget):
    """Control widget for creating and managing annotations.

    Signals:
        annotation_added: Emitted when annotation is added (dict with config)
        annotation_removed: Emitted when annotation is removed (str with id)
        annotation_type_changed: Emitted when type selector changes (str)
    """

    annotation_added = pyqtSignal(dict)
    annotation_removed = pyqtSignal(str)
    annotation_type_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize annotation control widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._current_color = "black"
        self._annotations: dict[str, dict] = {}
        self._setup_ui()
        self._connect_signals()
        logger.debug("AnnotationControlWidget initialized")

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)

        # Type selection group
        type_group = QGroupBox("Annotation Type", self)
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Text", "Dimension", "Boundary", "Axis"])
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Text input group
        text_group = QGroupBox("Annotation Content", self)
        text_layout = QVBoxLayout()

        text_label_layout = QHBoxLayout()
        text_label_layout.addWidget(QLabel("Text:"))
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(
            "Enter annotation text (LaTeX supported: $E=mc^2$)"
        )
        text_label_layout.addWidget(self.text_input)
        text_layout.addLayout(text_label_layout)

        # For dimensions: point input fields
        self.dim_group = QGroupBox("Dimension Points", self)
        dim_layout = QVBoxLayout()

        p1_layout = QHBoxLayout()
        p1_layout.addWidget(QLabel("Point 1:"))
        self.p1_x = QDoubleSpinBox()
        self.p1_y = QDoubleSpinBox()
        self.p1_z = QDoubleSpinBox()
        p1_layout.addWidget(QLabel("X:"))
        p1_layout.addWidget(self.p1_x)
        p1_layout.addWidget(QLabel("Y:"))
        p1_layout.addWidget(self.p1_y)
        p1_layout.addWidget(QLabel("Z:"))
        p1_layout.addWidget(self.p1_z)
        p1_layout.addStretch()
        dim_layout.addLayout(p1_layout)

        p2_layout = QHBoxLayout()
        p2_layout.addWidget(QLabel("Point 2:"))
        self.p2_x = QDoubleSpinBox()
        self.p2_y = QDoubleSpinBox()
        self.p2_z = QDoubleSpinBox()
        p2_layout.addWidget(QLabel("X:"))
        p2_layout.addWidget(self.p2_x)
        p2_layout.addWidget(QLabel("Y:"))
        p2_layout.addWidget(self.p2_y)
        p2_layout.addWidget(QLabel("Z:"))
        p2_layout.addWidget(self.p2_z)
        p2_layout.addStretch()
        dim_layout.addLayout(p2_layout)

        self.dim_group.setLayout(dim_layout)
        self.dim_group.setVisible(False)
        text_layout.addWidget(self.dim_group)

        text_group.setLayout(text_layout)
        layout.addWidget(text_group)

        # Formatting group
        format_group = QGroupBox("Formatting", self)
        format_layout = QVBoxLayout()

        # Font size
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Font Size:"))
        self.font_size_spinner = QSpinBox()
        self.font_size_spinner.setMinimum(8)
        self.font_size_spinner.setMaximum(72)
        self.font_size_spinner.setValue(12)
        font_layout.addWidget(self.font_size_spinner)
        font_layout.addWidget(QLabel("pt"))
        font_layout.addStretch()
        format_layout.addLayout(font_layout)

        # Color picker
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color:"))
        self.color_button = QPushButton()
        self.color_button.setFixedWidth(60)
        self._set_color_button_color(self._current_color)
        color_layout.addWidget(self.color_button)
        self.color_label = QLabel(self._current_color)
        color_layout.addWidget(self.color_label)
        color_layout.addStretch()
        format_layout.addLayout(color_layout)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Button group
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Annotation")
        self.edit_button = QPushButton("Edit Selected")
        self.edit_button.setEnabled(False)
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.setEnabled(False)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.remove_button)
        layout.addLayout(button_layout)

        # Annotation list
        list_layout = QVBoxLayout()
        list_layout.addWidget(QLabel("Annotations:"))
        self.annotation_list = QListWidget()
        self.annotation_list.setMaximumHeight(200)
        list_layout.addWidget(self.annotation_list)
        layout.addLayout(list_layout)

        # Options
        options_layout = QHBoxLayout()
        self.latex_checkbox = QCheckBox("LaTeX Mode")
        options_layout.addWidget(self.latex_checkbox)
        options_layout.addStretch()
        layout.addLayout(options_layout)

        layout.addStretch()
        self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Connect UI signals to slots."""
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.color_button.clicked.connect(self._on_color_picker_clicked)
        self.add_button.clicked.connect(self._on_add_annotation)
        self.edit_button.clicked.connect(self._on_edit_annotation)
        self.remove_button.clicked.connect(self._on_remove_annotation)
        self.annotation_list.itemSelectionChanged.connect(
            self._on_list_selection_changed
        )

    def _on_type_changed(self, ann_type: str) -> None:
        """Handle annotation type change.

        Args:
            ann_type: Selected annotation type
        """
        self.annotation_type_changed.emit(ann_type)

        # Show/hide dimension point inputs
        self.dim_group.setVisible(ann_type == "Dimension")
        logger.debug("Annotation type changed to: %s", ann_type)

    def _on_color_picker_clicked(self) -> None:
        """Handle color picker button click."""
        color = QColorDialog.getColor(
            QColor(self._current_color), self, "Select Annotation Color"
        )
        if color.isValid():
            self._current_color = color.name()
            self._set_color_button_color(self._current_color)
            self.color_label.setText(self._current_color)
            logger.debug("Color selected: %s", self._current_color)

    def _set_color_button_color(self, color: str) -> None:
        """Update color button appearance.

        Args:
            color: Color name or hex code
        """
        try:
            self.color_button.setStyleSheet(
                f"background-color: {color}; border: 1px solid black;"
            )
        except Exception as e:
            logger.warning("Failed to set color button: %s", str(e))

    def _on_add_annotation(self) -> None:
        """Handle add annotation button click."""
        ann_type = self.type_combo.currentText()
        text = self.text_input.text().strip()
        font_size = self.font_size_spinner.value()
        color = self._current_color

        if not text and ann_type != "Axis":
            logger.warning("Annotation text cannot be empty")
            return

        # Build annotation config
        config = {
            "type": ann_type.lower(),
            "text": text,
            "font_size": font_size,
            "color": color,
            "latex_enabled": self.latex_checkbox.isChecked(),
        }

        # Add dimension points if applicable
        if ann_type == "Dimension":
            config["point1"] = (
                self.p1_x.value(),
                self.p1_y.value(),
                self.p1_z.value(),
            )
            config["point2"] = (
                self.p2_x.value(),
                self.p2_y.value(),
                self.p2_z.value(),
            )

        # Generate ID
        ann_id = self._generate_annotation_id(ann_type)
        self._annotations[ann_id] = config

        # Add to list
        self._add_to_list(ann_id, config)

        # Emit signal
        self.annotation_added.emit(config)

        # Clear input
        self.text_input.clear()
        logger.debug("Added annotation: %s", ann_id)

    def _on_edit_annotation(self) -> None:
        """Handle edit annotation button click."""
        current_item = self.annotation_list.currentItem()
        if current_item is None:
            return

        # Get annotation ID from item
        ann_id = current_item.data(Qt.ItemDataRole.UserRole)
        if ann_id not in self._annotations:
            return

        config = self._annotations[ann_id]

        # Populate UI with current values
        self.type_combo.setCurrentText(config["type"].title())
        self.text_input.setText(config["text"])
        self.font_size_spinner.setValue(config["font_size"])
        self._current_color = config["color"]
        self._set_color_button_color(self._current_color)
        self.color_label.setText(self._current_color)
        self.latex_checkbox.setChecked(config.get("latex_enabled", False))

        logger.debug("Editing annotation: %s", ann_id)

    def _on_remove_annotation(self) -> None:
        """Handle remove annotation button click."""
        current_item = self.annotation_list.currentItem()
        if current_item is None:
            return

        ann_id = current_item.data(Qt.ItemDataRole.UserRole)
        if ann_id not in self._annotations:
            return

        del self._annotations[ann_id]
        self.annotation_list.takeItem(self.annotation_list.row(current_item))

        self.annotation_removed.emit(ann_id)
        logger.debug("Removed annotation: %s", ann_id)

    def _on_list_selection_changed(self) -> None:
        """Handle annotation list selection change."""
        has_selection = self.annotation_list.currentItem() is not None
        self.edit_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)

    def _add_to_list(self, ann_id: str, config: dict) -> None:
        """Add annotation to list widget.

        Args:
            ann_id: Annotation ID
            config: Annotation configuration
        """
        text = config["text"]
        ann_type = config["type"].title()
        item_text = f"[{ann_type}] {text[:30]}"
        if len(text) > 30:
            item_text += "..."

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, ann_id)
        self.annotation_list.addItem(item)

    @staticmethod
    def _generate_annotation_id(ann_type: str) -> str:
        """Generate a unique annotation ID.

        Args:
            ann_type: Annotation type

        Returns:
            Unique ID string
        """
        import uuid

        return f"{ann_type.lower()}_{uuid.uuid4().hex[:8]}"

    def get_annotations(self) -> dict[str, dict]:
        """Get all annotations.

        Returns:
            Dictionary of annotation configurations keyed by ID
        """
        return self._annotations.copy()

    def set_annotation_list(self, annotations: dict[str, dict]) -> None:
        """Set annotations from dictionary.

        Args:
            annotations: Dictionary of annotation configurations
        """
        self._annotations = annotations.copy()
        self.annotation_list.clear()

        for ann_id, config in annotations.items():
            self._add_to_list(ann_id, config)

    def clear_all(self) -> None:
        """Clear all annotations."""
        self._annotations.clear()
        self.annotation_list.clear()
        self.text_input.clear()
        logger.debug("Cleared all annotations")


__all__ = ["AnnotationControlWidget"]
