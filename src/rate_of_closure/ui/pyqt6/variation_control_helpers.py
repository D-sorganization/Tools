"""Small accessible control-layout helpers for variation views."""

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget


def add_labeled_control(
    layout: QHBoxLayout, text: str, control: QWidget, stretch: int = 0
) -> None:
    """Add one keyboard-associated label and its named control."""
    label = QLabel(text)
    label.setBuddy(control)
    layout.addWidget(label)
    layout.addWidget(control, stretch=stretch)


__all__ = ["add_labeled_control"]
