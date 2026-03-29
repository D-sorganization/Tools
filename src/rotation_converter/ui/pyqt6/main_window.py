# mypy: disable-error-code="attr-defined, misc"
"""Rotation Converter Main Window -- PyQt6 GUI.

Tabbed interface providing:
1. Rotation Converter -- live pairwise conversion between all representations
2. Rigid Transform -- frame-aware SE(3) with body/space twist conversion
3. Trajectory Plots -- screw axis, Euler, quaternion, and body-frame plots
4. 3D Visualiser -- interactive coordinate-frame and screw-axis rendering

Each tab is implemented in its own module (god-class decomposition):
- rotation_tab.py       -> RotationConverterTab
- rigid_transform_tab.py -> RigidTransformTab
- trajectory_tab.py     -> TrajectoryPlotsTab
- screw_visualiser_tab.py -> ScrewVisualiserTab
- console_tab.py        -> CommandConsoleTab
- reference_frame_tab.py -> ReferenceFrameTab

Shared plotting helpers live in plot_helpers.py.
"""

from __future__ import annotations

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import rotation_converter as rc
from rotation_converter.ui.pyqt6.console_tab import CommandConsoleTab
from rotation_converter.ui.pyqt6.reference_frame_tab import ReferenceFrameTab
from rotation_converter.ui.pyqt6.rigid_transform_tab import RigidTransformTab
from rotation_converter.ui.pyqt6.rotation_tab import RotationConverterTab
from rotation_converter.ui.pyqt6.screw_visualiser_tab import ScrewVisualiserTab
from rotation_converter.ui.pyqt6.trajectory_tab import TrajectoryPlotsTab

# ── Theme integration (optional -- graceful fallback) ──────────────
_THEME_AVAILABLE = False
try:
    from theme import get_theme_manager

    _THEME_AVAILABLE = True
except ImportError:
    pass

# Re-export tab classes for backward compatibility
__all__ = [
    "RotationConverterMainWindow",
    "RotationConverterTab",
    "RigidTransformTab",
    "TrajectoryPlotsTab",
    "ScrewVisualiserTab",
]

# Backward-compatible aliases for the helper functions that were
# previously defined here (used by tests and external callers).
from rotation_converter.ui.pyqt6.plot_helpers import (  # noqa: E402, F401
    EULER_CONVENTIONS,
)


class RotationConverterMainWindow(QMainWindow):
    """Main window with tabbed interface and theme integration."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rotation Converter")
        self.setMinimumSize(1200, 800)
        self._build_ui()
        self._build_menus()
        self._apply_theme()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()
        self._tabs.addTab(RotationConverterTab(), "Rotation Converter")
        self._tabs.addTab(RigidTransformTab(), "Rigid Transform")
        self._tabs.addTab(ReferenceFrameTab(), "Reference Frames & Lie Groups")
        self._tabs.addTab(TrajectoryPlotsTab(), "Trajectory Plots")
        self._tabs.addTab(ScrewVisualiserTab(), "3D Screw Visualiser")
        self._tabs.addTab(CommandConsoleTab(), "Python Console")
        layout.addWidget(self._tabs)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready -- select a tab to begin")

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()
        assert menu_bar is not None

        # File menu
        file_menu = menu_bar.addMenu("&File")
        assert file_menu is not None
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        assert help_menu is not None
        about = QAction("&About", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _apply_theme(self) -> None:
        if _THEME_AVAILABLE:
            try:
                mgr = get_theme_manager()
                mgr.apply_theme_to_window(self)
                mgr.themeChanged.connect(self._on_theme_changed)
            except Exception:  # noqa: BLE001
                pass

    def _on_theme_changed(self, theme_name: str) -> None:
        """Refresh all plots when the theme changes."""
        for i in range(self._tabs.count()):
            tab = self._tabs.widget(i)
            if tab is None:
                continue
            if hasattr(tab, "_update_outputs"):
                tab._update_outputs()
            elif hasattr(tab, "_update"):
                tab._update()
            elif hasattr(tab, "_plot"):
                tab._plot()
            elif hasattr(tab, "_draw_frame"):
                tab._draw_frame()

    def _show_about(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "Rotation Converter",
            f"<b>Rotation Converter v{rc.__version__}</b><br><br>"
            "Comprehensive rotation and rigid-body transform converter "
            "with interactive 3D visualization.<br><br>"
            "Supports: quaternions, Euler angles, rotation matrices, "
            "axis-angle, Rodrigues vectors, SE(3), twists, screw axes, "
            "frame-aware transforms, and Modern Robotics kinematics.<br><br>"
            "Part of the D-sorganization Tools suite.",
        )
