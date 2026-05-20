# mypy: ignore-errors
# ruff: noqa: E501
import logging

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("p1am_control.desktop.mimic")


class MimicReactorItem(QGraphicsRectItem):
    """QGraphicsRectItem representing a reactor zone in the plant."""

    def __init__(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        tag_id: int,
        name: str,
        description: str,
        tab_parent,
    ) -> None:
        super().__init__(x, y, w, h)
        self.tag_id = tag_id
        self.name = name
        self.description = description
        self.tab_parent = tab_parent
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        # Style
        self.normal_pen = QPen(QColor("#4facfe"), 2)
        self.hover_pen = QPen(QColor("#00f2fe"), 3)
        self.selected_pen = QPen(QColor("#ffd60a"), 3, Qt.PenStyle.DashLine)

        self.normal_brush = QBrush(QColor("#1e222b"))
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self.tab_parent.select_element(
            self.tag_id, "reactor", self.name, self.description
        )
        self.update()

    def hoverEnterEvent(self, event) -> None:
        if not self.isSelected():
            self.setPen(self.hover_pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if not self.isSelected():
            self.setPen(self.normal_pen)
        super().hoverLeaveEvent(event)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self.isSelected():
            self.setPen(self.selected_pen)
        else:
            self.setPen(self.normal_pen)
        super().paint(painter, option, widget)


class MimicValveItem(QGraphicsPolygonItem):
    """QGraphicsPolygonItem representing a control valve (hourglass shape)."""

    def __init__(
        self,
        x: float,
        y: float,
        cv_tag_id: int,
        name: str,
        description: str,
        tab_parent,
    ) -> None:
        # Create hourglass polygon relative to (0,0) then translate
        poly = QPolygonF(
            [
                QPointF(-10, -10),
                QPointF(10, 10),
                QPointF(10, -10),
                QPointF(-10, 10),
                QPointF(-10, -10),
            ]
        )
        super().__init__(poly)
        self.setPos(x, y)
        self.cv_tag_id = cv_tag_id
        self.name = name
        self.description = description
        self.tab_parent = tab_parent
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        self.normal_pen = QPen(QColor("#ff375f"), 2)
        self.hover_pen = QPen(QColor("#ff7b93"), 3)
        self.selected_pen = QPen(QColor("#ffd60a"), 3, Qt.PenStyle.DashLine)

        self.normal_brush = QBrush(QColor("#2d1218"))
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self.tab_parent.select_element(
            self.cv_tag_id, "valve", self.name, self.description
        )
        self.update()

    def hoverEnterEvent(self, event) -> None:
        if not self.isSelected():
            self.setPen(self.hover_pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if not self.isSelected():
            self.setPen(self.normal_pen)
        super().hoverLeaveEvent(event)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self.isSelected():
            self.setPen(self.selected_pen)
        else:
            self.setPen(self.normal_pen)
        super().paint(painter, option, widget)


class MimicTab(QWidget):
    """Mimic display tab showing plant diagram and overlaying live data."""

    # Emitted when an element is clicked/selected: (tag_id, element_type, name, description)
    elementSelected = pyqtSignal(int, str, str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mimic_tab")

        # Active selection tracking
        self.selected_tag_id = -1

        # Tag overlays dictionary: tag_id -> QGraphicsTextItem
        self.overlays = {}

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Top Control Toolbar
        toolbar = QHBoxLayout()
        self.btn_zoom_in = QPushButton("Zoom In (+)", self)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out = QPushButton("Zoom Out (-)", self)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_zoom_fit = QPushButton("Fit to Screen", self)
        self.btn_zoom_fit.clicked.connect(self._zoom_fit)

        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.btn_zoom_fit)
        toolbar.addStretch()

        layout.addLayout(toolbar)

        # QGraphicsView & Scene
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QBrush(QColor("#15181e")))
        self.scene.setSceneRect(0, 0, 1050, 450)

        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(self.view)

        # Draw the gasification plant schematic
        self._draw_plant()

    def _draw_plant(self) -> None:
        # Define positions for reactors
        # X coordinates spaced out nicely to show flow left-to-right
        self.nodes = {
            "drying": {
                "x": 30,
                "y": 120,
                "w": 100,
                "h": 160,
                "tag": 1,
                "name": "Drying Hopper",
                "desc": "Pre-dries hopper biomass feedstock",
            },
            "pyrolysis": {
                "x": 180,
                "y": 120,
                "w": 100,
                "h": 160,
                "tag": 3,
                "name": "Pyrolysis Zone",
                "desc": "Thermochemical decomposition zone",
            },
            "combustion": {
                "x": 330,
                "y": 120,
                "w": 100,
                "h": 160,
                "tag": 5,
                "name": "Combustion Zone",
                "desc": "Partial oxidation and cracking zone",
            },
            "reduction": {
                "x": 480,
                "y": 120,
                "w": 100,
                "h": 160,
                "tag": 7,
                "name": "Reduction Zone",
                "desc": "Char gasification zone",
            },
            "quench": {
                "x": 630,
                "y": 150,
                "w": 80,
                "h": 120,
                "tag": 13,
                "name": "Quench Cooler",
                "desc": "Cools syngas via water contact",
            },
            "scrubber": {
                "x": 760,
                "y": 150,
                "w": 80,
                "h": 120,
                "tag": 14,
                "name": "Scrubber",
                "desc": "Filters particulates and condensables",
            },
            "flare": {
                "x": 890,
                "y": 100,
                "w": 60,
                "h": 200,
                "tag": 15,
                "name": "Flare Stack",
                "desc": "Thermal oxidizer for waste syngas",
            },
        }

        # Draw pipes (lines connecting zones)
        pipe_pen = QPen(QColor("#7f8c8d"), 4)

        # Pipes between main reactors
        self.scene.addLine(130, 200, 180, 200, pipe_pen)
        self.scene.addLine(280, 200, 330, 200, pipe_pen)
        self.scene.addLine(430, 200, 480, 200, pipe_pen)
        self.scene.addLine(580, 200, 630, 200, pipe_pen)
        self.scene.addLine(710, 200, 760, 200, pipe_pen)
        self.scene.addLine(840, 200, 890, 200, pipe_pen)

        # Draw main reactors/zones
        for _key, node in self.nodes.items():
            reactor = MimicReactorItem(
                node["x"],
                node["y"],
                node["w"],
                node["h"],
                node["tag"],
                node["name"],
                node["desc"],
                self,
            )
            self.scene.addItem(reactor)

            # Reactor title text
            title = QGraphicsTextItem(node["name"])
            title.setDefaultTextColor(QColor("#e1e4e8"))
            title.setPos(node["x"] + 2, node["y"] - 25)
            # Make it bold
            font = title.font()
            font.setBold(True)
            font.setPointSize(9)
            title.setFont(font)
            self.scene.addItem(title)

            # Live tag label
            label = QGraphicsTextItem("PV: --")
            label.setDefaultTextColor(QColor("#ffd60a"))
            label.setPos(node["x"] + 10, node["y"] + (node["h"] // 2) - 10)
            self.scene.addItem(label)
            self.overlays[node["tag"]] = label

        # Draw Control Valves on pipelines
        self.valves = {
            "valve_feed": {
                "x": 155,
                "y": 200,
                "tag": 2,
                "name": "Feed Valve",
                "desc": "Controls biomass feed rate into Pyrolysis",
            },
            "valve_air": {
                "x": 305,
                "y": 200,
                "tag": 4,
                "name": "Air Control Valve",
                "desc": "Regulates oxygen intake into Combustion",
            },
            "valve_water": {
                "x": 605,
                "y": 200,
                "tag": 6,
                "name": "Quench Water Valve",
                "desc": "Regulates water flow to Quench Cooler",
            },
            "valve_flare": {
                "x": 865,
                "y": 200,
                "tag": 8,
                "name": "Flare Control Valve",
                "desc": "Regulates syngas flow to Flare stack",
            },
        }

        for _key, valve in self.valves.items():
            valve_item = MimicValveItem(
                valve["x"], valve["y"], valve["tag"], valve["name"], valve["desc"], self
            )
            self.scene.addItem(valve_item)

            # Valve label
            label = QGraphicsTextItem("CV: --")
            label.setDefaultTextColor(QColor("#ff7b93"))
            label.setPos(valve["x"] - 20, valve["y"] + 15)
            self.scene.addItem(label)
            self.overlays[valve["tag"]] = label

    def select_element(
        self, tag_id: int, element_type: str, name: str, description: str
    ) -> None:
        self.selected_tag_id = tag_id
        # Emit signal to notify sidebar
        self.elementSelected.emit(tag_id, element_type, name, description)
        logger.info(f"Selected HMI element: {name} (Tag {tag_id})")

    def update_telemetry(self, tags: list[float]) -> None:
        """Called by main window to update mimic telemetry values."""
        for tag_id, label_item in self.overlays.items():
            if tag_id < len(tags):
                val = tags[tag_id]
                # Format appropriately
                if tag_id in [1, 2, 4, 6, 8, 13, 14]:
                    # Level / flow / valve position usually 0-100%
                    label_item.setPlainText(f"Value: {val:.1f}%")
                elif tag_id in [3, 5, 7, 15]:
                    # Temperature in C
                    label_item.setPlainText(f"Temp: {val:.1f}°C")
                else:
                    label_item.setPlainText(f"Val: {val:.2f}")

    def _zoom_in(self) -> None:
        self.view.scale(1.2, 1.2)

    def _zoom_out(self) -> None:
        self.view.scale(1 / 1.2, 1 / 1.2)

    def _zoom_fit(self) -> None:
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
