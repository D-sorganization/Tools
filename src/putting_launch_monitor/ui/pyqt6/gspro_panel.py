"""GSPro connection panel: connect, last reply, club, putting mode, override.

The panel owns one :class:`GsproClient` (built by an injectable factory so
tests hand in a fake) and the :class:`GsproSink` the monitor delivers to.
A heartbeat timer keeps the 201 player information fresh while connected;
the *putting mode* light follows :attr:`GsproClient.putting_mode`, and the
override checkbox is :attr:`GsproSink.always` — the only way a putt leaves
this program while GSPro says the player is not holding a putter.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from putting_launch_monitor.gspro import DEFAULT_HOST, DEFAULT_PORT, GsproClient
from putting_launch_monitor.monitor import GsproSink, LogSink, Sink
from shared.python.theme import Colors, Sizes, Weights, get_display_font

from .painting import TOKENS

logger = logging.getLogger(__name__)

DEVICE_ID = "camera-putting-monitor"
HEARTBEAT_MS = 5_000
ClientFactory = Callable[[str, int], GsproClient]


def default_client(host: str, port: int) -> GsproClient:
    return GsproClient(device_id=DEVICE_ID, host=host, port=port)


class GsproPanel(QGroupBox):
    """Connection controls and the simulator's view of the player."""

    connection_changed = pyqtSignal(bool)

    def __init__(
        self,
        client_factory: ClientFactory = default_client,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("GSPro", parent)
        self._factory = client_factory
        self.client: GsproClient | None = None
        self.sink: GsproSink | None = None
        self.host = QLineEdit(DEFAULT_HOST)
        self.host.setToolTip("Where GSPro's Open Connect listener runs.")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(DEFAULT_PORT)
        self.port.setToolTip(
            "Open Connect port. GSPro itself opens 921; GSPconnect.exe alone "
            "listens on 1250 and does not forward shots."
        )
        self.connect_button = QPushButton("Connect")
        self.connect_button.setToolTip(
            "Open the connection and send a heartbeat; GSPro answers with the "
            "player's club. Click again to disconnect."
        )
        self.connect_button.clicked.connect(self.toggle)
        self.status = QLabel("disconnected")
        self.status.setToolTip("Connection state and the last error, if any.")
        self.reply = QLabel("—")
        self.reply.setToolTip("Code and message of GSPro's last reply.")
        self.reply.setWordWrap(True)
        self.club = QLabel("—")
        self.club.setToolTip("Player.Club from GSPro's last 201 message.")
        self.mode = QLabel("not putting")
        self.mode.setFont(get_display_font(Sizes.MD, Weights.SEMIBOLD))
        self.mode.setToolTip(
            "Lit when GSPro reports a putter in hand; putts are held otherwise."
        )
        self.always = QCheckBox("Send even outside putting mode")
        self.always.setToolTip(
            "Override the putting-mode gate: every accepted putt is sent, "
            "whatever club GSPro reports. Off is the safe default."
        )
        self.always.toggled.connect(self._on_always)
        self.counts = QLabel("sent 0, held 0")
        self.counts.setToolTip("Putts forwarded to GSPro and putts held back.")
        endpoint = QHBoxLayout()
        endpoint.setSpacing(TOKENS.spacing_px)
        endpoint.addWidget(self.host, 1)
        endpoint.addWidget(self.port)
        endpoint.addWidget(self.connect_button)
        form = QFormLayout(self)
        form.setSpacing(TOKENS.spacing_px)
        form.addRow("Endpoint", endpoint)
        form.addRow("Status", self.status)
        form.addRow("Last reply", self.reply)
        form.addRow("Club", self.club)
        form.addRow("Putting mode", self.mode)
        form.addRow(self.always)
        form.addRow("Shots", self.counts)
        self._timer = QTimer(self)
        self._timer.setInterval(HEARTBEAT_MS)
        self._timer.timeout.connect(self.heartbeat)
        self.refresh()

    # -- connection -------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self.client is not None and self.client.connected

    def toggle(self) -> None:
        if self.connected:
            self.close_connection()
        else:
            self.open_connection()

    def open_connection(self) -> None:
        """Build the client, connect and heartbeat once; errors land in ``status``."""
        client = self._factory(self.host.text().strip(), self.port.value())
        try:
            client.connect()
            client.heartbeat()
        except (OSError, ValueError) as exc:
            logger.warning("GSPro connect failed: %s", exc)
            client.close()
            self.status.setText(f"failed: {exc}")
            return
        self.bind(client)
        self._timer.start()
        self.connection_changed.emit(True)

    def close_connection(self) -> None:
        self._timer.stop()
        if self.client is not None:
            self.client.close()
        self.refresh()
        self.connection_changed.emit(False)

    def bind(self, client: GsproClient) -> None:
        """Adopt a client (already connected or not) as the sink's target."""
        self.client = client
        self.sink = GsproSink(client=client, always=self.always.isChecked())
        self.refresh()

    def sink_for_monitor(self) -> Sink:
        """The sink a monitor should deliver to right now: GSPro if connected."""
        if self.sink is not None and self.connected:
            return self.sink
        return LogSink()

    def heartbeat(self) -> None:
        if self.client is None or not self.client.connected:
            return
        try:
            self.client.heartbeat()
        except (OSError, ValueError) as exc:
            logger.warning("GSPro heartbeat failed: %s", exc)
            self.status.setText(f"lost: {exc}")
        self.refresh()

    # -- display ------------------------------------------------------------------
    def refresh(self) -> None:
        """Repaint every field from the client's and sink's state."""
        client = self.client
        connected = self.connected
        self.connect_button.setText("Disconnect" if connected else "Connect")
        self.host.setEnabled(not connected)
        self.port.setEnabled(not connected)
        if connected:
            self.status.setText("connected")
        elif client is None or not self.status.text().startswith(("failed", "lost")):
            self.status.setText("disconnected")
        reply = client.last_reply if client is not None else None
        self.reply.setText(
            f"{reply.code} {reply.message}".strip() if reply is not None else "—"
        )
        player = client.player if client is not None else None
        self.club.setText(player.club if player is not None else "—")
        putting = client is not None and client.putting_mode
        self.mode.setText("PUTTING" if putting else "not putting")
        tone = Colors.SUCCESS if putting else Colors.TEXT_MUTED
        self.mode.setStyleSheet(f"color: {tone};")
        sent, held = (self.sink.sent, self.sink.held) if self.sink else (0, 0)
        self.counts.setText(f"sent {sent}, held {held}")

    def _on_always(self, checked: bool) -> None:
        if self.sink is not None:
            self.sink.always = checked
        logger.info("GSPro override %s", "on" if checked else "off")
