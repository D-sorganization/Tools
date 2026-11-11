"""Honest attachment chip until file_upload is handled (#6023 wave 3.2, #5727)."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

if not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [str(ROOT / "src")]
sys.modules.setdefault("src", _src_pkg)

for _ns in (
    "src.shared",
    "src.shared.python",
    "src.shared.python.chat",
    "src.shared.python.ai",
    "src.shared.python.ai.gui",
):
    _parts = _ns.split(".")
    _mod = types.ModuleType(_ns)
    _mod.__path__ = [str(ROOT.joinpath(*_parts))]
    sys.modules.setdefault(_ns, _mod)

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication

from src.shared.python.chat._chat_dock_widget_qt import (
    ChatConnectionConfig,
    ChatDockWidget,
)
from src.shared.python.chat._qt.messaging import (
    FILE_UPLOAD_BACKEND_DELIVERED,
    MessagingMixin,
    attachment_bubble_implies_accepted,
    format_attachment_pending_bubble,
)

BACKEND_REJECTION = "Unknown action: file_upload"
ARTIFACTS = Path(os.environ.get("CURSOR_ARTIFACTS_DIR", "/opt/cursor/artifacts"))


class _RecordedBubble:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self._content = content

    def set_content(self, text: str) -> None:
        self._content = text

    @property
    def content(self) -> str:
        return self._content


class _HonestDock(MessagingMixin):
    def __init__(self) -> None:
        self.sent_ws: list[dict[str, Any]] = []
        self.bubbles: list[_RecordedBubble] = []
        self._attachment_manifest_by_sha: dict[str, dict[str, Any]] = {}

    def _send_ws(self, payload: dict[str, Any]) -> None:
        self.sent_ws.append(payload)

    def _add_bubble(self, role: str, content: str) -> _RecordedBubble:
        bubble = _RecordedBubble(role, content)
        self.bubbles.append(bubble)
        return bubble


def test_file_upload_unhandled_never_renders_accepted_chip(tmp_path: Path) -> None:
    assert FILE_UPLOAD_BACKEND_DELIVERED is False
    dock = _HonestDock()
    sample = tmp_path / "notes.txt"
    sample.write_text("sidekick attachment slice 3.2", encoding="utf-8")

    dock._ingest_attachment_path(sample)
    for bubble in dock.bubbles:
        assert not attachment_bubble_implies_accepted(bubble.content)
    assert format_attachment_pending_bubble("notes.txt") in dock.bubbles[-1].content

    assert dock._apply_file_upload_rejection(BACKEND_REJECTION)
    assert BACKEND_REJECTION in dock.bubbles[-1].content
    assert not attachment_bubble_implies_accepted(dock.bubbles[-1].content)


def test_deduplicated_attachment_not_accepted_while_unhandled(tmp_path: Path) -> None:
    dock = _HonestDock()
    body = json.dumps({"h2_fraction": 0.33})
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(body, encoding="utf-8")
    f2.write_text(body, encoding="utf-8")

    dock._ingest_attachment_path(f1)
    dock._ingest_attachment_path(f2)

    for bubble in dock.bubbles:
        assert not attachment_bubble_implies_accepted(bubble.content)


def test_chat_dock_maps_backend_rejection_to_bubble(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])

    widget = ChatDockWidget(
        connection=ChatConnectionConfig(
            app_context="test",
            app_name="honest_attachment_chip",
        ),
    )
    widget._attachment_manifest_by_sha = {}
    sample = tmp_path / "payload.bin"
    sample.write_bytes(b"unhandled-file-upload-oracle")
    widget._send_ws = MagicMock()  # type: ignore[method-assign]
    widget._ingest_attachment_path(sample)
    widget._on_message(json.dumps({"type": "error", "detail": BACKEND_REJECTION}))

    last_bubble = widget._message_layout.itemAt(
        widget._message_layout.count() - 2
    ).widget()
    assert last_bubble is not None
    assert not attachment_bubble_implies_accepted(last_bubble._content)
    assert BACKEND_REJECTION in last_bubble._content


@pytest.mark.parametrize(
    ("width", "height", "filename"),
    (
        (1280, 800, "sidekick_attachment_chip_1280w.png"),
        (390, 844, "sidekick_attachment_chip_390x844.png"),
    ),
)
def test_attachment_chip_screenshots(
    tmp_path: Path, width: int, height: int, filename: str
) -> None:
    """Capture desktop and mobile composer layouts with a rejected attachment chip."""
    app = QApplication.instance() or QApplication([])

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    widget = ChatDockWidget(
        connection=ChatConnectionConfig(
            app_context="test",
            app_name="honest_attachment_screenshots",
        ),
    )
    widget._attachment_manifest_by_sha = {}
    sample = tmp_path / "evidence.txt"
    sample.write_text("wave 3.2 honest chip", encoding="utf-8")
    widget._send_ws = MagicMock()  # type: ignore[method-assign]
    widget._ingest_attachment_path(sample)
    widget._on_message(json.dumps({"type": "error", "detail": BACKEND_REJECTION}))
    widget.set_collapsed(False)
    widget.resize(QSize(width, height))
    widget.show()
    app.processEvents()
    pixmap = widget.grab()
    target = ARTIFACTS / filename
    assert pixmap.save(str(target))
    assert target.stat().st_size > 0
