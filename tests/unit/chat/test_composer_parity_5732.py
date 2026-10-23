"""Unit tests for Sidekick composer parity (#5732, UD-PAIR #13660)."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

if importlib.util.find_spec("PyQt6") is None:
    _qt = MagicMock()
    for _mod in (
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.sip",
    ):
        sys.modules.setdefault(_mod, _qt)

from src.shared.python.chat._qt.messaging import (
    ActionChipModel,
    MessagingMixin,
    compute_attachment_intake,
    format_action_chip_text,
)
from src.shared.python.chat.voice_input_manager import VoiceInputManager


def _make_wav_bytes(duration_ms: int = 50) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * int(16 * duration_ms))
    return buf.getvalue()


class _FakeDock(MessagingMixin):
    def __init__(self) -> None:
        self.sent_ws: list[dict[str, Any]] = []
        self.bubbles: list[tuple[str, str]] = []
        self._voice_manager = VoiceInputManager()
        self._mic_btn = MagicMock()
        self._status_label = MagicMock()

    def _send_ws(self, payload: dict[str, Any]) -> None:
        self.sent_ws.append(payload)

    def _add_bubble(self, role: str, content: str) -> Any:
        self.bubbles.append((role, content))
        return {"role": role, "content": content}


def test_attachment_intake_sha256_deduplication_and_manifest(tmp_path: Path) -> None:
    dock = _FakeDock()
    f1 = tmp_path / "sim_report.json"
    f1.write_text(json.dumps({"syngas_h2": 0.42}), encoding="utf-8")
    f2 = tmp_path / "sim_report_copy.json"
    f2.write_text(json.dumps({"syngas_h2": 0.42}), encoding="utf-8")

    entry1, dup1 = dock._ingest_attachment_path(f1)
    assert dup1 is False
    assert len(entry1.sha256) == 64
    assert len(dock.sent_ws) == 1
    assert dock.sent_ws[0]["sha256"] == entry1.sha256

    entry2, dup2 = dock._ingest_attachment_path(f2)
    assert dup2 is True
    assert entry2.sha256 == entry1.sha256
    # Duplicate file must not re-upload over WebSocket
    assert len(dock.sent_ws) == 1
    assert "deduplicated" in dock.bubbles[-1][1]
    manifest = dock.attachment_manifest()
    assert len(manifest) == 1
    assert manifest[0]["filename"] == "sim_report.json"


def test_attachment_intake_rejects_missing_empty_and_oversized(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        compute_attachment_intake(tmp_path / "missing.txt")

    empty_file = tmp_path / "empty.txt"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        compute_attachment_intake(empty_file)

    big_file = tmp_path / "big.bin"
    big_file.write_bytes(b"x" * 128)
    with pytest.raises(ValueError, match="exceeds limit"):
        compute_attachment_intake(big_file, max_bytes=64)


def test_action_chip_rendering_and_validation() -> None:
    dock = _FakeDock()
    ok_chip = ActionChipModel(
        action_id="c1",
        name="run_simulation",
        status="ok",
        summary="H2/CO=2.14 converged in 12 iterations",
    )
    dock.add_action_chip(ok_chip)
    assert dock.bubbles[-1] == (
        "assistant",
        "[ActionChip OK: run_simulation#c1] H2/CO=2.14 converged in 12 iterations",
    )

    confirm_chip = ActionChipModel(
        action_id="c2",
        name="apply_reactor_preset",
        status="pending_confirmation",
        summary="Set T=1150 C, P=30 bar",
        requires_confirmation=True,
    )
    dock.add_action_chip(confirm_chip)
    assert "CONFIRM REQUIRED" in dock.bubbles[-1][1]

    with pytest.raises(ValueError, match="non-empty"):
        format_action_chip_text(
            ActionChipModel(action_id="c3", name="", status="ok", summary="bad")
        )


def test_voice_input_manager_whisper_status_and_wav_transcription() -> None:
    mgr = VoiceInputManager(model_name="base.en")
    with pytest.raises(ValueError, match="Invalid WAV"):
        mgr.transcribe_wav_bytes(b"not-a-wav")

    fake_seg = MagicMock()
    fake_seg.text = "set reactor temperature to 1100 celsius"
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([fake_seg], MagicMock())
    fake_fw = MagicMock()
    fake_fw.WhisperModel.return_value = fake_model

    with (
        patch(
            "src.shared.python.chat.voice_input_manager.detect_local_whisper_engine",
            return_value="faster-whisper",
        ),
        patch("importlib.import_module", return_value=fake_fw),
    ):
        status = mgr.whisper_status()
        assert status["available"] is True
        assert status["engine"] == "faster-whisper"
        assert "faster-whisper" in mgr.tooltip_text()
        text = mgr.transcribe_wav_bytes(_make_wav_bytes())
        assert text == "set reactor temperature to 1100 celsius"


def test_ui_builder_configures_immediately_clickable_input() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "shared"
        / "python"
        / "chat"
        / "_qt"
        / "ui_builder.py"
    ).read_text(encoding="utf-8")
    assert "view.input_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)" in source
    assert "view.input_edit.setEnabled(True)" in source
    assert "view.input_edit.setReadOnly(False)" in source
