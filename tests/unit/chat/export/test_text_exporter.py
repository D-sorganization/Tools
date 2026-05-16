"""RED tests for TextExporter (Tools issue #2735)."""

from __future__ import annotations

import pytest
from chat.export import ChatExportRequest, TextExporter
from chat.service_base import ChatSession


def test_text_exporter_strips_markdown(tmp_path) -> None:
    session = ChatSession(session_id="s1")
    session.add_message("user", "**bold** and `code` and # heading")
    out = tmp_path / "out.txt"
    TextExporter().export(
        session,
        ChatExportRequest(
            session_id=session.session_id,
            format="text",
            output_path=str(out),
        ),
    )
    text = out.read_text(encoding="utf-8")
    assert "**" not in text
    assert "`" not in text
    # heading hash removed
    assert "# heading" not in text
    assert "bold" in text
    assert "code" in text


def test_text_exporter_records_role_and_count(tmp_path) -> None:
    session = ChatSession(session_id="s2")
    session.add_message("user", "Q1")
    session.add_message("assistant", "A1")
    out = tmp_path / "out.txt"
    result = TextExporter().export(
        session,
        ChatExportRequest(
            session_id=session.session_id,
            format="text",
            output_path=str(out),
        ),
    )
    assert result.message_count == 2
    text = out.read_text(encoding="utf-8")
    assert "user" in text.lower()
    assert "assistant" in text.lower()


def test_text_exporter_empty_session_raises(tmp_path) -> None:
    session = ChatSession(session_id="empty")
    with pytest.raises(ValueError):
        TextExporter().export(
            session,
            ChatExportRequest(
                session_id="empty",
                format="text",
                output_path=str(tmp_path / "x.txt"),
            ),
        )
