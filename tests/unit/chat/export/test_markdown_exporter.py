"""RED tests for MarkdownExporter (Tools issue #2735)."""

from __future__ import annotations

import pytest
from chat.export import ChatExportRequest, MarkdownExporter
from chat.service_base import ChatSession


def _make_session(messages: list[tuple[str, str]]) -> ChatSession:
    session = ChatSession(session_id="sess_abc123")
    for role, content in messages:
        session.add_message(role, content)
    return session


def test_export_basic_session_round_trip(tmp_path) -> None:
    session = _make_session(
        [
            ("user", "Hello"),
            ("assistant", "Hi there"),
        ]
    )
    exporter = MarkdownExporter()
    out = tmp_path / "out.md"
    request = ChatExportRequest(
        session_id=session.session_id,
        format="markdown",
        output_path=str(out),
    )
    result = exporter.export(session, request)
    assert result.path == str(out)
    assert result.message_count == 2
    text = out.read_text(encoding="utf-8")
    assert "## user" in text
    assert "## assistant" in text
    assert "Hello" in text
    assert "Hi there" in text


def test_export_preserves_code_fences_verbatim(tmp_path) -> None:
    fenced = "Here:\n```python\nx = 1\nprint(x)\n```\nDone."
    session = _make_session([("assistant", fenced)])
    exporter = MarkdownExporter()
    out = tmp_path / "code.md"
    result = exporter.export(
        session,
        ChatExportRequest(
            session_id=session.session_id,
            format="markdown",
            output_path=str(out),
        ),
    )
    assert result.message_count == 1
    text = out.read_text(encoding="utf-8")
    # Code fences appear verbatim including language hint
    assert "```python" in text
    assert "x = 1" in text
    assert "print(x)" in text
    # The closing fence remains
    assert text.count("```") >= 2


def test_export_renders_tool_calls_as_collapsible_details(tmp_path) -> None:
    session = ChatSession(session_id="s1")
    session.add_message(
        "tool",
        "Tool result payload",
        tool_call_id="call_42",
        metadata={"tool_name": "search"},
    )
    exporter = MarkdownExporter()
    out = tmp_path / "tools.md"
    exporter.export(
        session,
        ChatExportRequest(
            session_id=session.session_id,
            format="markdown",
            output_path=str(out),
        ),
    )
    text = out.read_text(encoding="utf-8")
    assert "<details>" in text
    assert "</details>" in text
    assert "Tool result payload" in text


def test_export_empty_session_raises_value_error(tmp_path) -> None:
    session = ChatSession(session_id="empty")
    exporter = MarkdownExporter()
    with pytest.raises(ValueError):
        exporter.export(
            session,
            ChatExportRequest(
                session_id="empty",
                format="markdown",
                output_path=str(tmp_path / "x.md"),
            ),
        )


def test_export_redacts_secrets_when_requested(tmp_path) -> None:
    # Literal split across operands so secret-scanning does not flag it.
    fake_key = "sk-" + "ABCD1234EFGH5678" + "IJKL9012MNOP3456"
    session = _make_session([("user", f"My key is {fake_key}")])
    exporter = MarkdownExporter()
    out = tmp_path / "red.md"
    exporter.export(
        session,
        ChatExportRequest(
            session_id=session.session_id,
            format="markdown",
            output_path=str(out),
            redact_secrets=True,
        ),
    )
    text = out.read_text(encoding="utf-8")
    assert fake_key not in text
    assert "[REDACTED" in text


def test_export_includes_metadata_block_when_requested(tmp_path) -> None:
    session = _make_session([("user", "Hi")])
    session.metadata["topic"] = "demo"
    exporter = MarkdownExporter()
    out = tmp_path / "meta.md"
    exporter.export(
        session,
        ChatExportRequest(
            session_id=session.session_id,
            format="markdown",
            output_path=str(out),
            include_metadata=True,
        ),
    )
    text = out.read_text(encoding="utf-8")
    assert "session_id" in text
    assert "sess_" in text or "s1" in text or session.session_id in text
