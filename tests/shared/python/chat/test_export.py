"""Focused coverage for pure chat export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from chat.export import (
    ChatExportRequest,
    ChatExportResult,
    HtmlExporter,
    MarkdownExporter,
    MessageClipboardCopier,
    SecretRedactor,
    TextExporter,
)
from chat.export.copy_clipboard import _QtClipboardAdapter
from chat.service_base import ChatMessage, ChatSession

pytestmark = pytest.mark.unit


def _fake_openai_key() -> str:
    return "sk-" + ("a" * 16)


def _fake_github_token() -> str:
    return "ghp_" + "123456789012345678901"


def _fake_aws_key() -> str:
    return "AKIA" + ("A" * 16)


def _fake_stripe_key() -> str:
    return "sk_live_" + ("b" * 16)


def _fake_jwt() -> str:
    return "eyJabcdefghijk" + ".abcde" + ".abcdef"


def _session() -> ChatSession:
    session = ChatSession(
        session_id="session-export",
        metadata={"engine": "codex", "project": "Tools"},
    )
    session.add_message(
        "user",
        f"# Heading\nUse `code` and **bold** with {_fake_openai_key()}",
        timestamp=1_700_000_000.0,
    )
    session.add_message(
        "assistant",
        "Here is a response.",
        timestamp=1_700_000_001.0,
    )
    session.add_message(
        "tool",
        '{"ok": true}',
        timestamp=1_700_000_002.0,
        tool_call_id="call-1",
        metadata={"tool_name": "shell"},
    )
    return session


def test_export_contracts_validate_boundary_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session_id"):
        ChatExportRequest("", "markdown", str(tmp_path / "out.md"))
    with pytest.raises(ValueError, match="output_path"):
        ChatExportRequest("session", "markdown", " ")
    with pytest.raises(ValueError, match="format"):
        ChatExportRequest("session", "pdf", str(tmp_path / "out.pdf"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="byte_count"):
        ChatExportResult("out.md", -1, 1)
    with pytest.raises(ValueError, match="message_count"):
        ChatExportResult("out.md", 1, -1)


def test_secret_redactor_redacts_known_patterns_and_preserves_pinned_message() -> None:
    redactor = SecretRedactor()
    text = (
        f"Bearer abc.def {_fake_github_token()} "
        f"{_fake_aws_key()} {_fake_stripe_key()} "
        f"{_fake_openai_key()} {_fake_jwt()}"
    )

    redacted = redactor.redact(text)

    assert redacted.count("[REDACTED]") == 6
    with pytest.raises(TypeError, match="expects a str"):
        redactor.redact(None)  # type: ignore[arg-type]

    pinned = ChatMessage("user", _fake_openai_key(), metadata={"pin": True})
    assert redactor.redact_message(pinned) is pinned

    message = ChatMessage("user", f"token {_fake_openai_key()}")
    assert redactor.redact_message(message).content == "token [REDACTED]"


def test_markdown_exporter_writes_metadata_tool_calls_and_redacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "chat.md"
    request = ChatExportRequest(
        "session-export",
        "markdown",
        str(output),
        include_metadata=True,
        redact_secrets=True,
    )

    result = MarkdownExporter().export(_session(), request)

    text = output.read_text(encoding="utf-8")
    assert result.message_count == 3
    assert result.byte_count == output.stat().st_size
    assert "<!-- chat session metadata -->" in text
    assert "- engine: codex" in text
    assert "Tool call: shell (call-1)" in text
    assert _fake_openai_key() not in text
    assert "[REDACTED]" in text


def test_text_exporter_strips_markdown_and_handles_bad_timestamps(
    tmp_path: Path,
) -> None:
    session = _session()
    session.messages[0].timestamp = float("inf")
    output = tmp_path / "chat.txt"
    request = ChatExportRequest(
        "session-export",
        "text",
        str(output),
        include_metadata=True,
    )

    result = TextExporter().export(session, request)

    text = output.read_text(encoding="utf-8")
    assert result.message_count == 3
    assert "session_id: session-export" in text
    assert "# Heading" not in text
    assert "Heading" in text
    assert "`code`" not in text
    assert "code and bold" in text
    assert "[user inf]" in text


def test_html_exporter_escapes_content_metadata_and_tool_call(
    tmp_path: Path,
) -> None:
    session = _session()
    session.metadata["unsafe"] = "<script>alert(1)</script>"
    session.messages[1].content = "<b>assistant</b>"
    output = tmp_path / "chat.html"
    request = ChatExportRequest(
        "session-export",
        "html",
        str(output),
        include_metadata=True,
        redact_secrets=True,
    )

    result = HtmlExporter().export(session, request)

    html = output.read_text(encoding="utf-8")
    assert result.message_count == 3
    assert "<!DOCTYPE html>" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;assistant&lt;/b&gt;" in html
    assert "Tool call: shell (call-1)" in html
    assert _fake_openai_key() not in html
    assert "[REDACTED]" in html


@pytest.mark.parametrize(
    "exporter, extension",
    [
        (MarkdownExporter(), "md"),
        (TextExporter(), "txt"),
        (HtmlExporter(), "html"),
    ],
)
def test_exporters_reject_empty_sessions(
    tmp_path: Path,
    exporter: object,
    extension: str,
) -> None:
    request = ChatExportRequest("empty", "markdown", str(tmp_path / f"out.{extension}"))
    with pytest.raises(ValueError, match="empty chat session"):
        exporter.export(ChatSession(session_id="empty"), request)  # type: ignore[attr-defined]


class _FakeClipboard:
    def __init__(self) -> None:
        self.text = ""

    def set_text(self, text: str) -> None:
        self.text = text


def test_clipboard_copier_modes_and_validation() -> None:
    clipboard = _FakeClipboard()
    copier = MessageClipboardCopier(clipboard)
    message = ChatMessage(
        "assistant",
        "Before\n```python\nprint('hi')\n```\nAfter",
        timestamp=1_700_000_000.0,
        metadata={"source": "unit"},
    )

    assert copier.copy_message(message, "raw_text") == message.content
    markdown = copier.copy_message(message, "markdown")
    assert markdown.startswith("## assistant - ")
    assert "Before" in markdown
    assert copier.copy_message(message, "code_only") == "print('hi')\n"
    payload = json.loads(copier.copy_message(message, "json"))
    assert payload["metadata"] == {"source": "unit"}
    with pytest.raises(ValueError, match="unknown mode"):
        copier.copy_message(message, "xml")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clipboard_writer"):
        MessageClipboardCopier(None)  # type: ignore[arg-type]


def test_qt_clipboard_adapter_delegates_and_rejects_missing_setter() -> None:
    class QtClipboard:
        def __init__(self) -> None:
            self.text = ""

        def setText(self, value: str) -> None:  # noqa: N802 - Qt API name
            self.text = value

    qt_clipboard = QtClipboard()
    _QtClipboardAdapter(qt_clipboard).set_text("copied")
    assert qt_clipboard.text == "copied"

    with pytest.raises(RuntimeError, match="setText"):
        _QtClipboardAdapter(object()).set_text("boom")
