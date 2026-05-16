"""RED tests for MessageClipboardCopier (Tools issue #2735)."""

from __future__ import annotations

import json

import pytest
from chat.export.copy_clipboard import MessageClipboardCopier
from chat.service_base import ChatMessage


class _FakeClipboard:
    def __init__(self) -> None:
        self.last: str | None = None

    def set_text(self, text: str) -> None:
        self.last = text


def _msg(content: str = "Hello", role: str = "user") -> ChatMessage:
    return ChatMessage(role=role, content=content)


def test_copy_raw_text_mode() -> None:
    clip = _FakeClipboard()
    copier = MessageClipboardCopier(clipboard_writer=clip)
    copier.copy_message(_msg("Hello"), mode="raw_text")
    assert clip.last == "Hello"


def test_copy_markdown_mode_includes_role_block() -> None:
    clip = _FakeClipboard()
    copier = MessageClipboardCopier(clipboard_writer=clip)
    copier.copy_message(_msg("Hello", role="assistant"), mode="markdown")
    assert clip.last is not None
    assert clip.last.startswith("## ")
    assert "assistant" in clip.last
    assert "Hello" in clip.last


def test_copy_code_only_mode_extracts_fenced_blocks() -> None:
    clip = _FakeClipboard()
    copier = MessageClipboardCopier(clipboard_writer=clip)
    msg = _msg("Some text\n```python\nx = 1\n```\nMore text")
    copier.copy_message(msg, mode="code_only")
    assert clip.last is not None
    assert "x = 1" in clip.last
    assert "Some text" not in clip.last
    assert "More text" not in clip.last


def test_copy_code_only_no_fences_yields_empty_string() -> None:
    clip = _FakeClipboard()
    copier = MessageClipboardCopier(clipboard_writer=clip)
    copier.copy_message(_msg("Plain text only"), mode="code_only")
    assert clip.last == ""


def test_copy_json_mode_serialises_message_fields() -> None:
    clip = _FakeClipboard()
    copier = MessageClipboardCopier(clipboard_writer=clip)
    msg = _msg("Hi", role="user")
    copier.copy_message(msg, mode="json")
    assert clip.last is not None
    parsed = json.loads(clip.last)
    assert parsed["role"] == "user"
    assert parsed["content"] == "Hi"
    assert "timestamp" in parsed


def test_copy_unknown_mode_raises() -> None:
    clip = _FakeClipboard()
    copier = MessageClipboardCopier(clipboard_writer=clip)
    with pytest.raises(ValueError):
        copier.copy_message(_msg(), mode="nope")
