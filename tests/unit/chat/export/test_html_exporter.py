"""RED tests for HtmlExporter (Tools issue #2735)."""

from __future__ import annotations

import re

import pytest
from chat.export import ChatExportRequest, HtmlExporter
from chat.service_base import ChatSession


def test_html_export_is_single_self_contained_file(tmp_path) -> None:
    session = ChatSession(session_id="s1")
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    out = tmp_path / "out.html"
    HtmlExporter().export(
        session,
        ChatExportRequest(
            session_id=session.session_id,
            format="html",
            output_path=str(out),
        ),
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    # No external <link> tags
    assert not re.search(r"<link[^>]*rel=['\"]stylesheet['\"][^>]*>", text)
    # No external scripts
    assert not re.search(r"<script[^>]*src=", text)
    # CSS is inlined in <style>
    assert "<style>" in text
    assert "</style>" in text


def test_html_export_escapes_user_content(tmp_path) -> None:
    session = ChatSession(session_id="s1")
    session.add_message("user", "<script>alert('xss')</script>")
    out = tmp_path / "esc.html"
    HtmlExporter().export(
        session,
        ChatExportRequest(
            session_id=session.session_id,
            format="html",
            output_path=str(out),
        ),
    )
    text = out.read_text(encoding="utf-8")
    # Raw <script>alert(...)</script> from message must be escaped
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text


def test_html_export_empty_session_raises(tmp_path) -> None:
    session = ChatSession(session_id="empty")
    with pytest.raises(ValueError):
        HtmlExporter().export(
            session,
            ChatExportRequest(
                session_id="empty",
                format="html",
                output_path=str(tmp_path / "x.html"),
            ),
        )
