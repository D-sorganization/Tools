"""RED tests for Tools issue #2872 conversation management API.

Covers ChatSessionManager additions:
- ``unarchive_session``
- ``is_archived``
- ``search_sessions``
- ``export_session`` (markdown + json)
- ``load_context_from``

All tests run without a display server (use ``tmp_path`` for storage).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod


_ensure_pkg("src", ROOT / "src")
_ensure_pkg("src.shared", ROOT / "src" / "shared")
_ensure_pkg("src.shared.python", ROOT / "src" / "shared" / "python")
_ensure_pkg("src.shared.python.ai", ROOT / "src" / "shared" / "python" / "ai")
_ensure_pkg(
    "src.shared.python.ai.gui",
    ROOT / "src" / "shared" / "python" / "ai" / "gui",
)

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = logging.getLogger
logging_config.setup_logging = lambda *args, **kwargs: None
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

pytest.importorskip("PyQt6.QtCore", reason="PyQt6.QtCore required for QObject")


def _load_module(name: str, rel_path: str):
    full = ROOT / "src" / "shared" / "python" / rel_path
    spec = importlib.util.spec_from_file_location(name, full)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_types_mod = _load_module("src.shared.python.ai.types", "ai/types.py")
ConversationContext = _types_mod.ConversationContext
_sm_mod = _load_module(
    "src.shared.python.ai.gui.session_manager", "ai/gui/session_manager.py"
)
ChatSessionManager = _sm_mod.ChatSessionManager


# ───────────────────────────── helpers ─────────────────────────────


def _make_session(
    manager,
    *,
    title: str,
    user_msg: str = "hello",
    assistant_msg: str = "hi back",
    archived: bool = False,
) -> str:
    ctx = ConversationContext()
    ctx.metadata["title"] = title
    ctx.metadata["archived"] = archived
    ctx.add_message("user", user_msg)
    ctx.add_message("assistant", assistant_msg)
    manager.save_session(ctx)
    return ctx.session_id


# ───────────────────────────── unarchive ───────────────────────────


class TestUnarchiveSession:
    def test_unarchive_session_clears_flag(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        sid = _make_session(mgr, title="Old chat", archived=True)
        assert mgr.is_archived(sid) is True

        mgr.unarchive_session(sid)

        assert mgr.is_archived(sid) is False

    def test_unarchive_unknown_id_raises_keyerror(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        with pytest.raises(KeyError):
            mgr.unarchive_session("does-not-exist")


# ───────────────────────────── is_archived ─────────────────────────


class TestIsArchived:
    def test_is_archived_returns_correct_flag(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        active = _make_session(mgr, title="Active", archived=False)
        archived = _make_session(mgr, title="Archived", archived=True)

        assert mgr.is_archived(active) is False
        assert mgr.is_archived(archived) is True

    def test_is_archived_unknown_id_raises_keyerror(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        with pytest.raises(KeyError):
            mgr.is_archived("nope")


# ───────────────────────────── search ──────────────────────────────


class TestSearchSessions:
    def test_search_sessions_matches_title_substring(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        target = _make_session(mgr, title="Pendulum tuning")
        _make_session(mgr, title="Reactor pressure")

        hits = mgr.search_sessions("pendulum")

        ids = [h["id"] for h in hits]
        assert target in ids
        assert len(ids) == 1

    def test_search_sessions_matches_message_body(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        target = _make_session(
            mgr,
            title="Random",
            user_msg="What is the inertia tensor for a baseball bat?",
        )
        _make_session(mgr, title="Other", user_msg="Hello world")

        hits = mgr.search_sessions("baseball bat")

        ids = [h["id"] for h in hits]
        assert target in ids
        assert len(ids) == 1

    def test_search_sessions_excludes_archived_when_requested(
        self, tmp_path: Path
    ) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        active = _make_session(mgr, title="active alpha", archived=False)
        _make_session(mgr, title="archived alpha", archived=True)

        hits = mgr.search_sessions("alpha", include_archived=False)

        ids = [h["id"] for h in hits]
        assert active in ids
        assert len(ids) == 1

    def test_search_sessions_case_insensitive(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        sid = _make_session(mgr, title="MyChatSession")

        hits = mgr.search_sessions("mychat")

        ids = [h["id"] for h in hits]
        assert sid in ids


# ───────────────────────────── export ──────────────────────────────


class TestExportSession:
    def test_export_session_markdown_format(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        sid = _make_session(
            mgr,
            title="My Chat",
            user_msg="ping",
            assistant_msg="pong",
        )

        out = mgr.export_session(sid, "markdown")

        assert "# My Chat" in out
        assert "## user" in out or "**user**" in out
        assert "ping" in out
        assert "pong" in out

    def test_export_session_json_format(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        sid = _make_session(mgr, title="JSON Test")

        out = mgr.export_session(sid, "json")

        data = json.loads(out)
        assert data["session_id"] == sid
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_export_session_unknown_format_raises_valueerror(
        self, tmp_path: Path
    ) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        sid = _make_session(mgr, title="Doomed")
        with pytest.raises(ValueError):
            mgr.export_session(sid, "yaml")  # type: ignore[arg-type]

    def test_export_session_unknown_id_raises_keyerror(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        with pytest.raises(KeyError):
            mgr.export_session("nope", "markdown")


# ───────────────────────── load_context_from ───────────────────────


class TestLoadContextFrom:
    def test_load_context_from_concatenates_transcripts_in_order(
        self, tmp_path: Path
    ) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        first = _make_session(
            mgr,
            title="First",
            user_msg="FIRST_USER",
            assistant_msg="FIRST_ASSISTANT",
        )
        second = _make_session(
            mgr,
            title="Second",
            user_msg="SECOND_USER",
            assistant_msg="SECOND_ASSISTANT",
        )

        out = mgr.load_context_from([first, second])

        assert "FIRST_USER" in out
        assert "SECOND_USER" in out
        assert out.index("FIRST_USER") < out.index("SECOND_USER")

    def test_load_context_from_empty_list_returns_empty_string(
        self, tmp_path: Path
    ) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        assert mgr.load_context_from([]) == ""

    def test_load_context_from_unknown_id_raises_keyerror(self, tmp_path: Path) -> None:
        mgr = ChatSessionManager(storage_dir=tmp_path)
        sid = _make_session(mgr, title="Ok")
        with pytest.raises(KeyError):
            mgr.load_context_from([sid, "no-such-id"])
