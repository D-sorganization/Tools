"""Unit tests for MessageDisplayController."""

from __future__ import annotations

import sys
import types
from logging import getLogger
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(ROOT / "src")]
sys.modules.setdefault("src", src_pkg)

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = getLogger
logging_config.setup_logging = lambda *a, **k: None
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

config_pkg = types.ModuleType("src.shared.python.config")
environment = types.ModuleType("src.shared.python.config.environment")
environment.get_env = lambda _name, default=None, **_k: default
environment.get_env_float = lambda _name, default=None, **_k: default
sys.modules.setdefault("src.shared.python.config", config_pkg)
sys.modules.setdefault("src.shared.python.config.environment", environment)

pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def display(qapp):
    from src.shared.python.ai.gui._message_display import MessageDisplayController

    return MessageDisplayController()


def test_initial_layout_has_only_stretch(display):
    # The trailing stretch is a non-widget item; count() == 1 means just stretch.
    assert display.message_layout.count() == 1


def test_add_message_inserts_widget(display):
    w = display.add_message("user", "hello")
    assert w is not None
    # one widget + stretch
    assert display.message_layout.count() == 2


def test_add_system_message_emits_signal(display):
    received: list = []
    display.message_added.connect(received.append)
    w = display.add_system_message("welcome")
    assert received and received[-1] is w


def test_clear_messages_keeps_only_stretch(display):
    display.add_message("user", "a")
    display.add_message("assistant", "b")
    display.add_system_message("c")
    assert display.message_layout.count() == 4

    display.clear_messages()
    assert display.message_layout.count() == 1


def test_restore_from_context_skips_system(display):
    from src.shared.python.ai.types import ConversationContext

    ctx = ConversationContext()
    ctx.add_user_message("hi")
    ctx.add_assistant_message("hello back")
    # add_user_message / add_assistant_message both push real (non-system) items
    display.restore_from_context(ctx)
    assert display.message_layout.count() == 1 + len(ctx.messages)
