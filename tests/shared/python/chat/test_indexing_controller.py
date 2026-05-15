"""Unit tests for IndexingController."""

from __future__ import annotations

import sys
import types
from logging import getLogger
from pathlib import Path
from unittest.mock import MagicMock

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

pytest.importorskip("PyQt6.QtCore")


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


def _make_controller(qapp):
    from src.shared.python.ai.gui._indexing import IndexingController

    return IndexingController(MagicMock())


def test_finished_signal_emitted_with_count(qapp):
    ctrl = _make_controller(qapp)
    out: list[int] = []
    ctrl.finished.connect(out.append)
    ctrl._on_finished(42)  # noqa: SLF001 - exercising lifecycle handler
    assert out == [42]
    assert ctrl.worker is None


def test_failed_signal_emitted_with_message(qapp):
    ctrl = _make_controller(qapp)
    errs: list[str] = []
    ctrl.failed.connect(errs.append)
    ctrl._on_error("boom")  # noqa: SLF001
    assert errs == ["boom"]


def test_status_changed_on_completion(qapp):
    ctrl = _make_controller(qapp)
    statuses: list[str] = []
    ctrl.status_changed.connect(statuses.append)
    ctrl._on_finished(7)  # noqa: SLF001
    assert any("Index ready" in s for s in statuses)


def test_system_message_on_error(qapp):
    ctrl = _make_controller(qapp)
    msgs: list[str] = []
    ctrl.system_message.connect(msgs.append)
    ctrl._on_error("x")  # noqa: SLF001
    assert any("indexing failed" in m for m in msgs)


def test_is_running_false_when_no_worker(qapp):
    ctrl = _make_controller(qapp)
    assert ctrl.is_running is False
