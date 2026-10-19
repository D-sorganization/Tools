"""Owner-parented single-shot timers (Tools #5667 follow-up).

A bare ``QTimer.singleShot(msec, lambda: widget.x())`` is owned by nobody:
when ``widget`` is deleted first the lambda still fires and raises
``RuntimeError: wrapped C/C++ object ... has been deleted`` inside the Qt
event loop of whichever unrelated test runs next on the same xdist worker.

Every check here is structural (``findChildren`` / ``sip.isdeleted``) or
drives ``timeout`` synchronously via ``emit()`` -- the event loop is never
spun, so timers leaked by other tests on the worker cannot fire in here.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 required")

from PyQt6 import sip  # noqa: E402
from PyQt6.QtCore import QObject, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

from shared.python.ui.hover_copy_browser import HoverCopyTextBrowser  # noqa: E402
from shared.python.ui.owned_timer import (  # noqa: E402
    restart_owned_single_shot,
    start_owned_single_shot,
)


@pytest.fixture
def owner(qapp: QApplication) -> Any:
    obj = QObject()
    yield obj
    if not sip.isdeleted(obj):
        sip.delete(obj)


# ── start_owned_single_shot ─────────────────────────────────────────


def test_start_creates_active_single_shot_child(owner: QObject) -> None:
    timer = start_owned_single_shot(owner, 1500, lambda: None)

    assert isinstance(timer, QTimer)
    assert timer.parent() is owner
    assert owner.findChildren(QTimer) == [timer]
    assert timer.isActive()
    assert timer.isSingleShot()
    assert timer.interval() == 1500


def test_start_timer_is_deleted_with_owner(owner: QObject) -> None:
    timer = start_owned_single_shot(owner, 1000, lambda: None)

    sip.delete(owner)

    assert sip.isdeleted(timer)


def test_start_timeout_runs_callback(owner: QObject) -> None:
    calls: list[str] = []
    timer = start_owned_single_shot(owner, 1000, lambda: calls.append("fired"))

    timer.timeout.emit()

    assert calls == ["fired"]


def test_start_stacks_independent_timers(owner: QObject) -> None:
    first = start_owned_single_shot(owner, 1000, lambda: None)
    second = start_owned_single_shot(owner, 1000, lambda: None)

    assert first is not second
    assert len(owner.findChildren(QTimer)) == 2


def test_start_accepts_zero_delay(owner: QObject) -> None:
    timer = start_owned_single_shot(owner, 0, lambda: None)

    assert timer.interval() == 0
    assert timer.isActive()


@pytest.mark.parametrize(
    ("make_args", "error", "match"),
    [
        (lambda o: (None, 10, lambda: None), TypeError, "owner"),
        (lambda o: (object(), 10, lambda: None), TypeError, "owner"),
        (lambda o: (o, -1, lambda: None), ValueError, "msec"),
        (lambda o: (o, 1.5, lambda: None), TypeError, "msec"),
        (lambda o: (o, 10, "not callable"), TypeError, "callback"),
    ],
)
def test_start_rejects_bad_arguments(
    owner: QObject, make_args: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        start_owned_single_shot(*make_args(owner))
    assert owner.findChildren(QTimer) == []


# ── restart_owned_single_shot ───────────────────────────────────────


def test_restart_reuses_one_named_timer(owner: QObject) -> None:
    first = restart_owned_single_shot(owner, 5000, lambda: None, name="reset")
    second = restart_owned_single_shot(owner, 5000, lambda: None, name="reset")

    assert first is second
    assert owner.findChildren(QTimer) == [first]
    assert first.objectName() == "reset"
    assert first.isActive()
    assert first.isSingleShot()


def test_restart_latest_callback_wins(owner: QObject) -> None:
    calls: list[str] = []
    restart_owned_single_shot(owner, 5000, lambda: calls.append("old"), name="r")
    timer = restart_owned_single_shot(
        owner, 5000, lambda: calls.append("new"), name="r"
    )

    timer.timeout.emit()

    assert calls == ["new"]


def test_restart_timer_survives_firing_for_reuse(owner: QObject) -> None:
    calls: list[int] = []
    timer = restart_owned_single_shot(owner, 10, lambda: calls.append(1), name="r")
    timer.timeout.emit()

    again = restart_owned_single_shot(owner, 10, lambda: calls.append(2), name="r")

    assert again is timer
    assert not sip.isdeleted(timer)
    assert again.isActive()


def test_restart_distinct_names_are_distinct_timers(owner: QObject) -> None:
    a = restart_owned_single_shot(owner, 10, lambda: None, name="a")
    b = restart_owned_single_shot(owner, 10, lambda: None, name="b")

    assert a is not b
    assert len(owner.findChildren(QTimer)) == 2


def test_restart_timer_is_deleted_with_owner(owner: QObject) -> None:
    timer = restart_owned_single_shot(owner, 10, lambda: None, name="r")

    sip.delete(owner)

    assert sip.isdeleted(timer)


@pytest.mark.parametrize(
    ("make_args", "error", "match"),
    [
        (lambda o: ((None, 10, lambda: None), {"name": "r"}), TypeError, "owner"),
        (lambda o: ((o, -5, lambda: None), {"name": "r"}), ValueError, "msec"),
        (lambda o: ((o, 10, 42), {"name": "r"}), TypeError, "callback"),
        (lambda o: ((o, 10, lambda: None), {"name": ""}), ValueError, "name"),
    ],
)
def test_restart_rejects_bad_arguments(
    owner: QObject, make_args: Any, error: type[Exception], match: str
) -> None:
    args, kwargs = make_args(owner)
    with pytest.raises(error, match=match):
        restart_owned_single_shot(*args, **kwargs)
    assert owner.findChildren(QTimer) == []


# ── site: HoverCopyTextBrowser.copy_all_text ────────────────────────


def test_hover_copy_restore_timer_dies_with_widget(qapp: QApplication) -> None:
    widget = HoverCopyTextBrowser()
    widget.setPlainText("payload")
    widget.copy_all_text()

    timers = widget.copy_btn.findChildren(QTimer)
    assert len(timers) == 1
    restore_timer = timers[0]
    assert restore_timer.isActive()
    assert restore_timer.isSingleShot()

    sip.delete(widget)
    assert sip.isdeleted(restore_timer)


def test_hover_copy_restore_timer_restores_button(qapp: QApplication) -> None:
    widget = HoverCopyTextBrowser()
    try:
        widget.setPlainText("payload")
        btn: QPushButton = widget.copy_btn
        orig_text = btn.text()
        widget.copy_all_text()
        assert btn.toolTip() == "Copied!"

        btn.findChildren(QTimer)[0].timeout.emit()

        assert btn.toolTip() == "Copy to clipboard"
        assert btn.text() == orig_text
    finally:
        sip.delete(widget)
