"""Owner-parented single-shot timers that cannot outlive their widget.

``QTimer.singleShot(msec, callback)`` with a lambda, ``functools.partial`` or
closure is owned by nobody: if the widget the callback touches is deleted
before the delay elapses, the callback still runs and raises
``RuntimeError: wrapped C/C++ object ... has been deleted`` inside whatever
event loop happens to be spinning (Tools #5667). Qt only auto-disconnects a
``singleShot`` whose slot is a *bound method of a live QObject*; every other
callable needs an owner.

The helpers here parent the timer to ``owner``. Deleting ``owner`` deletes
the timer, so the callback can never fire against a dead widget. Pass the
widget the callback touches (or one of its ancestors) as ``owner``.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, Qt, QTimer

__all__ = ["restart_owned_single_shot", "start_owned_single_shot"]

_DIRECT_CHILDREN = Qt.FindChildOption.FindDirectChildrenOnly


def _require_timer_args(owner: object, msec: object, callback: object) -> None:
    """Validate the arguments shared by both helpers.

    Raises:
        TypeError: ``owner`` is not a ``QObject``, ``msec`` is not an ``int``
            or ``callback`` is not callable.
        ValueError: ``msec`` is negative.
    """
    if not isinstance(owner, QObject):
        raise TypeError(f"owner must be a QObject, got {type(owner).__name__}")
    if isinstance(msec, bool) or not isinstance(msec, int):
        raise TypeError(f"msec must be an int, got {type(msec).__name__}")
    if msec < 0:
        raise ValueError(f"msec must be >= 0, got {msec}")
    if not callable(callback):
        raise TypeError(f"callback must be callable, got {type(callback).__name__}")


def start_owned_single_shot(
    owner: QObject, msec: int, callback: Callable[[], object]
) -> QTimer:
    """Run ``callback`` once after ``msec`` ms unless ``owner`` dies first.

    Each call creates an independent timer that deletes itself after firing.
    Use :func:`restart_owned_single_shot` when a newer request should replace
    a pending one instead of stacking another.

    Args:
        owner: QObject that parents (and therefore outlives) the timer.
        msec: Delay in milliseconds, ``>= 0``.
        callback: Zero-argument callable run on timeout.

    Returns:
        The started, single-shot timer (a child of ``owner``).

    Raises:
        TypeError: On a non-QObject owner, non-int delay or non-callable.
        ValueError: On a negative delay.
    """
    _require_timer_args(owner, msec, callback)
    timer = QTimer(owner)
    timer.setSingleShot(True)
    timer.timeout.connect(callback)
    timer.timeout.connect(timer.deleteLater)
    timer.start(msec)
    return timer


def restart_owned_single_shot(
    owner: QObject, msec: int, callback: Callable[[], object], *, name: str
) -> QTimer:
    """(Re)start the single owned timer called ``name`` on ``owner``.

    The first call creates the timer; later calls with the same ``name``
    reuse it, replace its callback (the latest one wins) and restart the
    countdown, so repeated requests never queue stale callbacks.

    Args:
        owner: QObject that parents (and therefore outlives) the timer.
        msec: Delay in milliseconds, ``>= 0``.
        callback: Zero-argument callable run on timeout.
        name: Non-empty object name identifying the timer among the
            owner's direct children.

    Returns:
        The running, single-shot timer (a child of ``owner``).

    Raises:
        TypeError: On a non-QObject owner, non-int delay or non-callable.
        ValueError: On a negative delay or an empty ``name``.
    """
    _require_timer_args(owner, msec, callback)
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    existing = owner.findChildren(QTimer, name, _DIRECT_CHILDREN)
    if existing:
        timer = existing[0]
        timer.timeout.disconnect()
    else:
        timer = QTimer(owner)
        timer.setObjectName(name)
        timer.setSingleShot(True)
    timer.timeout.connect(callback)
    timer.start(msec)
    return timer
