"""Helpers for surfacing failures from UI background threads."""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def start_background_thread(
    owner: Any,
    target: Callable[..., None],
    *,
    name: str,
    on_error: Callable[[BaseException, str], None],
    args: tuple[Any, ...] = (),
) -> threading.Thread:
    """Start a daemon thread and route unexpected failures to the UI thread."""

    def run_guarded() -> None:
        try:
            target(*args)
        except Exception as exc:
            formatted_traceback = traceback.format_exc()
            logger.exception("Unhandled exception in background UI task %s", name)
            owner.after(
                0,
                lambda e=exc, tb=formatted_traceback: on_error(e, tb),
            )

    thread = threading.Thread(target=run_guarded, name=name, daemon=True)
    thread.start()
    return thread
