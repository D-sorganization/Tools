"""Low-priority bootstrap for the CPU-bound plot subprocess."""

from __future__ import annotations

import os


def _lower_process_priority() -> None:
    """Yield CPU scheduling priority to the interactive parent application."""
    if os.name == "nt":
        import ctypes

        below_normal_priority_class = 0x00004000
        windll = vars(ctypes).get("windll")
        if windll is None:  # pragma: no cover - guarded by os.name
            return
        kernel32 = windll.kernel32
        kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(), below_normal_priority_class
        )
        return
    try:
        nice = getattr(os, "nice", None)
        if callable(nice):
            nice(5)
    except OSError:
        # Sandboxed Unix runners may deny nice changes; thread caps still apply.
        pass


if __name__ == "__main__":
    _lower_process_priority()
    from rate_of_closure.ui.pyqt6.plots_process_worker import _worker_main

    raise SystemExit(_worker_main())
