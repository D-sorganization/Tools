"""Standalone PyQt launcher with bounded Morris authority ownership."""

from __future__ import annotations

import logging
from typing import Any

from rate_of_closure.application.morris.client import MorrisAuthorityClient
from rate_of_closure.application.morris.runtime import MorrisAuthorityRuntime
from shared.python.gui_launcher import GUIType, LaunchConfig, launch_pyqt6_app

logger = logging.getLogger(__name__)


def _launch_config(morris_client: Any) -> LaunchConfig:
    """Return the registered app config with an explicit private-client seam."""
    from rate_of_closure.gui_registration import GUI_INFO

    pyqt = GUI_INFO["pyqt6"]
    minimum = pyqt.get("min_size")
    return LaunchConfig(
        tool_name=str(GUI_INFO["tool_name"]),
        gui_type=GUIType.PYQT6,
        module_path=str(pyqt["module"]),
        class_name=str(pyqt["class"]),
        dependencies=list(pyqt.get("dependencies", ())),
        title=str(GUI_INFO["name"]),
        settings_app=str(pyqt.get("settings_app", "RateOfClosure")),
        min_size=tuple(minimum) if minimum else None,
        window_kwargs={"morris_client": morris_client},
    )


def launch_rate_pyqt6() -> int:
    """Launch the app and own its authority for exactly the Qt event loop."""
    runtime: MorrisAuthorityRuntime | None = None
    client: MorrisAuthorityClient | None = None
    try:
        runtime = MorrisAuthorityRuntime.start()
        client = MorrisAuthorityClient(
            runtime.base_url,
            runtime.authorization_headers,
        )
    except (ImportError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        logger.warning("Morris Screening unavailable: %s", exc)
    try:
        return int(launch_pyqt6_app(_launch_config(client)))
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                logger.exception(
                    "Morris authority cleanup failed after the Qt event loop"
                )


__all__ = ["launch_rate_pyqt6"]
