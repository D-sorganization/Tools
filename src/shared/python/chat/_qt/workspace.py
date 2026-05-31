# ruff: noqa: E501
"""Workspace / plot slash-command bridge (Tools issue #2849).

Free helpers consumed by ``ChatDockWidget``. All routes take the chat-dock
instance explicitly so the parent module fits the repo's 1500-line budget.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .._workspace_protocol import WorkspaceVariableInfo

logger = logging.getLogger(__name__)


def build_workspace_context_block(provider: Any) -> str:
    """Return a bounded system-prompt fragment listing workspace vars.

    Returns an empty string when no provider is wired so the dock's
    outbound payload stays byte-for-byte identical to pre-#2849 shape.
    """
    if provider is None:
        return ""
    try:
        variables = provider.describe()
    except Exception:  # noqa: BLE001 - host adapter must not crash chat
        logger.exception("workspace provider describe() failed")
        return ""
    if not variables:
        return ""

    lines = ["Available workspace variables:"]
    for info in variables:
        if not isinstance(info, WorkspaceVariableInfo):
            continue
        shape_str = (
            ", ".join(str(dim) for dim in info.shape)
            if info.shape is not None
            else "scalar"
        )
        lines.append(
            f"- {info.name}: {info.dtype}, shape ({shape_str}), "
            f'preview="{info.preview}"'
        )
    return "\n".join(lines)


def handle_ws_read(dock: Any, arg: str) -> None:
    """``/ws.read NAME`` — read a workspace variable and show its preview."""
    name = arg.strip()
    if not name:
        dock._add_bubble("assistant", "Usage: /ws.read NAME")
        return
    provider = dock._workspace_provider
    if provider is None:
        dock._add_bubble("assistant", "Workspace bridge not available in this chat.")
        return
    try:
        value = provider.read(name)
    except KeyError:
        dock._add_bubble("assistant", f"Workspace variable not found: {name}")
        return
    except Exception as exc:  # noqa: BLE001 - host adapter errors
        logger.exception("workspace read failed for %s", name)
        dock._add_bubble("assistant", f"Workspace read failed: {exc}")
        return
    preview = repr(value)
    if len(preview) > 200:
        preview = preview[:197] + "..."
    dock._add_bubble("assistant", f"{name} = {preview}")


def handle_ws_write(dock: Any, arg: str) -> None:
    """``/ws.write NAME JSON_VALUE`` — write a value into the workspace."""
    parts = arg.split(maxsplit=1)
    if len(parts) != 2:
        dock._add_bubble("assistant", "Usage: /ws.write NAME JSON_VALUE")
        return
    name, raw_value = parts[0].strip(), parts[1].strip()
    if not name:
        dock._add_bubble("assistant", "Usage: /ws.write NAME JSON_VALUE")
        return
    provider = dock._workspace_provider
    if provider is None:
        dock._add_bubble("assistant", "Workspace bridge not available in this chat.")
        return
    try:
        value = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError) as exc:
        dock._add_bubble("assistant", f"Could not parse JSON value: {exc}")
        return
    try:
        provider.write(name, value)
    except TypeError as exc:
        dock._add_bubble("assistant", f"Workspace write rejected: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - host adapter errors
        logger.exception("workspace write failed for %s", name)
        dock._add_bubble("assistant", f"Workspace write failed: {exc}")
        return
    dock._add_bubble("assistant", f"Wrote workspace variable: {name}")


def handle_plot(dock: Any, arg: str) -> None:
    """``/plot {json}`` — forward a plot spec to the host sink."""
    spec_text = arg.strip()
    if not spec_text:
        dock._add_bubble("assistant", "Usage: /plot {json plot spec}")
        return
    sink = dock._plot_request_sink
    if sink is None:
        dock._add_bubble("assistant", "Plot tab not available in this chat.")
        return
    try:
        spec = json.loads(spec_text)
    except (json.JSONDecodeError, TypeError) as exc:
        dock._add_bubble("assistant", f"Could not parse plot spec JSON: {exc}")
        return
    try:
        sink(spec)
    except Exception as exc:  # noqa: BLE001 - host adapter errors
        logger.exception("plot request sink failed")
        dock._add_bubble("assistant", f"Plot request failed: {exc}")
        return
    dock._add_bubble("assistant", "Plot request submitted.")


def dispatch_workspace_command(dock: Any, cmd: str, arg: str) -> None:
    """Route ``/ws.read``, ``/ws.write`` and ``/plot`` slash commands."""
    if cmd == "ws.read":
        handle_ws_read(dock, arg)
        return
    if cmd == "ws.write":
        handle_ws_write(dock, arg)
        return
    if cmd == "plot":
        handle_plot(dock, arg)
        return
    raise ValueError(f"unknown workspace command: {cmd}")
