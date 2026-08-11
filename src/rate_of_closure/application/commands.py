"""Stable application-command identifiers and availability semantics.

The contract deliberately contains no Qt, React, menu, or toolbar concepts.
Every client binds these stable wire identifiers to its own presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum


class AppCommandId(StrEnum):
    """Stable command identifiers shared by desktop, web, and automation."""

    FILE_NEW_WORKSPACE = "file.new_workspace"
    FILE_OPEN_WORKSPACE = "file.open_workspace"
    FILE_OPEN_RECENT_WORKSPACE = "file.open_recent_workspace"
    FILE_SAVE_WORKSPACE = "file.save_workspace"
    FILE_SAVE_WORKSPACE_AS = "file.save_workspace_as"
    FILE_IMPORT_WORKSPACE = "file.import_workspace"
    FILE_EXPORT_WORKSPACE = "file.export_workspace"
    FILE_CLOSE_WORKSPACE = "file.close_workspace"
    FILE_OPEN_REGIONAL_GROUND_VARIATION_REQUEST = (
        "file.open_regional_ground_variation_request"
    )
    FILE_SAVE_REGIONAL_GROUND_VARIATION_REQUEST_AS = (
        "file.save_regional_ground_variation_request_as"
    )
    VIEW_MANAGE_MODULES = "view.manage_modules"
    VIEW_RESTORE_DEFAULT_WORKSPACE = "view.restore_default_workspace"
    VIEW_SHOW_IMPACT = "view.show_impact"
    VIEW_SHOW_SWING = "view.show_swing"
    VIEW_SHOW_FLIGHT = "view.show_flight"
    GLOBAL_OPEN_GLOSSARY = "global.open_glossary"
    GLOBAL_TOGGLE_THEME = "global.toggle_theme"
    GLOBAL_SHOW_SHORTCUTS = "global.show_shortcuts"
    GLOBAL_OPEN_CURRENT_MODULE_HELP = "global.open_current_module_help"


APP_COMMAND_IDS: tuple[AppCommandId, ...] = tuple(AppCommandId)
"""Canonical command order for registries and parity checks."""


class CommandUnavailableError(RuntimeError):
    """Raised when a caller attempts to invoke a disabled command."""

    def __init__(self, command_id: AppCommandId, reason: str) -> None:
        """Record the rejected command and its actionable reason."""
        self.command_id = command_id
        self.reason = reason
        super().__init__(f"{command_id.value} is unavailable: {reason}")


@dataclass(frozen=True)
class CommandAvailability:
    """One unambiguous command-enabled state.

    Invariant:
        Enabled commands have no disabled reason. Disabled commands always
        expose a non-empty reason suitable for status text or a tooltip.
    """

    enabled: bool
    disabled_reason: str | None

    def __post_init__(self) -> None:
        """Reject states whose boolean and reason disagree."""
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if self.enabled and self.disabled_reason is not None:
            raise ValueError("enabled commands cannot have a disabled reason")
        if not self.enabled:
            reason = self.disabled_reason
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("disabled commands require a non-empty reason")
            object.__setattr__(self, "disabled_reason", reason.strip())

    @classmethod
    def available(cls) -> CommandAvailability:
        """Return the canonical enabled state."""
        return cls(enabled=True, disabled_reason=None)

    @classmethod
    def disabled(cls, reason: str) -> CommandAvailability:
        """Return a disabled state carrying an actionable reason."""
        return cls(enabled=False, disabled_reason=reason)

    def require_enabled(self, command_id: AppCommandId) -> None:
        """Raise with stable command context when this state is disabled.

        Args:
            command_id: Command the caller intends to invoke.

        Raises:
            TypeError: If ``command_id`` is not an :class:`AppCommandId`.
            CommandUnavailableError: If the command is disabled.
        """
        if not isinstance(command_id, AppCommandId):
            raise TypeError("command_id must be an AppCommandId")
        if self.enabled:
            return
        reason = self.disabled_reason
        if reason is None:
            raise RuntimeError("disabled command state is missing disabled_reason")
        raise CommandUnavailableError(command_id, reason)


__all__ = [
    "APP_COMMAND_IDS",
    "AppCommandId",
    "CommandAvailability",
    "CommandUnavailableError",
]
