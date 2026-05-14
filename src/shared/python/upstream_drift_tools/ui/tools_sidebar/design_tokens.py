"""Design tokens and stylesheet bridge for the Sidekick sidebar."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

SIDEKICK_SIDEBAR_OBJECT_NAME = "SidekickToolsSidebar"
SIDEKICK_DOCK_OBJECT_NAME = "SidekickToolsSidebarDock"
SIDEKICK_TABS_OBJECT_NAME = "SidekickTabs"
SIDEKICK_TAB_BAR_OBJECT_NAME = "SidekickTabBar"
SIDEKICK_TOOLBAR_OBJECT_NAME = "SidekickToolbar"
SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME = "SidekickProjectExplorer"
SIDEKICK_PROJECT_TREE_OBJECT_NAME = "SidekickProjectTree"
SIDEKICK_WORKSPACE_TAB_OBJECT_NAME = "SidekickWorkspaceTab"
SIDEKICK_WORKSPACE_LIST_OBJECT_NAME = "SidekickWorkspaceList"
SIDEKICK_PLACEHOLDER_OBJECT_NAME = "SidekickPlaceholder"
SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME = "SidekickPlaceholderLabel"

_DEFAULT_TOKEN_VALUES: dict[str, str] = {
    "color.background": "#f7f9fc",
    "color.surface": "#ffffff",
    "color.surface.raised": "#eef3f8",
    "color.border": "#d8e1ec",
    "color.border.strong": "#b8c6d8",
    "color.text": "#1c2430",
    "color.text.muted": "#5d6b7c",
    "color.accent": "#2563eb",
    "color.accent.hover": "#1d4ed8",
    "color.accent.soft": "#dbeafe",
    "color.focus": "#3b82f6",
    "color.selection": "#c7dfff",
    "color.danger": "#dc2626",
    "color.warning": "#d97706",
    "color.success": "#16a34a",
    "radius.panel": "8px",
    "radius.control": "6px",
    "shadow.panel": "0 12px 28px rgba(15, 23, 42, 0.14)",
    "space.1": "4px",
    "space.2": "8px",
    "space.3": "12px",
    "space.4": "16px",
    "size.control.height": "28px",
    "font.size": "12px",
    "font.size.small": "11px",
}

SIDEKICK_TOKEN_NAMES: tuple[str, ...] = tuple(_DEFAULT_TOKEN_VALUES)

_SIDEKICK_TOKEN_ALIASES: dict[str, str] = {
    "sidekick.color.canvas": "color.background",
    "sidekick.color.background": "color.background",
    "sidekick.color.surface": "color.surface",
    "sidekick.color.surface.elevated": "color.surface.raised",
    "sidekick.color.surface.raised": "color.surface.raised",
    "sidekick.color.surface.muted": "color.surface.raised",
    "sidekick.color.border": "color.border",
    "sidekick.color.border.strong": "color.border.strong",
    "sidekick.color.text": "color.text",
    "sidekick.color.text.muted": "color.text.muted",
    "sidekick.color.text.subtle": "color.text.muted",
    "sidekick.color.accent": "color.accent",
    "sidekick.color.accent.hover": "color.accent.hover",
    "sidekick.color.accent.soft": "color.accent.soft",
    "sidekick.color.focus": "color.focus",
    "sidekick.color.selection": "color.selection",
    "sidekick.color.error": "color.danger",
    "sidekick.color.danger": "color.danger",
    "sidekick.color.warning": "color.warning",
    "sidekick.color.success": "color.success",
    "sidekick.radius.lg": "radius.panel",
    "sidekick.radius.chat": "radius.panel",
    "sidekick.radius.md": "radius.control",
    "sidekick.radius.control": "radius.control",
    "sidekick.shadow.md": "shadow.panel",
    "sidekick.shadow.panel": "shadow.panel",
    "sidekick.control.height": "size.control.height",
    "sidekick.size.control.height": "size.control.height",
}


def _token_slug(name: str) -> str:
    return name.replace(".", "-").replace("_", "-")


def _normalize_token_values(values: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, value in values.items():
        token_name = _SIDEKICK_TOKEN_ALIASES.get(name, name)
        if token_name in _DEFAULT_TOKEN_VALUES:
            normalized[token_name] = value
    return normalized


@dataclass(frozen=True)
class SidekickDesignTokens:
    """Reusable Sidekick token set that can emit CSS and QSS mappings."""

    values: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        merged = {**_DEFAULT_TOKEN_VALUES, **_normalize_token_values(self.values)}
        missing = [name for name in SIDEKICK_TOKEN_NAMES if not merged.get(name)]
        if missing:
            missing_names = ", ".join(missing)
            raise ValueError(f"Missing Sidekick design tokens: {missing_names}")
        object.__setattr__(self, "values", MappingProxyType(merged))

    def __getitem__(self, name: str) -> str:
        return self.values[name]

    def css_variables(self, prefix: str = "sidekick") -> dict[str, str]:
        """Return browser CSS custom properties for the token set."""
        return {
            f"--{prefix}-{_token_slug(name)}": value
            for name, value in self.values.items()
        }

    def qss_variables(self, prefix: str = "sidekick") -> dict[str, str]:
        """Return QSS-friendly token names for stylesheet templating."""
        return {
            f"{prefix}-{_token_slug(name)}": value
            for name, value in self.values.items()
        }

    def with_overrides(self, **overrides: str) -> SidekickDesignTokens:
        """Return a copy with selected token values replaced."""
        return SidekickDesignTokens({**self.values, **overrides})

    @classmethod
    def from_sidekick_tokens(cls, tokens: Mapping[str, str]) -> SidekickDesignTokens:
        """Create tokens from host maps using canonical ``sidekick.*`` names."""
        return cls(tokens)


SIDEKICK_DESIGN_TOKENS = SidekickDesignTokens()


def sidekick_qss(tokens: SidekickDesignTokens | None = None) -> str:
    """Build the canonical Qt stylesheet for Sidekick widgets."""
    token_set = tokens or SIDEKICK_DESIGN_TOKENS
    value = token_set.__getitem__
    return f"""
QWidget#{SIDEKICK_SIDEBAR_OBJECT_NAME} {{
    background: {value("color.background")};
    color: {value("color.text")};
    font-size: {value("font.size")};
}}

QDockWidget#{SIDEKICK_DOCK_OBJECT_NAME} {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    border: 1px solid {value("color.border")};
}}

QToolBar#{SIDEKICK_TOOLBAR_OBJECT_NAME} {{
    background: {value("color.surface")};
    border: 0;
    border-bottom: 1px solid {value("color.border")};
    spacing: {value("space.1")};
    padding: {value("space.2")};
}}

QToolBar#{SIDEKICK_TOOLBAR_OBJECT_NAME} QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {value("radius.control")};
    color: {value("color.text.muted")};
    min-height: {value("size.control.height")};
    padding: {value("space.1")} {value("space.2")};
}}

QToolBar#{SIDEKICK_TOOLBAR_OBJECT_NAME} QToolButton:hover {{
    background: {value("color.accent.soft")};
    border-color: {value("color.border.strong")};
    color: {value("color.text")};
}}

QToolBar#{SIDEKICK_TOOLBAR_OBJECT_NAME} QToolButton:focus,
QTreeView#{SIDEKICK_PROJECT_TREE_OBJECT_NAME}:focus,
QListWidget#{SIDEKICK_WORKSPACE_LIST_OBJECT_NAME}:focus {{
    border: 1px solid {value("color.focus")};
}}

QTabWidget#{SIDEKICK_TABS_OBJECT_NAME}::pane {{
    border: 0;
    background: {value("color.surface")};
}}

QTabBar#{SIDEKICK_TAB_BAR_OBJECT_NAME}::tab {{
    background: {value("color.surface.raised")};
    border: 1px solid {value("color.border")};
    border-bottom: 0;
    border-top-left-radius: {value("radius.control")};
    border-top-right-radius: {value("radius.control")};
    color: {value("color.text.muted")};
    margin-right: {value("space.1")};
    padding: {value("space.2")} {value("space.3")};
}}

QTabBar#{SIDEKICK_TAB_BAR_OBJECT_NAME}::tab:selected {{
    background: {value("color.surface")};
    border-color: {value("color.border.strong")};
    color: {value("color.text")};
}}

QTreeView#{SIDEKICK_PROJECT_TREE_OBJECT_NAME},
QListWidget#{SIDEKICK_WORKSPACE_LIST_OBJECT_NAME} {{
    background: {value("color.surface")};
    border: 0;
    color: {value("color.text")};
    selection-background-color: {value("color.selection")};
    selection-color: {value("color.text")};
}}

QLabel#{SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME} {{
    color: {value("color.text.muted")};
    font-size: {value("font.size.small")};
    padding: {value("space.4")};
}}
""".strip()
