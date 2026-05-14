"""Unified tools sidebar public API.

The backend registry/state classes import without Qt. Widget classes are loaded
on demand so non-GUI hosts can still use persistence and workspace contracts.
"""

from __future__ import annotations

from .calculator_assist import (
    CALCULATOR_HELP,
    CalculatorPredictiveText,
    StaticCalculatorPredictionProvider,
)
from .calculator_plotting import (
    CALCULATOR_PLOT_TAB_ID,
    CalculatorPlotRequest,
    CalculatorPlotSource,
    CalculatorPlotTabConfig,
    build_calculator_plot_spec,
)
from .calculator_startup import (
    CalculatorStartupConfig,
    CalculatorStartupImport,
    CalculatorStartupResult,
    CalculatorStartupWarning,
    default_calculator_startup_config,
)
from .calculator_workspace import (
    CALCULATOR_WORKSPACE_SCOPE,
    CalculatorWorkspaceController,
    CalculatorWorkspaceLoadResult,
    CalculatorWorkspaceSettings,
    validate_calculator_workspace_path,
)
from .command_history import (
    DEFAULT_COMMAND_HISTORY_LIMIT,
    CommandHistoryController,
)
from .design_tokens import (
    SIDEKICK_DESIGN_TOKENS,
    SIDEKICK_DOCK_OBJECT_NAME,
    SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME,
    SIDEKICK_PLACEHOLDER_OBJECT_NAME,
    SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME,
    SIDEKICK_PROJECT_TREE_OBJECT_NAME,
    SIDEKICK_ROTATION_CONVERTER_OBJECT_NAME,
    SIDEKICK_SIDEBAR_OBJECT_NAME,
    SIDEKICK_TAB_BAR_OBJECT_NAME,
    SIDEKICK_TABS_OBJECT_NAME,
    SIDEKICK_TOKEN_NAMES,
    SIDEKICK_TOOLBAR_OBJECT_NAME,
    SIDEKICK_WORKSPACE_LIST_OBJECT_NAME,
    SIDEKICK_WORKSPACE_TAB_OBJECT_NAME,
    SidekickDesignTokens,
    SidekickTerminalTheme,
    sidekick_qss,
)
from .file_navigation import (
    CommonLocation,
    CommonLocationsProvider,
    DefaultCommonLocationsProvider,
    FileNavigationController,
    FileNavigationState,
)
from .registry import WorkspaceRegistry, WorkspaceVariable
from .state import SidebarState
from .state_profiles import (
    CLEAR_SIDEKICK_DATA_CONFIRMATION,
    CLEAR_SIDEKICK_DATA_WARNING,
    SidekickStateProfileResult,
    SidekickStateProfileStore,
    validate_profile_name,
)
from .theme_settings import (
    SidekickFontSettings,
    SidekickThemeMode,
    SidekickThemeSettings,
    resolve_sidekick_theme,
)

__all__ = [
    "SIDEKICK_DESIGN_TOKENS",
    "CommonLocation",
    "CommonLocationsProvider",
    "DEFAULT_COMMAND_HISTORY_LIMIT",
    "CALCULATOR_HELP",
    "CALCULATOR_WORKSPACE_SCOPE",
    "CalculatorStartupConfig",
    "CalculatorStartupImport",
    "CalculatorStartupResult",
    "CalculatorStartupWarning",
    "CalculatorPredictiveText",
    "CalculatorPlotRequest",
    "CalculatorPlotSource",
    "CalculatorPlotTabConfig",
    "CalculatorWorkspaceController",
    "CalculatorWorkspaceLoadResult",
    "CalculatorWorkspaceSettings",
    "CommandHistoryController",
    "DefaultCommonLocationsProvider",
    "FileNavigationController",
    "FileNavigationState",
    "ProjectFileExplorer",
    "SidebarState",
    "SidebarTabDefinition",
    "SIDEKICK_DOCK_OBJECT_NAME",
    "SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME",
    "SIDEKICK_PLACEHOLDER_OBJECT_NAME",
    "SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME",
    "SIDEKICK_PROJECT_TREE_OBJECT_NAME",
    "SIDEKICK_ROTATION_CONVERTER_OBJECT_NAME",
    "SIDEKICK_SIDEBAR_OBJECT_NAME",
    "SIDEKICK_TAB_BAR_OBJECT_NAME",
    "SIDEKICK_TABS_OBJECT_NAME",
    "SIDEKICK_TOKEN_NAMES",
    "SIDEKICK_TOOLBAR_OBJECT_NAME",
    "SIDEKICK_WORKSPACE_LIST_OBJECT_NAME",
    "SIDEKICK_WORKSPACE_TAB_OBJECT_NAME",
    "SidekickDesignTokens",
    "SidekickFontSettings",
    "SidekickTerminalTheme",
    "SidekickThemeMode",
    "SidekickThemeSettings",
    "StaticCalculatorPredictionProvider",
    "CALCULATOR_PLOT_TAB_ID",
    "CLEAR_SIDEKICK_DATA_CONFIRMATION",
    "CLEAR_SIDEKICK_DATA_WARNING",
    "SidekickSidebar",
    "SidekickStateProfileResult",
    "SidekickStateProfileStore",
    "ToolsSidebarInstallResult",
    "UnifiedToolsSidebar",
    "WorkspaceRegistry",
    "WorkspaceVariable",
    "build_calculator_plot_spec",
    "create_tools_sidebar",
    "default_calculator_startup_config",
    "install_tools_sidebar",
    "resolve_sidekick_theme",
    "sidekick_qss",
    "validate_calculator_workspace_path",
    "validate_profile_name",
]


def __getattr__(name: str) -> object:
    if name in {
        "UnifiedToolsSidebar",
        "SidekickSidebar",
        "SidebarTabDefinition",
    }:
        from . import sidebar

        return getattr(sidebar, name)
    if name in {
        "ToolsSidebarInstallResult",
        "create_tools_sidebar",
        "install_tools_sidebar",
    }:
        from . import api

        return getattr(api, name)
    if name == "ProjectFileExplorer":
        from .project_file_explorer import ProjectFileExplorer

        return ProjectFileExplorer
    raise AttributeError(name)
