"""
UI Widgets
==========

Reusable UI components for the simulation overlay.

This package provides interactive widgets for controlling the simulation,
displaying information, and enhancing the educational experience.

Originally a single ``widgets.py`` module; split into submodules grouped by
widget purpose while preserving the public API via re-exports here.
"""

import logging

from ._base import PanelStyle
from .date_time import DateTimePicker, TimeNavigationPanel
from .historical_events_panel import HistoricalEventsPanel
from .immersion import ImmersionChecklistPanel, ImmersionTask
from .info_panels import EducationalInfoPanel, InfoPanel, StatusBar
from .overlays import HelpOverlay, TooltipManager, TransferPlanner
from .settings_nav import Checkbox, NavigationPanel, SettingsPanel
from .sidebar_controls import (
    Button,
    MissionListPanel,
    SidebarPanel,
    Tab,
    UnifiedControlPanel,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Button",
    "Checkbox",
    "DateTimePicker",
    "EducationalInfoPanel",
    "HelpOverlay",
    "HistoricalEventsPanel",
    "ImmersionChecklistPanel",
    "ImmersionTask",
    "InfoPanel",
    "MissionListPanel",
    "NavigationPanel",
    "PanelStyle",
    "SettingsPanel",
    "SidebarPanel",
    "StatusBar",
    "Tab",
    "TimeNavigationPanel",
    "TooltipManager",
    "TransferPlanner",
    "UnifiedControlPanel",
    "logger",
]
