"""Help system module for the Unified Tools Launcher.

This module provides a reusable help system with:
- HelpDialog: Modal dialog for displaying markdown documentation
- HelpButton: Context-sensitive help button widget
- TooltipManager: Enhanced tooltip management for complex inputs
- Help content loading from markdown files

Usage:
    from help import HelpDialog, HelpButton, TooltipManager

    # Show help dialog
    dialog = HelpDialog(parent, "Topic Title", "# Markdown content")
    dialog.exec()

    # Create a help button
    help_btn = HelpButton("topic_id", parent)

    # Set up tooltips
    tooltip_mgr = TooltipManager()
    tooltip_mgr.register_widget(widget, "tooltip_key")
"""

from .help_system import (
    HelpButton,
    HelpDialog,
    TooltipManager,
    get_help_manager,
    load_help_from_file,
)

__all__ = [
    "HelpButton",
    "HelpDialog",
    "TooltipManager",
    "get_help_manager",
    "load_help_from_file",
]
