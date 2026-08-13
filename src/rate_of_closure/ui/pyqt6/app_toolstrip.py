"""Application-level toolstrip and module-management presentation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QTextBrowser,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.commands import AppCommandId, CommandAvailability
from rate_of_closure.ui.pyqt6.regional_ground_file_menu import (
    RegionalGroundFileCommandGroup,
)
from rate_of_closure.ui.pyqt6.workspace_navigation import PrimaryModuleEntry

_PROJECT_DISABLED_REASON = (
    "Unavailable until the canonical project document contract is implemented."
)
_COMPOSITOR_DISABLED_REASON = (
    "Unavailable until the synchronized multi-view compositor contract is implemented."
)
_MODULE_ID_ROLE = Qt.ItemDataRole.UserRole


class ToolstripHost(Protocol):
    """Narrow command facade consumed by the application shell."""

    def open_glossary(self, term: str = "") -> None:
        """Open the glossary module."""

    def show_help(self) -> None:
        """Open help for the active module."""

    def show_module_manager(self) -> None:
        """Open module management."""

    def restore_default_workspace(self) -> None:
        """Restore the declared workspace defaults."""

    def open_regional_ground_variation_request(self) -> None:
        """Open a combined seeded regional-ground request."""

    def save_regional_ground_variation_request_as(self) -> None:
        """Save the combined current request to a chosen native path."""


class ModuleManagerHost(Protocol):
    """Workspace operations used by the module manager dialog."""

    def primary_module_entries(self) -> tuple[PrimaryModuleEntry, ...]:
        """Return current module presentation state."""

    def set_primary_module_visible(self, module_id: str, visible: bool) -> bool:
        """Apply module visibility."""

    def move_primary_module(self, module_id: str, offset: int) -> bool:
        """Move a module in the workspace order."""

    def restore_default_workspace(self) -> None:
        """Restore the declared workspace defaults."""


class ApplicationToolstrip(QToolBar):
    """Compact application commands with stable IDs and accessible reasons."""

    def __init__(self, host: ToolstripHost, parent: QWidget | None = None) -> None:
        super().__init__("Application Commands", parent)
        self.setObjectName("applicationToolstrip")
        self.setAccessibleName("Application Commands")
        self.setMovable(False)
        self.setFloatable(False)
        self._host = host
        self._actions: dict[AppCommandId, QAction] = {}
        self._regional_ground_files = RegionalGroundFileCommandGroup(
            host, self, self._actions
        )
        self._shortcut_dialog: QDialog | None = None
        self._tools_menu = QMenu("Tools", self)
        self._theme_button = self._make_theme_button()
        self._build()

    def command(self, command_id: AppCommandId) -> QAction:
        """Return a registered action by stable command ID."""
        if not isinstance(command_id, AppCommandId):
            raise TypeError("command_id must be an AppCommandId")
        try:
            return self._actions[command_id]
        except KeyError as exc:
            raise ValueError(f"unknown application command: {command_id}") from exc

    def shortcut_dialog(self) -> QDialog | None:
        """Return the currently displayed shortcut dialog, if any."""
        return self._shortcut_dialog

    def bind_theme_menu(self, menu: QMenu) -> None:
        """Attach the launcher-owned theme menu without duplicating its state."""
        if menu is None:
            raise ValueError("theme menu must be provided")
        self._theme_button.setMenu(menu)
        self._theme_button.setEnabled(True)
        self._theme_button.setToolTip("Choose the application theme")
        placeholder = self.command(AppCommandId.GLOBAL_TOGGLE_THEME)
        placeholder.setVisible(False)
        self._apply_availability(placeholder, CommandAvailability.available())
        self._tools_menu.addMenu(menu)

    def set_active_module(self, module_id: str) -> None:
        """Enable combined request commands only in their relevant modules."""
        self._regional_ground_files.set_active_module(module_id)

    def show_shortcut_help(self) -> None:
        """Show every supported application shortcut in one modeless dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setModal(False)
        dialog.resize(480, 360)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser(dialog)
        browser.setObjectName("shortcutHelpBrowser")
        lines = ["Keyboard Shortcuts", ""]
        for action in self._actions.values():
            shortcut = action.shortcut().toString(
                QKeySequence.SequenceFormat.NativeText
            )
            if shortcut:
                lines.append(f"{action.text().replace('&', '')}: {shortcut}")
        lines.append("")
        lines.append(
            "Shortcuts use modifier keys and do not replace ordinary text editing keys."
        )
        browser.setPlainText("\n".join(lines))
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        self._shortcut_dialog = dialog
        dialog.show()

    def _build(self) -> None:
        """Assemble the top-level command groups and direct actions."""
        file_menu = QMenu("File", self)
        self._regional_ground_files.add_to(file_menu)
        file_menu.addSeparator()
        self._add_disabled_file_commands(file_menu)
        self._add_menu_button("File", "fileMenuButton", file_menu)
        self._build_view_menu()
        glossary, shortcuts = self._build_tools_menu()
        self.addSeparator()
        self._add_direct_button(glossary, "glossaryToolButton", "Glossary")
        self.addWidget(self._theme_button)
        self._add_direct_button(
            shortcuts, "shortcutHelpToolButton", "Keyboard Shortcuts"
        )

    def _build_view_menu(self) -> None:
        """Create workspace management and future compositor actions."""
        view_menu = QMenu("View", self)
        manage = self._make_action(
            AppCommandId.VIEW_MANAGE_MODULES,
            "Manage Modules…",
            self._host.show_module_manager,
            "Ctrl+Shift+M",
        )
        manage.setToolTip("Show, hide, and reorder workspace modules")
        view_menu.addAction(manage)
        restore = self._make_action(
            AppCommandId.VIEW_RESTORE_DEFAULT_WORKSPACE,
            "Restore Default Workspace",
            self._host.restore_default_workspace,
        )
        restore.setToolTip("Restore the default module order and visibility")
        view_menu.addAction(restore)
        view_menu.addSeparator()
        self._add_disabled_view_commands(view_menu)
        self._add_menu_button("View", "viewMenuButton", view_menu)

    def _add_disabled_view_commands(self, menu: QMenu) -> None:
        """Register compositor commands with truthful availability reasons."""
        for command_id, label in (
            (AppCommandId.VIEW_SHOW_IMPACT, "Show Impact View"),
            (AppCommandId.VIEW_SHOW_SWING, "Show Swing View"),
            (AppCommandId.VIEW_SHOW_FLIGHT, "Show Flight View"),
        ):
            action = self._make_action(command_id, label)
            self._apply_availability(
                action, CommandAvailability.disabled(_COMPOSITOR_DISABLED_REASON)
            )
            menu.addAction(action)

    def _build_tools_menu(self) -> tuple[QAction, QAction]:
        """Create global tools and return actions also shown directly."""
        glossary = self._make_action(
            AppCommandId.GLOBAL_OPEN_GLOSSARY,
            "Glossary",
            self._host.open_glossary,
            "Ctrl+G",
        )
        glossary.setToolTip("Open the searchable engineering glossary")
        shortcuts = self._make_action(
            AppCommandId.GLOBAL_SHOW_SHORTCUTS,
            "Keyboard Shortcuts",
            self.show_shortcut_help,
            "Ctrl+/",
        )
        shortcuts.setToolTip("Show all application keyboard shortcuts")
        current_help = self._make_action(
            AppCommandId.GLOBAL_OPEN_CURRENT_MODULE_HELP,
            "Current Module Help",
            self._host.show_help,
            "F1",
        )
        current_help.setToolTip("Open help for the active workspace module")
        self._tools_menu.addActions((glossary, shortcuts, current_help))
        theme_placeholder = self._make_action(
            AppCommandId.GLOBAL_TOGGLE_THEME, "Theme (launcher not ready)"
        )
        self._apply_availability(
            theme_placeholder,
            CommandAvailability.disabled(
                "Theme choices become available after launcher setup."
            ),
        )
        self._tools_menu.addAction(theme_placeholder)
        self._add_menu_button("Tools", "toolsMenuButton", self._tools_menu)
        return glossary, shortcuts

    def _add_disabled_file_commands(self, menu: QMenu) -> None:
        commands = (
            (AppCommandId.FILE_NEW_WORKSPACE, "New", QKeySequence.StandardKey.New),
            (AppCommandId.FILE_OPEN_WORKSPACE, "Open…", QKeySequence.StandardKey.Open),
            (AppCommandId.FILE_OPEN_RECENT_WORKSPACE, "Open Recent", None),
            (AppCommandId.FILE_SAVE_WORKSPACE, "Save", QKeySequence.StandardKey.Save),
            (
                AppCommandId.FILE_SAVE_WORKSPACE_AS,
                "Save As…",
                QKeySequence.StandardKey.SaveAs,
            ),
            (AppCommandId.FILE_IMPORT_WORKSPACE, "Import…", None),
            (AppCommandId.FILE_EXPORT_WORKSPACE, "Export…", None),
            (
                AppCommandId.FILE_CLOSE_WORKSPACE,
                "Close",
                QKeySequence.StandardKey.Close,
            ),
        )
        for command_id, label, shortcut in commands:
            action = self._make_action(command_id, label)
            if shortcut is not None:
                action.setShortcut(QKeySequence(shortcut))
            self._apply_availability(
                action, CommandAvailability.disabled(_PROJECT_DISABLED_REASON)
            )
            menu.addAction(action)

    def _make_action(
        self,
        command_id: AppCommandId,
        label: str,
        callback: Callable[[], None] | None = None,
        shortcut: str | None = None,
    ) -> QAction:
        action = QAction(label, self)
        if not isinstance(command_id, AppCommandId):
            raise TypeError("command_id must be an AppCommandId")
        action.setObjectName(command_id.value)
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if callback is not None:
            action.triggered.connect(callback)
        self._actions[command_id] = action
        return action

    @staticmethod
    def _apply_availability(action: QAction, availability: CommandAvailability) -> None:
        """Apply the UI-neutral availability invariant to one Qt action."""
        action.setEnabled(availability.enabled)
        reason = availability.disabled_reason or ""
        action.setToolTip(reason)
        action.setStatusTip(reason)

    def _add_menu_button(self, label: str, name: str, menu: QMenu) -> None:
        button = QToolButton(self)
        button.setObjectName(name)
        button.setText(label)
        button.setAccessibleName(f"{label} Commands")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setMenu(menu)
        self.addWidget(button)

    def _make_theme_button(self) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("themeMenuButton")
        button.setText("Theme")
        button.setAccessibleName("Theme")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setEnabled(False)
        button.setToolTip("Theme choices become available after launcher setup")
        return button

    def _add_direct_button(self, action: QAction, name: str, label: str) -> None:
        self.addAction(action)
        button = self.widgetForAction(action)
        if isinstance(button, QToolButton):
            button.setObjectName(name)
            button.setAccessibleName(label)


class ModuleManagerDialog(QDialog):
    """Modeless show/hide/reorder editor over a narrow workspace facade."""

    def __init__(self, host: ModuleManagerHost, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Workspace Modules")
        self.setModal(False)
        self.resize(460, 420)
        self._host = host
        layout = QVBoxLayout(self)
        self._list = QListWidget(self)
        self._list.setAccessibleName("Workspace Modules")
        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)
        controls = QHBoxLayout()
        up = self._move_button("Move Up", "moveModuleUpButton", -1)
        down = self._move_button("Move Down", "moveModuleDownButton", 1)
        controls.addWidget(up)
        controls.addWidget(down)
        restore = QToolButton(self)
        restore.setText("Restore Defaults")
        restore.setObjectName("restoreModulesButton")
        restore.clicked.connect(self._restore)
        controls.addWidget(restore)
        layout.addLayout(controls)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close.rejected.connect(self.close)
        layout.addWidget(close)
        self._refresh()

    def module_item(self, module_id: str) -> QListWidgetItem:
        """Return one list item by stable module ID."""
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item is not None and item.data(_MODULE_ID_ROLE) == module_id:
                return item
        raise ValueError(f"unknown module item: {module_id}")

    def _refresh(self, selected_id: str | None = None) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for entry in self._host.primary_module_entries():
            label = f"{entry.label} (Required)" if entry.required else entry.label
            item = QListWidgetItem(label)
            item.setData(_MODULE_ID_ROLE, entry.module_id)
            item.setCheckState(
                Qt.CheckState.Checked if entry.visible else Qt.CheckState.Unchecked
            )
            if entry.required:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                item.setToolTip("Required core module; it cannot be hidden")
            self._list.addItem(item)
            if entry.module_id == selected_id:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        module_id = str(item.data(_MODULE_ID_ROLE))
        visible = item.checkState() == Qt.CheckState.Checked
        self._host.set_primary_module_visible(module_id, visible)
        self._refresh(module_id)

    def _move_selected(self, offset: int) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        module_id = str(item.data(_MODULE_ID_ROLE))
        self._host.move_primary_module(module_id, offset)
        self._refresh(module_id)

    def _restore(self) -> None:
        self._host.restore_default_workspace()
        self._refresh()

    def _move_button(self, label: str, name: str, offset: int) -> QToolButton:
        button = QToolButton(self)
        button.setText(label)
        button.setObjectName(name)
        button.setAccessibleName(label)
        button.clicked.connect(lambda _checked=False: self._move_selected(offset))
        return button
