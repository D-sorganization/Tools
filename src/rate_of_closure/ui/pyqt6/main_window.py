"""Rate of Closure Impact Explorer PyQt6 application shell."""

from __future__ import annotations

from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.derivation import (
    METRIC_EXPLANATIONS,
    RESULT_EXPLANATIONS,
)
from rate_of_closure.helptext import HELP_TEXTS
from rate_of_closure.ui.pyqt6.app_style import showcase_stylesheet
from rate_of_closure.ui.pyqt6.app_toolstrip import (
    ApplicationToolstrip,
    ModuleManagerDialog,
)
from rate_of_closure.ui.pyqt6.club_view import Club3DView
from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel
from rate_of_closure.ui.pyqt6.derivation_view import DerivationView
from rate_of_closure.ui.pyqt6.durable_ensemble_tab import DurableEnsembleTab
from rate_of_closure.ui.pyqt6.durable_ensemble_worker import (
    DurableEnsembleAuthorityPort,
)
from rate_of_closure.ui.pyqt6.flight_explorer_tab import FlightExplorerTab
from rate_of_closure.ui.pyqt6.glossary_tab import GlossaryTab
from rate_of_closure.ui.pyqt6.launch_monitor_analytics_tab import (
    LaunchMonitorAnalyticsTab,
)
from rate_of_closure.ui.pyqt6.main_window_club import MainWindowClubMixin
from rate_of_closure.ui.pyqt6.main_window_contracts import (
    _METRIC_ROWS,
    _RESULT_ROWS,
    _TAB_HELP_KEYS,
)
from rate_of_closure.ui.pyqt6.main_window_layout import (
    PrimaryTabSpec,
    ResultsSidebar,
    create_primary_tabs,
)
from rate_of_closure.ui.pyqt6.morris_tab import MorrisScreeningTab
from rate_of_closure.ui.pyqt6.morris_worker import MorrisAuthorityPort
from rate_of_closure.ui.pyqt6.neural_model_lab_tab import NeuralModelLabTab
from rate_of_closure.ui.pyqt6.plots_tab import PlotsTab
from rate_of_closure.ui.pyqt6.putting_tab import PuttingTab
from rate_of_closure.ui.pyqt6.result_row import ResultRow as _ResultRow
from rate_of_closure.ui.pyqt6.result_row import explanation_html
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab
from rate_of_closure.ui.pyqt6.variation_tab import VariationTab
from rate_of_closure.ui.pyqt6.variation_workspace import VariationWorkspace
from rate_of_closure.ui.pyqt6.workspace_layout import WorkspaceLayoutMixin
from rate_of_closure.ui.pyqt6.workspace_navigation import (
    _DEFAULT_TAB_IDS,
    _NAVIGATION_SETTINGS_APP,
    _NAVIGATION_SETTINGS_ORG,
    _NAVIGATION_STATE_KEY,
    _NAVIGATION_STATE_VERSION,
    _REQUIRED_TAB_IDS,
    NavigationSettings,
    WorkspaceNavigationMixin,
)
from shared.python.gui_launcher.tools_sidebar_integration import (
    ToolsSidebarInstallStatus,
    install_tools_sidebar,
)
from shared.python.swing_sim.variation import VariationDataset

__all__ = ["RateOfClosureMainWindow"]

# Compatibility exports retained for existing shell and help-contract tests.
__all__ += [
    "_DEFAULT_TAB_IDS",
    "_NAVIGATION_SETTINGS_APP",
    "_NAVIGATION_SETTINGS_ORG",
    "_NAVIGATION_STATE_KEY",
    "_NAVIGATION_STATE_VERSION",
    "_REQUIRED_TAB_IDS",
    "_TAB_HELP_KEYS",
]

# ── Theme integration (optional — graceful fallback) ───────────────
try:
    from shared.python.theme.integration import ThemedWindowMixin

    _THEME_AVAILABLE = True
except ImportError:  # standalone / vendored use
    _THEME_AVAILABLE = False

    class ThemedWindowMixin:  # type: ignore[no-redef]
        """No-op stand-in when the shared theme package is unavailable."""

        def setup_theme_support(self, settings_app: str = "") -> None:
            """Match the themed mixin's interface; do nothing."""


class RateOfClosureMainWindow(
    MainWindowClubMixin,
    WorkspaceLayoutMixin,
    WorkspaceNavigationMixin,
    ThemedWindowMixin,
    QMainWindow,
):
    """Interactive explorer for rotation-induced impact-point deviations."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        navigation_settings: NavigationSettings | None = None,
        morris_client: MorrisAuthorityPort | None = None,
        durable_ensemble_client: DurableEnsembleAuthorityPort | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rate of Closure Impact Explorer")
        self.setMinimumSize(1024, 700)

        self._create_views(morris_client, durable_ensemble_client)
        self._build_application_shell(navigation_settings)
        self._connect_view_signals()
        self._initialize_view_content()

    def _create_views(
        self,
        morris_client: MorrisAuthorityPort | None,
        durable_client: DurableEnsembleAuthorityPort | None,
    ) -> None:
        """Create the application views without coupling their signal graph."""

        self._controls = ControlsPanel()
        self._rows: dict[str, _ResultRow] = {}
        self._club_view = Club3DView()
        self._plots_tab = PlotsTab()
        self._derivation_view = DerivationView()
        self._simulation_tab = SimulationTab()
        self._simulation_tab.runCompleted.connect(self._plots_tab.set_run)
        self._flight_explorer_tab = FlightExplorerTab()
        self._launch_monitor_analytics_tab = LaunchMonitorAnalyticsTab()
        self._neural_model_lab_tab = NeuralModelLabTab()
        self._variation_tab = VariationTab()
        self._durable_ensemble_tab = DurableEnsembleTab(
            durable_client, self._variation_tab.build_plan
        )
        self._morris_tab = MorrisScreeningTab(morris_client)
        self._morris_tab.shutdownReady.connect(self._resume_pending_close)
        self._durable_ensemble_tab.shutdownReady.connect(self._resume_pending_close)
        self._close_pending = False
        self._variation_workspace = VariationWorkspace(
            self._variation_tab, self._morris_tab, self._durable_ensemble_tab
        )
        self._variation_tab.studyCompleted.connect(self._on_variation_study)
        self._putting_tab = PuttingTab()
        self._glossary_tab = GlossaryTab()

    def _build_application_shell(
        self, navigation_settings: NavigationSettings | None
    ) -> None:
        """Assemble navigation, toolstrip, result sidebar, and central splitter."""
        sidebar = ResultsSidebar(
            self._controls, self._show_explanation, self._on_explanation_link
        )
        self._rows = sidebar.rows
        self._explanation = sidebar.explanation
        self._navigation_settings = (
            navigation_settings
            if navigation_settings is not None
            else QSettings(_NAVIGATION_SETTINGS_ORG, _NAVIGATION_SETTINGS_APP)
        )
        self._tabs = create_primary_tabs(self._primary_tab_specs())
        self._restore_primary_navigation()
        tab_bar = self._primary_tab_bar()
        tab_bar.tabMoved.connect(self._persist_primary_navigation)
        self._tabs.currentChanged.connect(self._persist_primary_navigation)
        self._configure_toolstrip_and_help()
        sidebar.setMinimumWidth(200)
        self._tabs.setMinimumWidth(640)
        self._shell_splitter = QSplitter()
        self._shell_splitter.setChildrenCollapsible(False)
        self._shell_splitter.addWidget(sidebar)
        self._shell_splitter.addWidget(self._tabs)
        self._shell_splitter.setStretchFactor(0, 0)
        self._shell_splitter.setStretchFactor(1, 1)
        self.setCentralWidget(self._shell_splitter)
        self._restore_visual_layout()
        self._connect_visual_layout_persistence()
        QTimer.singleShot(0, self._reapply_visual_layout_geometry)
        self.setStatusBar(QStatusBar())
        self.setStyleSheet(showcase_stylesheet(self.palette()))
        self._sidekick_status: ToolsSidebarInstallStatus = install_tools_sidebar(
            self,
            context_provider=self._get_sidekick_context,
        )
        if self._sidekick_status.dock and hasattr(self._sidekick_status.dock, "hide"):
            self._sidekick_status.dock.hide()

    def _primary_tab_specs(self) -> tuple[PrimaryTabSpec, ...]:
        """Return stable primary-module registrations in first-run order."""
        return (
            ("clubhead", self._club_view, "3D Clubhead"),
            ("plots", self._plots_tab, "Plots"),
            (
                "calculation_description",
                self._derivation_view,
                "Calculation Description",
            ),
            ("simulation", self._simulation_tab, "Simulation"),
            ("flight_explorer", self._flight_explorer_tab, "Flight Explorer"),
            (
                "launch_monitor_analytics",
                self._launch_monitor_analytics_tab,
                "Launch Monitor Analytics",
            ),
            ("neural_model_lab", self._neural_model_lab_tab, "Neural Model Lab"),
            ("variation", self._variation_workspace, "Variation"),
            ("putting", self._putting_tab, "Putting"),
            ("glossary", self._glossary_tab, "Glossary"),
        )

    def _configure_toolstrip_and_help(self) -> None:
        """Install the top toolstrip and contextual-help entry point."""
        self._module_manager_dialog: ModuleManagerDialog | None = None
        self._app_toolstrip = ApplicationToolstrip(self, self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._app_toolstrip)
        help_button = QToolButton()
        help_button.setText("?")
        help_button.setToolTip(
            "Open detailed help for the current tab: what it does, the "
            "workflow, and a control reference."
        )
        help_button.clicked.connect(self.show_help)
        self._tabs.setCornerWidget(help_button, Qt.Corner.TopRightCorner)
        self._help_dialog: QDialog | None = None

    def _connect_view_signals(self) -> None:
        """Connect cross-view signals after every view has been constructed."""
        self._controls.scenarioChanged.connect(self._on_scenario)
        self._controls.clubHeadRequested.connect(self._on_club_head)
        self._controls.distanceUnitChanged.connect(self._on_distance_unit)
        self._simulation_tab.glossaryRequested.connect(self.open_glossary)
        self._simulation_tab.configChanged.connect(self._on_derivation_config_changed)
        self._simulation_tab.simulationConfigChanged.connect(
            self._on_simulation_config_changed
        )
        self._simulation_tab.clubSelectionChanged.connect(self._controls.set_club_name)
        self._flight_explorer_tab.glossaryRequested.connect(self.open_glossary)
        self._putting_tab.glossaryRequested.connect(self.open_glossary)

    def _initialize_view_content(self) -> None:
        """Populate all views with a representative, internally consistent run."""
        self._derivation_view.set_config(self._simulation_tab.derivation_config())
        self._on_scenario(self._controls.scenario())
        # A share-ready scene should never open as a placeholder wireframe.
        # Load the selected representative driver immediately; users can still
        # regenerate another library head or load a measured STL.
        self._on_club_head(self._controls.club_spec())
        # Match the web experience: the Swing view opens with a meaningful
        # result instead of empty axes that look like a rendering failure.
        self._simulation_tab.run_now()
        self._show_explanation(_RESULT_ROWS[0][0])
        self._seed_sidekick_workspace()

    # ── behaviour ───────────────────────────────────────────────────
    def _show_explanation(self, field: str) -> None:
        labels = dict(_RESULT_ROWS) | dict(_METRIC_ROWS)
        text = RESULT_EXPLANATIONS.get(field) or METRIC_EXPLANATIONS.get(field, "")
        # Persistent single selection across both row groups (#4120 V4).
        for row_field, row in self._rows.items():
            row.set_selected(row_field == field)
        self._explanation.setHtml(explanation_html(labels[field], text, field))

    def _on_explanation_link(self, url) -> None:  # type: ignore[no-untyped-def]
        """Route ``glossary:<term>`` links to the Glossary tab."""
        text = url.toString()
        if not text.startswith("glossary:"):
            return
        self.open_glossary(text.partition(":")[2])

    def show_help(self) -> None:
        """Open the rich-text help panel for the current tab (V4)."""
        entry = HELP_TEXTS[self._current_primary_tab_id()]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Help — {entry.title}")
        dialog.resize(560, 520)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setObjectName("helpBrowser")
        browser.setOpenExternalLinks(False)
        browser.setHtml(f"<h2>{entry.title}</h2>{entry.html}")
        layout.addWidget(browser)
        self._help_dialog = dialog
        dialog.show()

    def _on_distance_unit(self, _unit: str) -> None:
        """Re-render distance surfaces in the new display unit (H6)."""
        self._simulation_tab.refresh_units()
        self._flight_explorer_tab.refresh_units()
        self._putting_tab.refresh_units()

    def _on_derivation_config_changed(self, config: object) -> None:
        """Preserve the established derivation-only signal contract."""
        from rate_of_closure.derivation_models import DerivationConfig

        if isinstance(config, DerivationConfig):
            self._derivation_view.set_config(config)

    def _on_simulation_config_changed(self, config: object) -> None:
        """Keep both variation workflows on one exact runnable base."""
        from rate_of_closure.simulation import SimulationConfig

        if not isinstance(config, SimulationConfig):
            message = (
                "Current Simulation inputs are incomplete or invalid; repair them "
                "before running variation analysis."
            )
            self._variation_tab.set_simulation_unavailable(message)
            self._durable_ensemble_tab.set_simulation_unavailable(message)
            self._morris_tab.set_simulation_unavailable(message)
            return
        self._variation_tab.set_simulation_config(config)
        self._durable_ensemble_tab.set_simulation_config(config)
        self._morris_tab.set_simulation_config(config)

    def _on_variation_study(self, dataset: VariationDataset) -> None:
        """Forward a completed study's landing scatter (#4125 H7b)."""
        names = dataset.output_names
        if "carry_m" not in names or "lateral_m" not in names:
            return  # impact-only study: no landing plane to overlay
        self._simulation_tab.set_landing_scatter(
            dataset.output_column("carry_m"), dataset.output_column("lateral_m")
        )

    def open_glossary(self, term: str = "") -> None:
        """Show the Glossary tab, pre-selecting ``term`` when known."""
        self.show_primary_module("glossary")
        if term:
            self._glossary_tab.select_term(term)

    def show_module_manager(self) -> None:
        """Open the modeless workspace module manager."""
        if self._module_manager_dialog is not None:
            self._module_manager_dialog.close()
        dialog = ModuleManagerDialog(self, self)
        self._module_manager_dialog = dialog
        dialog.show()

    def module_manager_dialog(self) -> ModuleManagerDialog | None:
        """Return the current workspace module manager, if one was opened."""
        return self._module_manager_dialog

    def bind_theme_menu(self, menu) -> None:  # type: ignore[no-untyped-def]
        """Expose the launcher-owned theme choices in the top toolstrip."""
        self._app_toolstrip.bind_theme_menu(menu)

    def shortcut_help_dialog(self) -> QDialog | None:
        """Return the current keyboard-shortcut help dialog."""
        dialog = self._app_toolstrip.shortcut_dialog()
        return dialog if isinstance(dialog, QDialog) else None

    def _bind_launcher_theme_menu(self) -> None:
        """Move the launcher-provided Theme surface into the top toolstrip."""
        theme_button = self._app_toolstrip.findChild(QToolButton, "themeMenuButton")
        if theme_button is not None and theme_button.isEnabled():
            return
        menu_bar = self.menuBar()
        if menu_bar is None:
            return
        for action in menu_bar.actions():
            menu = action.menu()
            if menu is None or menu.title().replace("&", "") != "Theme":
                continue
            self.bind_theme_menu(menu)
            menu_bar.removeAction(action)
            return

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Bind launcher services after its post-construction setup completes."""
        self._bind_launcher_theme_menu()
        super().showEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Stop the animation timers before the window goes away."""
        self._persist_primary_navigation()
        self._club_view.stop()
        self._simulation_tab.stop()
        self._variation_tab.stop()
        durable_ready = self._durable_ensemble_tab.stop()
        if not self._morris_tab.stop() or not durable_ready:
            self._close_pending = True
            event.ignore()
            return
        self._close_pending = False
        self._flight_explorer_tab.stop()
        super().closeEvent(event)

    def _resume_pending_close(self) -> None:
        """Retry a deferred close after all retained transport threads finish."""
        if self._close_pending:
            QTimer.singleShot(0, self.close)

    def _get_sidekick_context(self) -> dict[str, object]:
        """Provide host-specific context for Sidekick assist."""
        active_club = self._controls.club_spec() if hasattr(self, "_controls") else None
        return {
            "tool_name": "rate_of_closure",
            "active_club": active_club,
        }

    def _seed_sidekick_workspace(self) -> None:
        """Publish live simulation and club models into Sidekick's workspace."""
        if not (hasattr(self, "_sidekick_status") and self._sidekick_status.sidebar):
            return
        registry = getattr(self._sidekick_status.sidebar, "registry", None)
        if registry is None or not hasattr(registry, "set"):
            return
        if hasattr(self, "_controls"):
            registry.set("active_club", self._controls.club_spec())
        if hasattr(self, "_plots_tab"):
            run = getattr(self._plots_tab, "_run", None)
            if run is not None:
                registry.set("simulation_run", run)
        if hasattr(self, "_variation_tab"):
            dataset = getattr(self._variation_tab, "_dataset", None)
            if dataset is not None:
                registry.set("variation_dataset", dataset)

    def toggle_sidekick_sidebar(self) -> None:
        """Toggle visibility of the embedded Sidekick Tools dock."""
        if hasattr(self, "_sidekick_status") and self._sidekick_status.dock:
            dock = self._sidekick_status.dock
            if hasattr(dock, "isVisible") and hasattr(dock, "setVisible"):
                dock.setVisible(not dock.isVisible())
