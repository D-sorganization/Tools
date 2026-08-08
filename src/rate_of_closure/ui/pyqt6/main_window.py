"""Rate of Closure Impact Explorer PyQt6 application shell."""

from __future__ import annotations

import math

from PyQt6.QtCore import QSettings, Qt
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

from rate_of_closure.club import (
    ClubSpec,
    head_cog,
    hosel_point,
    parametric_head_mesh,
)
from rate_of_closure.derivation import (
    METRIC_EXPLANATIONS,
    RESULT_EXPLANATIONS,
)
from rate_of_closure.helptext import HELP_TEXTS
from rate_of_closure.model import ImpactScenario, closure_metrics, solve
from rate_of_closure.ui.pyqt6.app_style import showcase_stylesheet
from rate_of_closure.ui.pyqt6.app_toolstrip import (
    ApplicationToolstrip,
    ModuleManagerDialog,
)
from rate_of_closure.ui.pyqt6.club_view import Club3DView
from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel
from rate_of_closure.ui.pyqt6.derivation_view import DerivationView
from rate_of_closure.ui.pyqt6.flight_explorer_tab import FlightExplorerTab
from rate_of_closure.ui.pyqt6.glossary_tab import GlossaryTab
from rate_of_closure.ui.pyqt6.launch_monitor_analytics_tab import (
    LaunchMonitorAnalyticsTab,
)
from rate_of_closure.ui.pyqt6.main_window_contracts import (
    _METRIC_ROWS,
    _QUANTITY_ROWS,
    _RESULT_ROWS,
    _TAB_HELP_KEYS,
    _UNITS,
)
from rate_of_closure.ui.pyqt6.main_window_layout import (
    PrimaryTabSpec,
    ResultsSidebar,
    create_primary_tabs,
)
from rate_of_closure.ui.pyqt6.plots_tab import PlotsTab
from rate_of_closure.ui.pyqt6.putting_tab import PuttingTab
from rate_of_closure.ui.pyqt6.result_row import ResultRow as _ResultRow
from rate_of_closure.ui.pyqt6.result_row import explanation_html
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab
from rate_of_closure.ui.pyqt6.variation_tab import VariationTab
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
from rate_of_closure.units import convert_from_canonical
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


class RateOfClosureMainWindow(WorkspaceNavigationMixin, ThemedWindowMixin, QMainWindow):
    """Interactive explorer for rotation-induced impact-point deviations."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        navigation_settings: NavigationSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rate of Closure Impact Explorer")
        self.setMinimumSize(1024, 700)

        self._create_views()
        self._build_application_shell(navigation_settings)
        self._connect_view_signals()
        self._initialize_view_content()

    def _create_views(self) -> None:
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
        self._variation_tab = VariationTab()
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
        splitter = QSplitter()
        splitter.addWidget(sidebar)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.setStyleSheet(showcase_stylesheet(self.palette()))

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
            ("variation", self._variation_tab, "Variation"),
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
        self._simulation_tab.configChanged.connect(self._derivation_view.set_config)
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

    def _format_row(self, field: str, value: float) -> str:
        """Format one row's value in the user's selected display unit."""
        if not math.isfinite(value):
            return "∞ (not closing)"
        quantity = _QUANTITY_ROWS.get(field)
        if quantity is None:
            return f"{value:+.2f}{_UNITS[field]}"
        unit = self._controls.unit_for(quantity)
        displayed = convert_from_canonical(quantity, unit, value)
        return f"{displayed:+.2f} {unit}"

    def _on_club_head(self, spec: ClubSpec) -> None:
        """Build the parametric head for a club spec and display it.

        The generated head carries its per-type hosel point (the shaft
        line attaches there) and its divergence-theorem volumetric COG
        for the "Show CG" marker.
        """
        report = head_cog(spec)
        self._club_view.set_head_mesh(
            parametric_head_mesh(spec),
            hosel_point=hosel_point(spec),
            cog_point=report.cog,
        )
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(
                f"Representative head generated: {spec.name} — loft "
                f"{spec.loft_deg:.1f}°, "
                + (
                    "curved face (bulge "
                    f"{spec.face_bulge_radius_m * 1000.0:.0f} mm, roll "
                    f"{spec.face_roll_radius_m * 1000.0:.0f} mm)"
                    if spec.face_bulge_radius_m is not None
                    and spec.face_roll_radius_m is not None
                    else "flat face"
                )
            )

    def _on_scenario(self, scenario: ImpactScenario) -> None:
        result = solve(scenario)
        metrics = closure_metrics(scenario)
        for field, _ in _RESULT_ROWS:
            self._rows[field].value_label.setText(
                self._format_row(field, getattr(result, field))
            )
        for field, _ in _METRIC_ROWS:
            self._rows[field].value_label.setText(
                self._format_row(field, getattr(metrics, field))
            )
        self._club_view.set_scenario(scenario)
        self._plots_tab.set_scenario(scenario)
        self._derivation_view.set_scenario(scenario)
        self._simulation_tab.set_club_spec(self._controls.club_spec())
        self._simulation_tab.set_scenario(scenario)
        self._variation_tab.set_scenario(scenario)
        self._variation_tab.set_simulation_config(self._simulation_tab.config())
        status_bar = self.statusBar()
        if status_bar is None:  # pragma: no cover - Qt always provides one here
            return
        status_bar.showMessage(
            f"Reference {result.reference_speed_mph:.1f} mph — impact point "
            f"path {result.path_deviation_deg:+.2f}° "
            f"({'left' if result.path_deviation_deg < 0 else 'right'}), "
            f"AoA {result.aoa_deviation_deg:+.2f}°, "
            f"CCV {result.closure_rate_dps:.0f} °/s "
            f"({result.normalized_closure_deg_per_ft:.1f} °/ft)"
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Stop the animation timers before the window goes away."""
        self._persist_primary_navigation()
        self._club_view.stop()
        self._simulation_tab.stop()
        self._variation_tab.stop()
        self._flight_explorer_tab.stop()
        super().closeEvent(event)
