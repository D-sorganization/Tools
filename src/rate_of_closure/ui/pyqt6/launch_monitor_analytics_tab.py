"""Full PyQt6 launch-monitor analytics and player-insight workbench."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.launch_monitor_analysis import AnalysisResult
from rate_of_closure.launch_monitor_data import CampaignDatasetCatalog
from rate_of_closure.ui.pyqt6.launch_monitor_analysis_mixin import (
    LaunchMonitorAnalysisMixin,
)
from rate_of_closure.ui.pyqt6.launch_monitor_data_mixin import (
    LaunchMonitorDataMixin,
    demo_frame,
)
from rate_of_closure.ui.pyqt6.launch_monitor_player_controls import (
    LaunchMonitorPlayerControls,
)
from rate_of_closure.ui.pyqt6.launch_monitor_plot_widget import (
    LaunchMonitorPlotWidget,
)
from shared.python.swing_sim.conventions import ConventionId


class LaunchMonitorAnalyticsTab(
    LaunchMonitorAnalysisMixin, LaunchMonitorDataMixin, QWidget
):
    """Load full campaign tables and run statistical/player analyses."""

    def __init__(
        self, parent: QWidget | None = None, *, auto_discover_campaign: bool = True
    ) -> None:
        super().__init__(parent)
        self.frame = demo_frame()
        self.source_name = "Built-In Demonstration Data"
        self.dataset_id = "demo"
        self.source_sha256 = ""
        self.catalog: CampaignDatasetCatalog | None = None
        self.last_result: AnalysisResult | None = None
        self.player_payload: dict[str, object] = {}
        self._build_ui()
        self._refresh_columns()
        if auto_discover_campaign:
            self.refresh_campaign_catalog()

    @staticmethod
    def _help(control: QWidget, name: str, tip: str) -> None:
        control.setAccessibleName(name)
        control.setToolTip(tip)

    def _build_ui(self) -> None:
        heading = QLabel("Launch Monitor Player Analytics")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        boundary = QLabel(
            "Full private campaign tables remain in their source repository. "
            "Associations, fitted models, PCA, and feature importance are not "
            "causal evidence or vendor-device emulation."
        )
        boundary.setWordWrap(True)
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        toolbar = self._build_data_toolbar()
        self._build_statistical_controls()
        self.player_controls = LaunchMonitorPlayerControls()
        control_tabs = QTabWidget()
        control_tabs.addTab(self.statistics_controls, "Statistics")
        control_tabs.addTab(self.player_controls, "Player Analytics")
        self.run_button = QPushButton("Run Analysis and Plot")
        self._help(
            self.run_button,
            "Run Analysis",
            "Run statistics and the selected unit-aware plot",
        )
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(control_tabs)
        left_layout.addWidget(self.run_button)
        left_layout.addStretch(1)
        outputs = self._build_output_tabs()
        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(left)
        body.addWidget(outputs)
        body.setSizes([390, 1050])
        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(boundary)
        layout.addLayout(toolbar)
        layout.addWidget(self.source_label)
        layout.addWidget(body, 1)
        self._connect_signals()
        self._refresh_guidance()

    def _build_data_toolbar(self) -> QHBoxLayout:
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("Built-In Demonstration Data", "demo")
        self.refresh_button = QPushButton("Refresh Campaign")
        self.import_button = QPushButton("Import Data...")
        self.demo_button = QPushButton("Load Demo")
        self.save_project_button = QPushButton("Save Project...")
        self.load_project_button = QPushButton("Load Project...")
        self.export_data_button = QPushButton("Export Data...")
        self.export_result_button = QPushButton("Export Analysis...")
        self.export_plot_button = QPushButton("Export Plot...")
        self.export_plot_data_button = QPushButton("Export Plot Data...")
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.dataset_combo, 1)
        buttons = (
            self.refresh_button,
            self.import_button,
            self.demo_button,
            self.save_project_button,
            self.load_project_button,
            self.export_data_button,
            self.export_result_button,
            self.export_plot_button,
            self.export_plot_data_button,
        )
        for button in buttons:
            toolbar.addWidget(button)
        controls = (
            (
                self.dataset_combo,
                "Campaign Dataset",
                "Select any full private campaign CSV",
            ),
            (
                self.refresh_button,
                "Refresh Campaign",
                "Rediscover and recatalog private data",
            ),
            (
                self.import_button,
                "Import Data",
                "Import a local CSV or record-array JSON",
            ),
            (self.demo_button, "Load Demo", "Restore non-vendor demonstration shots"),
            (
                self.save_project_button,
                "Save Project",
                "Persist source identity and selections",
            ),
            (
                self.load_project_button,
                "Load Project",
                "Reload and verify a saved project",
            ),
            (
                self.export_data_button,
                "Export Data",
                "Export every retained row and column",
            ),
            (
                self.export_result_button,
                "Export Analysis",
                "Export results, formulas, and backing values",
            ),
            (
                self.export_plot_button,
                "Export Plot",
                "Save the plot as PNG, SVG, or PDF",
            ),
            (
                self.export_plot_data_button,
                "Export Plot Data",
                "Export exact plotted rows",
            ),
        )
        for control, name, tip in controls:
            self._help(control, name, tip)
        return toolbar

    def _build_statistical_controls(self) -> None:
        self.statistics_controls = QWidget()
        self.convention_combo = QComboBox()
        for label, value in (
            ("App-Native", ConventionId.APP_NATIVE),
            ("TrackMan-Comparable", ConventionId.TRACKMAN_COMPARABLE),
            ("Foresight-Comparable", ConventionId.FORESIGHT_COMPARABLE),
        ):
            self.convention_combo.addItem(label, value)
        self.convention_evidence = QLabel()
        self.convention_evidence.setWordWrap(True)
        self.convention_evidence.setOpenExternalLinks(True)
        self.outcome_combo = QComboBox()
        self.predictor_list = QListWidget()
        self.predictor_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["comprehensive", "correlation", "regression"])
        self.method_combo = QComboBox()
        self.method_combo.addItems(["pearson", "spearman", "kendall"])
        self.missing_combo = QComboBox()
        self.missing_combo.addItems(["pairwise", "listwise", "fail"])
        self.group_combo = QComboBox()
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.51, 0.999)
        self.confidence_spin.setValue(0.95)
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(3, 1_000_000)
        self.min_samples_spin.setValue(10)
        controls = (
            (
                self.convention_combo,
                "Interpretation Convention",
                "Choose a documented comparable frame",
            ),
            (self.outcome_combo, "Outcome Variable", "Choose any numeric outcome"),
            (
                self.predictor_list,
                "Predictor Variables",
                "Choose one or more numeric predictors",
            ),
            (
                self.mode_combo,
                "Analysis Mode",
                "Run correlation, OLS regression, or both",
            ),
            (
                self.method_combo,
                "Correlation Method",
                "Choose Pearson, Spearman, or Kendall",
            ),
            (
                self.missing_combo,
                "Missing-Data Policy",
                "Choose pairwise, listwise, or fail closed",
            ),
            (self.group_combo, "Optional Group", "Repeat analysis within each group"),
            (
                self.confidence_spin,
                "Confidence Level",
                "Set analytical interval coverage",
            ),
            (
                self.min_samples_spin,
                "Minimum Sample Count",
                "Reject insufficient analyses",
            ),
        )
        form = QFormLayout(self.statistics_controls)
        for control, name, tip in controls:
            self._help(control, name, tip)
            form.addRow(f"{name}:", control)
        form.insertRow(1, self.convention_evidence)

    def _build_output_tabs(self) -> QTabWidget:
        self.data_preview = QTableWidget()
        self.data_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._help(
            self.data_preview,
            "Dataset Preview",
            "Preview up to 500 rows; analysis and export retain the complete dataset",
        )
        self.result_table = QTableWidget()
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._help(
            self.result_table,
            "Statistical Results",
            "Correlation and OLS results with uncertainty",
        )
        self.plot_widget = LaunchMonitorPlotWidget()
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self._help(
            self.details,
            "Analysis Traceability",
            "Request, results, source hashes, and backing summaries",
        )
        self.guidance = QTextBrowser()
        self.guidance.setOpenExternalLinks(True)
        self._help(
            self.guidance,
            "Calculation Guide",
            "Formulas, assumptions, interpretations, and method sources",
        )
        outputs = QTabWidget()
        outputs.addTab(self.data_preview, "Dataset Preview")
        outputs.addTab(self.result_table, "Statistical Results")
        outputs.addTab(self.plot_widget, "Plot")
        outputs.addTab(self.details, "Backing Data / Lineage")
        outputs.addTab(self.guidance, "Calculations and Tips")
        return outputs

    def _connect_signals(self) -> None:
        self.dataset_combo.currentIndexChanged.connect(self._dataset_selected)
        self.refresh_button.clicked.connect(self.refresh_campaign_catalog)
        self.import_button.clicked.connect(self.import_dialog)
        self.demo_button.clicked.connect(self.load_demo)
        self.save_project_button.clicked.connect(self.save_project_dialog)
        self.load_project_button.clicked.connect(self.load_project_dialog)
        self.export_data_button.clicked.connect(self.export_data_dialog)
        self.export_result_button.clicked.connect(self.export_result_dialog)
        self.export_plot_button.clicked.connect(self.plot_widget.save_plot_dialog)
        self.export_plot_data_button.clicked.connect(
            self.plot_widget.export_backing_dialog
        )
        self.run_button.clicked.connect(self.run_analysis_safely)
        self.convention_combo.currentIndexChanged.connect(
            self._refresh_convention_evidence
        )
        self.outcome_combo.currentTextChanged.connect(self._refresh_convention_evidence)


__all__ = ["LaunchMonitorAnalyticsTab"]
