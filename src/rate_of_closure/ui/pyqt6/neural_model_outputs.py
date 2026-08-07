"""Model inspection, query, batch prediction, and export presentation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QFileDialog,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.neural_model import (
    NeuralModelBundle,
    load_neural_bundle,
    predict_frame,
)


class NeuralModelOutputs(QWidget):
    """Own imported models and their inspectable prediction outputs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model: NeuralModelBundle | None = None
        self.predictions: pd.DataFrame | None = None
        self.current_frame = pd.DataFrame()
        self._build_ui()

    @staticmethod
    def _help(widget: QWidget, name: str, tip: str) -> None:
        widget.setAccessibleName(name)
        widget.setToolTip(tip)

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        self.log: QPlainTextEdit = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.model_summary: QPlainTextEdit = QPlainTextEdit()
        self.model_summary.setReadOnly(True)
        self.query_table: QTableWidget = QTableWidget(0, 3)
        self.query_table.setHorizontalHeaderLabels(["Feature", "Value", "Unit"])
        query_panel = self._query_panel()
        self.learning_figure = Figure(figsize=(7, 4), constrained_layout=True)
        self.learning_canvas = FigureCanvasQTAgg(self.learning_figure)
        self.help_browser = QTextBrowser()
        self.help_browser.setHtml(self._help_html())
        tabs.addTab(self.log, "Training Log")
        tabs.addTab(self.model_summary, "Metrics / Provenance")
        tabs.addTab(query_panel, "Query / Batch")
        tabs.addTab(self.learning_canvas, "Learning Curve")
        tabs.addTab(self.help_browser, "Procedure and Tips")
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        self._apply_help()

    def _query_panel(self) -> QWidget:
        self.query_button = QPushButton("Query Manual Inputs")
        self.batch_button = QPushButton("Predict Current Dataset")
        self.export_predictions_button: QPushButton = QPushButton(
            "Export Predictions..."
        )
        self.export_predictions_button.setEnabled(False)
        self.query_result = QPlainTextEdit()
        self.query_result.setReadOnly(True)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.query_table)
        for button in (
            self.query_button,
            self.batch_button,
            self.export_predictions_button,
        ):
            layout.addWidget(button)
        layout.addWidget(self.query_result)
        self.query_button.clicked.connect(self.query_manual)
        self.batch_button.clicked.connect(self.predict_current_dataset)
        self.export_predictions_button.clicked.connect(self._export_dialog)
        return panel

    def _apply_help(self) -> None:
        controls = (
            (
                self.log,
                "Training Log",
                "Exact private CLI output and lifecycle messages",
            ),
            (
                self.model_summary,
                "Model Metrics",
                "Holdout metrics, feature ranges, identity, and provenance",
            ),
            (
                self.query_table,
                "Manual Query Inputs",
                "Enter one value per feature in its documented unit",
            ),
            (
                self.query_button,
                "Run Manual Query",
                "Evaluate one row and report applicability warnings",
            ),
            (
                self.batch_button,
                "Run Batch Prediction",
                "Evaluate every current row with the required features",
            ),
            (
                self.export_predictions_button,
                "Export Predictions",
                "Export inputs and predictions as CSV or JSON",
            ),
            (
                self.learning_canvas,
                "Learning Curve",
                "Training and validation loss by epoch",
            ),
            (
                self.help_browser,
                "Neural Model Help",
                "Training, validation, inference, limitations, and safety procedure",
            ),
        )
        for widget, name, tip in controls:
            self._help(widget, name, tip)

    def set_frame(self, frame: pd.DataFrame) -> None:
        """Set the current batch-query dataframe."""

        self.current_frame = frame.copy()

    def import_model(self, path: Path) -> None:
        """Load and display a safe JSON bundle."""

        self.model = load_neural_bundle(path)
        summary = {
            "model_id": self.model.model_id,
            "vendor": self.model.vendor,
            "created_at": self.model.created_at,
            "features": [vars(item) for item in self.model.features],
            "outputs": [vars(item) for item in self.model.outputs],
            "metrics": self.model.metrics,
            "provenance": self.model.provenance,
        }
        self.model_summary.setPlainText(json.dumps(summary, indent=2, default=str))
        self._populate_query()
        self._plot_learning_curve()

    def _populate_query(self) -> None:
        assert self.model is not None
        self.query_table.setRowCount(len(self.model.features))
        for row, feature in enumerate(self.model.features):
            self.query_table.setItem(row, 0, QTableWidgetItem(feature.name))
            self.query_table.setItem(row, 1, QTableWidgetItem(str(feature.mean)))
            self.query_table.setItem(row, 2, QTableWidgetItem(feature.unit))

    def query_manual(self) -> None:
        """Evaluate the currently entered single-row query."""

        if self.model is None:
            self.query_result.setPlainText("Import a neural model before querying.")
            return
        record: dict[str, object] = {}
        for row, feature in enumerate(self.model.features):
            item = self.query_table.item(row, 1)
            if item is None:
                raise ValueError(f"missing manual value for {feature.name}")
            record[feature.name] = item.text()
        result = predict_frame(self.model, pd.DataFrame([record]))
        rendered = result.frame.to_json(orient="records", indent=2)
        self.query_result.setPlainText(rendered + "\n" + "\n".join(result.warnings))

    def predict_current_dataset(self) -> pd.DataFrame:
        """Evaluate every retained row of the current dataset."""

        if self.model is None:
            raise ValueError("import a neural model before batch prediction")
        result = predict_frame(self.model, self.current_frame)
        self.predictions = result.frame
        self.export_predictions_button.setEnabled(True)
        status = f"Predicted {len(result.frame):,} rows.\n"
        self.query_result.setPlainText(status + "\n".join(result.warnings))
        output: pd.DataFrame = result.frame
        return output

    def _plot_learning_curve(self) -> None:
        assert self.model is not None
        self.learning_figure.clear()
        axis = self.learning_figure.add_subplot(111)
        curve = self.model.learning_curve
        if curve:
            if "training_fraction" in curve[0]:
                x_values = [item.get("training_fraction") for item in curve]
                losses = [item.get("validation_standardized_rmse") for item in curve]
                axis.plot(x_values, losses, marker="o", label="Validation")
                axis.set_xlabel("Training data fraction (unitless)")
            else:
                x_values = [item.get("epoch") for item in curve]
                axis.plot(
                    x_values,
                    [item.get("trainLoss") for item in curve],
                    label="Train",
                )
                axis.plot(
                    x_values,
                    [item.get("validationLoss") for item in curve],
                    label="Validation",
                )
                axis.set_xlabel("Epoch (count)")
            axis.legend()
        axis.set_ylabel("Loss (bundle-defined target units²)")
        axis.grid(alpha=0.25)
        self.learning_canvas.draw_idle()

    def _export_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Neural Predictions",
            "predictions.csv",
            "CSV (*.csv);;JSON (*.json)",
        )
        if not selected or self.predictions is None:
            return
        path = Path(selected)
        if path.suffix.lower() == ".json":
            content = self.predictions.to_json(orient="records", indent=2) + "\n"
            path.write_text(content, encoding="utf-8")
        else:
            self.predictions.to_csv(path, index=False)

    @staticmethod
    def _help_html() -> str:
        return (
            "<h2>Evidence-bounded neural surrogates</h2><ol>"
            "<li>Select a shot-level dataset and vendor-comparable target.</li>"
            "<li>Select launch-condition inputs and measured outcome targets. "
            "Do not include an outcome as its own input.</li>"
            "<li>Choose architecture, regularization, seed, epoch cap, and a "
            "held-out fraction.</li>"
            "<li>Train in the private repository; review validation loss and "
            "holdout metrics before importing the JSON bundle.</li>"
            "<li>Query manually or batch-predict the current dataset. Treat "
            "out-of-range warnings as extrapolation.</li></ol>"
            "<p><b>Limitations:</b> a surrogate approximates relationships in "
            "available observations. It is not vendor firmware, certification, "
            "causal evidence, or proof of the vendor's internal physics model. "
            "Foresight and FlightScope remain unavailable for training until "
            "reusable shot-level targets exist.</p>"
        )


__all__ = ["NeuralModelOutputs"]
