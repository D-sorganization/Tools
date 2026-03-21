"""Financial Calculator PyQt6 Main Window.

Provides a comprehensive GUI for financial modeling of plant operations
using the Catppuccin Mocha dark theme.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Catppuccin Mocha colors
COLORS = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "subtext1": "#bac2de",
    "blue": "#89b4fa",
    "green": "#a6e3a1",
    "red": "#f38ba8",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "mauve": "#cba6f7",
    "teal": "#94e2d5",
    "lavender": "#b4befe",
}

# LoD: extract deep enum references to module-level constants (avoids obj.prop.subprop chains)
_SCROLLBAR_OFF = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
_BOLD_WEIGHT = QFont.Weight.Bold


def get_stylesheet() -> str:
    """Generate Catppuccin Mocha stylesheet."""
    return f"""
        QMainWindow, QWidget {{
            background-color: {COLORS["base"]};
            color: {COLORS["text"]};
        }}
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {COLORS["surface1"]};
            border-radius: 6px;
            margin-top: 12px;
            padding: 10px;
            background-color: {COLORS["mantle"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {COLORS["lavender"]};
        }}
        QLabel {{
            color: {COLORS["text"]};
        }}
        QSpinBox, QDoubleSpinBox {{
            background-color: {COLORS["surface0"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["surface1"]};
            border-radius: 4px;
            padding: 4px 8px;
            min-width: 100px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {COLORS["blue"]};
        }}
        QPushButton {{
            background-color: {COLORS["blue"]};
            color: {COLORS["crust"]};
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {COLORS["lavender"]};
        }}
        QPushButton:pressed {{
            background-color: {COLORS["mauve"]};
        }}
        QTableWidget {{
            background-color: {COLORS["mantle"]};
            color: {COLORS["text"]};
            gridline-color: {COLORS["surface1"]};
            border: 1px solid {COLORS["surface1"]};
            border-radius: 4px;
        }}
        QTableWidget::item {{
            padding: 5px;
        }}
        QHeaderView::section {{
            background-color: {COLORS["surface0"]};
            color: {COLORS["text"]};
            padding: 8px;
            border: none;
            font-weight: bold;
        }}
        QTabWidget::pane {{
            border: 1px solid {COLORS["surface1"]};
            border-radius: 4px;
            background-color: {COLORS["mantle"]};
        }}
        QTabBar::tab {{
            background-color: {COLORS["surface0"]};
            color: {COLORS["text"]};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            background-color: {COLORS["blue"]};
            color: {COLORS["crust"]};
        }}
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
    """


@dataclass
class FinancialDesign:
    """Results container for display."""

    annual_feedstock_tons: float = 0.0
    total_revenue: float = 0.0
    total_costs: float = 0.0
    net_income: float = 0.0
    ebitda: float = 0.0
    roe: float = 0.0
    payback_years: float = 0.0


class FinancialCalculatorEngine:
    """Wrapper for the financial calculator engine."""

    def __init__(self) -> None:
        """Initialize the engine."""
        from upstream_drift_tools.process_calculators.financial_calculator import (
            FinancialModelCalculator,
            FinancialParameters,
        )

        self._calculator = FinancialModelCalculator()
        self._params_class = FinancialParameters

    def calculate(
        self,
        plant_capacity: float,
        operating_days: int,
        utilization: float,
        product_price: float,
        feedstock_cost: float,
        labor_cost: float,
        utilities_cost: float,
        maintenance_cost: float,
        fixed_labor: float,
        insurance: float,
        capital: float,
        debt_ratio: float,
        interest_rate: float,
        depreciation_years: int,
        tax_rate: float,
    ) -> FinancialDesign:
        """Run financial calculation.

        Preconditions:
            plant_capacity: non-negative float (tons per day)
            operating_days: integer in [0, 365]
            utilization: float in [0, 100] (percent)
            product_price: non-negative float
            feedstock_cost: non-negative float
            labor_cost: non-negative float
            utilities_cost: non-negative float
            maintenance_cost: non-negative float
            fixed_labor: non-negative float
            insurance: non-negative float
            capital: non-negative float
            debt_ratio: float in [0, 100] (percent)
            interest_rate: float in [0, 100] (percent)
            depreciation_years: positive integer
            tax_rate: float in [0, 100] (percent)

        Raises:
            TypeError: if any argument has the wrong type.
            ValueError: if any numeric argument is out of valid range.
        """
        if not isinstance(plant_capacity, (int, float)):
            raise TypeError(
                f"plant_capacity must be a number, got {type(plant_capacity)}"
            )
        if plant_capacity < 0:
            raise ValueError(
                f"plant_capacity must be non-negative, got {plant_capacity}"
            )
        if not isinstance(operating_days, int):
            raise TypeError(
                f"operating_days must be an int, got {type(operating_days)}"
            )
        if not 0 <= operating_days <= 365:
            raise ValueError(
                f"operating_days must be in [0, 365], got {operating_days}"
            )
        if not isinstance(utilization, (int, float)):
            raise TypeError(f"utilization must be a number, got {type(utilization)}")
        if not 0 <= utilization <= 100:
            raise ValueError(f"utilization must be in [0, 100], got {utilization}")
        if not isinstance(product_price, (int, float)):
            raise TypeError(
                f"product_price must be a number, got {type(product_price)}"
            )
        if product_price < 0:
            raise ValueError(f"product_price must be non-negative, got {product_price}")
        if not isinstance(depreciation_years, int):
            raise TypeError(
                f"depreciation_years must be an int, got {type(depreciation_years)}"
            )
        if depreciation_years <= 0:
            raise ValueError(
                f"depreciation_years must be positive, got {depreciation_years}"
            )

        params = self._params_class(
            plant_capacity_tpd=plant_capacity,
            operating_days_per_year=operating_days,
            capacity_utilization=utilization / 100,
            product_price_per_ton=product_price,
            byproduct_revenue_per_ton=50,
            byproduct_yield_factor=0.1,
            feedstock_cost_per_ton=feedstock_cost,
            labor_cost_per_ton=labor_cost,
            utilities_cost_per_ton=utilities_cost,
            maintenance_cost_per_ton=maintenance_cost,
            consumables_cost_per_ton=10,
            fixed_labor_cost_annual=fixed_labor,
            insurance_annual=insurance,
            property_tax_annual=50000,
            admin_overhead_annual=200000,
            total_capital_investment=capital,
            debt_ratio=debt_ratio / 100,
            interest_rate=interest_rate / 100,
            depreciation_years=depreciation_years,
            tax_rate=tax_rate / 100,
        )

        results = self._calculator.calculate_financial_model(params)

        return FinancialDesign(
            annual_feedstock_tons=results.annual_feedstock_tons,
            total_revenue=results.total_revenue,
            total_costs=results.total_variable_costs + results.total_fixed_costs,
            net_income=results.net_income,
            ebitda=results.ebitda,
            roe=results.roe * 100,
            payback_years=results.payback_period_years,
        )

    def generate_projections(self, years: int = 10) -> list[dict]:
        """Generate yearly projections.

        Preconditions:
            years: positive integer.

        Raises:
            TypeError: if years is not an integer.
            ValueError: if years is not positive.
        """
        if not isinstance(years, int):
            raise TypeError(f"years must be an int, got {type(years)}")
        if years <= 0:
            raise ValueError(f"years must be positive, got {years}")
        result = self._calculator.generate_yearly_projections(years)
        return list(result)


class FinancialCalculatorMainWindow(QMainWindow):
    """Main window for Financial Calculator application."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the main window.

        Preconditions:
            parent: a QWidget instance or None.

        Raises:
            TypeError: if parent is not a QWidget or None.
        """
        if parent is not None and not isinstance(parent, QWidget):
            raise TypeError(f"parent must be a QWidget or None, got {type(parent)}")
        super().__init__(parent)
        self.setWindowTitle("Financial Calculator")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(get_stylesheet())

        self.engine = FinancialCalculatorEngine()
        self._notes_dock: QWidget | None = None

        self._setup_ui()

    # -- Notes integration (shared workspace) --
    def _toggle_notes(self) -> None:
        """Show/hide the shared notes dock widget."""
        try:
            from pathlib import Path

            from notes.integration import attach_notes_dock
        except ImportError:
            return
        if self._notes_dock is None:
            project_dir = Path(__file__).resolve().parents[4]
            self._notes_dock = attach_notes_dock(self, project_dir=project_dir)
        self._notes_dock.setVisible(not self._notes_dock.isVisible())

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Menu bar with Notes toggle — break into named intermediates (LoD)
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("&View")  # type: ignore[union-attr]
        notes_action = view_menu.addAction("Toggle &Notes")  # type: ignore[union-attr]
        notes_action.triggered.connect(self._toggle_notes)  # type: ignore[union-attr]

        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Left panel - inputs
        left_panel = self._create_input_panel()
        layout.addWidget(left_panel, stretch=1)

        # Right panel - results
        right_panel = self._create_results_panel()
        layout.addWidget(right_panel, stretch=1)

    def _create_input_panel(self) -> QWidget:
        """Create the input panel with all parameters."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # LoD: use module-level constant instead of Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        scroll.setHorizontalScrollBarPolicy(_SCROLLBAR_OFF)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        title = QLabel("Financial Model Calculator")
        # LoD: use module-level constant instead of QFont.Weight.Bold
        title.setFont(QFont("", 18, _BOLD_WEIGHT))
        title.setStyleSheet(f"color: {COLORS['blue']};")
        layout.addWidget(title)

        layout.addWidget(self._create_operations_group())
        layout.addWidget(self._create_revenue_group())
        layout.addWidget(self._create_variable_costs_group())
        layout.addWidget(self._create_fixed_costs_group())
        layout.addWidget(self._create_capital_group())

        self.calculate_btn = QPushButton("Calculate Financial Model")
        self.calculate_btn.setMinimumHeight(50)
        self.calculate_btn.clicked.connect(self._on_calculate)
        layout.addWidget(self.calculate_btn)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _create_operations_group(self) -> QGroupBox:
        """Create the plant operations input group."""
        group = QGroupBox("Plant Operations")
        form = QFormLayout(group)

        self.plant_capacity_input = QDoubleSpinBox()
        self.plant_capacity_input.setRange(0, 10000)
        self.plant_capacity_input.setValue(100)
        self.plant_capacity_input.setSuffix(" TPD")
        form.addRow("Plant Capacity:", self.plant_capacity_input)

        self.operating_days_input = QSpinBox()
        self.operating_days_input.setRange(0, 365)
        self.operating_days_input.setValue(330)
        self.operating_days_input.setSuffix(" days/yr")
        form.addRow("Operating Days:", self.operating_days_input)

        self.utilization_input = QDoubleSpinBox()
        self.utilization_input.setRange(0, 100)
        self.utilization_input.setValue(85)
        self.utilization_input.setSuffix(" %")
        form.addRow("Capacity Utilization:", self.utilization_input)

        return group

    def _create_revenue_group(self) -> QGroupBox:
        """Create the revenue parameters input group."""
        group = QGroupBox("Revenue Parameters")
        form = QFormLayout(group)

        self.product_price_input = QDoubleSpinBox()
        self.product_price_input.setRange(0, 10000)
        self.product_price_input.setValue(500)
        self.product_price_input.setPrefix("$")
        self.product_price_input.setSuffix("/ton")
        form.addRow("Product Price:", self.product_price_input)

        return group

    def _create_variable_costs_group(self) -> QGroupBox:
        """Create the variable costs input group."""
        group = QGroupBox("Variable Costs ($/ton)")
        form = QFormLayout(group)

        self.feedstock_cost_input = QDoubleSpinBox()
        self.feedstock_cost_input.setRange(0, 5000)
        self.feedstock_cost_input.setValue(200)
        self.feedstock_cost_input.setPrefix("$")
        form.addRow("Feedstock Cost:", self.feedstock_cost_input)

        self.labor_cost_input = QDoubleSpinBox()
        self.labor_cost_input.setRange(0, 1000)
        self.labor_cost_input.setValue(30)
        self.labor_cost_input.setPrefix("$")
        form.addRow("Variable Labor:", self.labor_cost_input)

        self.utilities_cost_input = QDoubleSpinBox()
        self.utilities_cost_input.setRange(0, 1000)
        self.utilities_cost_input.setValue(40)
        self.utilities_cost_input.setPrefix("$")
        form.addRow("Utilities:", self.utilities_cost_input)

        self.maintenance_cost_input = QDoubleSpinBox()
        self.maintenance_cost_input.setRange(0, 500)
        self.maintenance_cost_input.setValue(15)
        self.maintenance_cost_input.setPrefix("$")
        form.addRow("Maintenance:", self.maintenance_cost_input)

        return group

    def _create_fixed_costs_group(self) -> QGroupBox:
        """Create the fixed costs input group."""
        group = QGroupBox("Fixed Costs ($/year)")
        form = QFormLayout(group)

        self.fixed_labor_input = QDoubleSpinBox()
        self.fixed_labor_input.setRange(0, 10000000)
        self.fixed_labor_input.setValue(500000)
        self.fixed_labor_input.setPrefix("$")
        self.fixed_labor_input.setDecimals(0)
        form.addRow("Fixed Labor:", self.fixed_labor_input)

        self.insurance_input = QDoubleSpinBox()
        self.insurance_input.setRange(0, 1000000)
        self.insurance_input.setValue(100000)
        self.insurance_input.setPrefix("$")
        self.insurance_input.setDecimals(0)
        form.addRow("Insurance:", self.insurance_input)

        return group

    def _create_capital_group(self) -> QGroupBox:
        """Create the capital and financing input group."""
        group = QGroupBox("Capital & Financing")
        form = QFormLayout(group)

        self.capital_input = QDoubleSpinBox()
        self.capital_input.setRange(0, 1000000000)
        self.capital_input.setValue(10000000)
        self.capital_input.setPrefix("$")
        self.capital_input.setDecimals(0)
        form.addRow("Total Capital:", self.capital_input)

        self.debt_ratio_input = QDoubleSpinBox()
        self.debt_ratio_input.setRange(0, 100)
        self.debt_ratio_input.setValue(60)
        self.debt_ratio_input.setSuffix(" %")
        form.addRow("Debt Ratio:", self.debt_ratio_input)

        self.interest_rate_input = QDoubleSpinBox()
        self.interest_rate_input.setRange(0, 30)
        self.interest_rate_input.setValue(7)
        self.interest_rate_input.setSuffix(" %")
        form.addRow("Interest Rate:", self.interest_rate_input)

        self.depreciation_input = QSpinBox()
        self.depreciation_input.setRange(1, 40)
        self.depreciation_input.setValue(10)
        self.depreciation_input.setSuffix(" years")
        form.addRow("Depreciation Period:", self.depreciation_input)

        self.tax_rate_input = QDoubleSpinBox()
        self.tax_rate_input.setRange(0, 50)
        self.tax_rate_input.setValue(25)
        self.tax_rate_input.setSuffix(" %")
        form.addRow("Tax Rate:", self.tax_rate_input)

        return group

    def _create_results_panel(self) -> QWidget:
        """Create the results panel."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        # Results title
        title = QLabel("Financial Analysis Results")
        # LoD: use module-level constant instead of QFont.Weight.Bold
        title.setFont(QFont("", 18, _BOLD_WEIGHT))
        title.setStyleSheet(f"color: {COLORS['green']};")
        layout.addWidget(title)

        # Tabs for results
        tabs = QTabWidget()

        # Summary tab
        summary_tab = self._create_summary_tab()
        tabs.addTab(summary_tab, "Summary")

        # Projections tab
        projections_tab = self._create_projections_tab()
        tabs.addTab(projections_tab, "10-Year Projections")

        layout.addWidget(tabs)

        return container

    def _create_summary_tab(self) -> QWidget:
        """Create summary results tab."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        # Key metrics grid
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet(
            f"background-color: {COLORS['surface0']}; "
            f"border-radius: 8px; padding: 15px;"
        )
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setSpacing(20)

        self.metric_labels = {}
        metrics = [
            ("annual_tons", "Annual Feedstock", "0 tons", COLORS["blue"]),
            ("revenue", "Total Revenue", "$0", COLORS["green"]),
            ("costs", "Total Costs", "$0", COLORS["red"]),
            ("net_income", "Net Income", "$0", COLORS["yellow"]),
            ("ebitda", "EBITDA", "$0", COLORS["peach"]),
            ("roe", "Return on Equity", "0%", COLORS["mauve"]),
            ("payback", "Payback Period", "0 years", COLORS["teal"]),
        ]

        for i, (key, label, default, color) in enumerate(metrics):
            row, col = divmod(i, 2)

            name_label = QLabel(label)
            name_label.setStyleSheet(f"color: {COLORS['subtext0']};")

            value_label = QLabel(default)
            # LoD: use module-level constant instead of QFont.Weight.Bold
            value_label.setFont(QFont("", 16, _BOLD_WEIGHT))
            value_label.setStyleSheet(f"color: {color};")
            self.metric_labels[key] = value_label

            cell = QVBoxLayout()
            cell.addWidget(name_label)
            cell.addWidget(value_label)
            metrics_layout.addLayout(cell, row, col)

        layout.addWidget(metrics_frame)
        layout.addStretch()

        return container

    def _create_projections_tab(self) -> QWidget:
        """Create projections table tab."""
        container = QWidget()
        layout = QVBoxLayout(container)

        self.projections_table = QTableWidget()
        self.projections_table.setColumnCount(6)
        self.projections_table.setHorizontalHeaderLabels(
            ["Year", "Revenue", "Costs", "EBITDA", "Net Income", "Cumulative CF"]
        )
        header = self.projections_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)

        layout.addWidget(self.projections_table)

        return container

    def _on_calculate(self) -> None:
        """Handle calculate button click."""
        results = self.engine.calculate(
            plant_capacity=self.plant_capacity_input.value(),
            operating_days=self.operating_days_input.value(),
            utilization=self.utilization_input.value(),
            product_price=self.product_price_input.value(),
            feedstock_cost=self.feedstock_cost_input.value(),
            labor_cost=self.labor_cost_input.value(),
            utilities_cost=self.utilities_cost_input.value(),
            maintenance_cost=self.maintenance_cost_input.value(),
            fixed_labor=self.fixed_labor_input.value(),
            insurance=self.insurance_input.value(),
            capital=self.capital_input.value(),
            debt_ratio=self.debt_ratio_input.value(),
            interest_rate=self.interest_rate_input.value(),
            depreciation_years=self.depreciation_input.value(),
            tax_rate=self.tax_rate_input.value(),
        )

        self._update_results(results)

        # Generate and display projections
        projections = self.engine.generate_projections(10)
        self._update_projections(projections)

    def _update_results(self, results: FinancialDesign) -> None:
        """Update results display.

        Preconditions:
            results: a FinancialDesign instance.

        Raises:
            TypeError: if results is not a FinancialDesign.
        """
        if not isinstance(results, FinancialDesign):
            raise TypeError(f"results must be a FinancialDesign, got {type(results)}")
        self.metric_labels["annual_tons"].setText(
            f"{results.annual_feedstock_tons:,.0f} tons"
        )
        self.metric_labels["revenue"].setText(f"${results.total_revenue:,.0f}")
        self.metric_labels["costs"].setText(f"${results.total_costs:,.0f}")
        self.metric_labels["net_income"].setText(f"${results.net_income:,.0f}")
        self.metric_labels["ebitda"].setText(f"${results.ebitda:,.0f}")
        self.metric_labels["roe"].setText(f"{results.roe:.1f}%")
        self.metric_labels["payback"].setText(f"{results.payback_years:.1f} years")

    def _update_projections(self, projections: list[dict]) -> None:
        """Update projections table.

        Preconditions:
            projections: a list of projection dicts.

        Raises:
            TypeError: if projections is not a list.
        """
        if not isinstance(projections, list):
            raise TypeError(f"projections must be a list, got {type(projections)}")
        self.projections_table.setRowCount(len(projections))

        for row, proj in enumerate(projections):
            self.projections_table.setItem(row, 0, QTableWidgetItem(str(proj["year"])))
            self.projections_table.setItem(
                row, 1, QTableWidgetItem(f"${proj['total_revenue']:,.0f}")
            )
            self.projections_table.setItem(
                row, 2, QTableWidgetItem(f"${proj['total_costs']:,.0f}")
            )
            self.projections_table.setItem(
                row, 3, QTableWidgetItem(f"${proj['ebitda']:,.0f}")
            )
            self.projections_table.setItem(
                row, 4, QTableWidgetItem(f"${proj['net_income']:,.0f}")
            )
            self.projections_table.setItem(
                row, 5, QTableWidgetItem(f"${proj['cumulative_cash_flow']:,.0f}")
            )
