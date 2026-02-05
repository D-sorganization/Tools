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
        """Run financial calculation."""
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
        """Generate yearly projections."""
        result = self._calculator.generate_yearly_projections(years)
        return list(result)


class FinancialCalculatorMainWindow(QMainWindow):
    """Main window for Financial Calculator application."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the main window."""
        super().__init__(parent)
        self.setWindowTitle("Financial Calculator")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(get_stylesheet())

        self.engine = FinancialCalculatorEngine()

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        # Title
        title = QLabel("Financial Model Calculator")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['blue']};")
        layout.addWidget(title)

        # Plant Operations Group
        ops_group = QGroupBox("Plant Operations")
        ops_layout = QFormLayout(ops_group)

        self.plant_capacity_input = QDoubleSpinBox()
        self.plant_capacity_input.setRange(0, 10000)
        self.plant_capacity_input.setValue(100)
        self.plant_capacity_input.setSuffix(" TPD")
        ops_layout.addRow("Plant Capacity:", self.plant_capacity_input)

        self.operating_days_input = QSpinBox()
        self.operating_days_input.setRange(0, 365)
        self.operating_days_input.setValue(330)
        self.operating_days_input.setSuffix(" days/yr")
        ops_layout.addRow("Operating Days:", self.operating_days_input)

        self.utilization_input = QDoubleSpinBox()
        self.utilization_input.setRange(0, 100)
        self.utilization_input.setValue(85)
        self.utilization_input.setSuffix(" %")
        ops_layout.addRow("Capacity Utilization:", self.utilization_input)

        layout.addWidget(ops_group)

        # Revenue Group
        rev_group = QGroupBox("Revenue Parameters")
        rev_layout = QFormLayout(rev_group)

        self.product_price_input = QDoubleSpinBox()
        self.product_price_input.setRange(0, 10000)
        self.product_price_input.setValue(500)
        self.product_price_input.setPrefix("$")
        self.product_price_input.setSuffix("/ton")
        rev_layout.addRow("Product Price:", self.product_price_input)

        layout.addWidget(rev_group)

        # Variable Costs Group
        var_group = QGroupBox("Variable Costs ($/ton)")
        var_layout = QFormLayout(var_group)

        self.feedstock_cost_input = QDoubleSpinBox()
        self.feedstock_cost_input.setRange(0, 5000)
        self.feedstock_cost_input.setValue(200)
        self.feedstock_cost_input.setPrefix("$")
        var_layout.addRow("Feedstock Cost:", self.feedstock_cost_input)

        self.labor_cost_input = QDoubleSpinBox()
        self.labor_cost_input.setRange(0, 1000)
        self.labor_cost_input.setValue(30)
        self.labor_cost_input.setPrefix("$")
        var_layout.addRow("Variable Labor:", self.labor_cost_input)

        self.utilities_cost_input = QDoubleSpinBox()
        self.utilities_cost_input.setRange(0, 1000)
        self.utilities_cost_input.setValue(40)
        self.utilities_cost_input.setPrefix("$")
        var_layout.addRow("Utilities:", self.utilities_cost_input)

        self.maintenance_cost_input = QDoubleSpinBox()
        self.maintenance_cost_input.setRange(0, 500)
        self.maintenance_cost_input.setValue(15)
        self.maintenance_cost_input.setPrefix("$")
        var_layout.addRow("Maintenance:", self.maintenance_cost_input)

        layout.addWidget(var_group)

        # Fixed Costs Group
        fixed_group = QGroupBox("Fixed Costs ($/year)")
        fixed_layout = QFormLayout(fixed_group)

        self.fixed_labor_input = QDoubleSpinBox()
        self.fixed_labor_input.setRange(0, 10000000)
        self.fixed_labor_input.setValue(500000)
        self.fixed_labor_input.setPrefix("$")
        self.fixed_labor_input.setDecimals(0)
        fixed_layout.addRow("Fixed Labor:", self.fixed_labor_input)

        self.insurance_input = QDoubleSpinBox()
        self.insurance_input.setRange(0, 1000000)
        self.insurance_input.setValue(100000)
        self.insurance_input.setPrefix("$")
        self.insurance_input.setDecimals(0)
        fixed_layout.addRow("Insurance:", self.insurance_input)

        layout.addWidget(fixed_group)

        # Capital & Financing Group
        cap_group = QGroupBox("Capital & Financing")
        cap_layout = QFormLayout(cap_group)

        self.capital_input = QDoubleSpinBox()
        self.capital_input.setRange(0, 1000000000)
        self.capital_input.setValue(10000000)
        self.capital_input.setPrefix("$")
        self.capital_input.setDecimals(0)
        cap_layout.addRow("Total Capital:", self.capital_input)

        self.debt_ratio_input = QDoubleSpinBox()
        self.debt_ratio_input.setRange(0, 100)
        self.debt_ratio_input.setValue(60)
        self.debt_ratio_input.setSuffix(" %")
        cap_layout.addRow("Debt Ratio:", self.debt_ratio_input)

        self.interest_rate_input = QDoubleSpinBox()
        self.interest_rate_input.setRange(0, 30)
        self.interest_rate_input.setValue(7)
        self.interest_rate_input.setSuffix(" %")
        cap_layout.addRow("Interest Rate:", self.interest_rate_input)

        self.depreciation_input = QSpinBox()
        self.depreciation_input.setRange(1, 40)
        self.depreciation_input.setValue(10)
        self.depreciation_input.setSuffix(" years")
        cap_layout.addRow("Depreciation Period:", self.depreciation_input)

        self.tax_rate_input = QDoubleSpinBox()
        self.tax_rate_input.setRange(0, 50)
        self.tax_rate_input.setValue(25)
        self.tax_rate_input.setSuffix(" %")
        cap_layout.addRow("Tax Rate:", self.tax_rate_input)

        layout.addWidget(cap_group)

        # Calculate Button
        self.calculate_btn = QPushButton("Calculate Financial Model")
        self.calculate_btn.setMinimumHeight(50)
        self.calculate_btn.clicked.connect(self._on_calculate)
        layout.addWidget(self.calculate_btn)

        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _create_results_panel(self) -> QWidget:
        """Create the results panel."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        # Results title
        title = QLabel("Financial Analysis Results")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
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
            value_label.setFont(QFont("", 16, QFont.Weight.Bold))
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
        """Update results display."""
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
        """Update projections table."""
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
