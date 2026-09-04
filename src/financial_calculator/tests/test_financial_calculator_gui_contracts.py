# ruff: noqa: E501
# mypy: disable-error-code="no-untyped-def"
"""Focused financial calculator GUI contract coverage."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FINANCIAL_PYTHON = _REPO_ROOT / "src" / "financial_calculator" / "python"
sys.path.insert(0, str(_FINANCIAL_PYTHON))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import financial_calculator as _financial_calculator_package

_financial_calculator_package.__path__.append(
    str(_FINANCIAL_PYTHON / "financial_calculator")
)


def _valid_calculate_args() -> dict[str, float | int]:
    return {
        "plant_capacity": 100.0,
        "operating_days": 330,
        "utilization": 85.0,
        "product_price": 500.0,
        "feedstock_cost": 200.0,
        "labor_cost": 30.0,
        "utilities_cost": 40.0,
        "maintenance_cost": 15.0,
        "fixed_labor": 500000.0,
        "insurance": 100000.0,
        "capital": 10000000.0,
        "debt_ratio": 60.0,
        "interest_rate": 7.0,
        "depreciation_years": 10,
        "tax_rate": 25.0,
    }


@pytest.fixture(autouse=True)
def _reset_theme_manager():
    """Reset shared Qt theme state so tests do not retain deleted windows."""
    from shared.python.theme.theme_manager import ThemeManager

    ThemeManager.reset_instance()
    yield
    ThemeManager.reset_instance()


@pytest.fixture
def app():
    """Create QApplication for tests."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_calculate_returns_financial_design_for_valid_inputs():
    """calculate() maps sidekick results into the GUI design dataclass."""
    from financial_calculator.ui.pyqt6.main_window import FinancialCalculatorEngine

    result = FinancialCalculatorEngine().calculate(**_valid_calculate_args())

    assert result.annual_feedstock_tons > 0
    assert result.total_revenue > 0
    assert result.total_costs > 0
    assert result.roe == pytest.approx(result.roe)


def test_generate_projections_returns_list_after_calculation():
    """generate_projections() returns a concrete list for valid year counts."""
    from financial_calculator.ui.pyqt6.main_window import FinancialCalculatorEngine

    engine = FinancialCalculatorEngine()
    engine.calculate(**_valid_calculate_args())

    projections = engine.generate_projections(years=3)

    assert len(projections) == 3
    assert projections[0]["year"] == 1


def test_update_results_formats_summary_labels(app):
    """_update_results renders each display field from a valid design."""
    from financial_calculator.ui.pyqt6.main_window import (
        FinancialCalculatorMainWindow,
        FinancialDesign,
    )

    window = FinancialCalculatorMainWindow()
    window._update_results(
        FinancialDesign(
            annual_feedstock_tons=12345.6,
            total_revenue=987654.3,
            total_costs=456789.1,
            net_income=530865.2,
            ebitda=600000.0,
            roe=13.37,
            payback_years=4.2,
        )
    )

    assert window.metric_labels["annual_tons"].text() == "12,346 tons"
    assert window.metric_labels["revenue"].text() == "$987,654"
    assert window.metric_labels["roe"].text() == "13.4%"


def test_update_projections_populates_table_rows(app):
    """_update_projections writes projection values to the Qt table."""
    from financial_calculator.ui.pyqt6.main_window import FinancialCalculatorMainWindow

    window = FinancialCalculatorMainWindow()
    window._update_projections(
        [
            {
                "year": 1,
                "total_revenue": 1000.0,
                "total_costs": 400.0,
                "ebitda": 300.0,
                "net_income": 200.0,
                "cumulative_cash_flow": 150.0,
            }
        ]
    )

    assert window.projections_table.rowCount() == 1
    assert window.projections_table.item(0, 0).text() == "1"
    assert window.projections_table.item(0, 5).text() == "$150"


def test_on_calculate_updates_summary_and_projection_table(app):
    """_on_calculate drives the engine and refreshes visible outputs."""
    from financial_calculator.ui.pyqt6.main_window import FinancialCalculatorMainWindow

    window = FinancialCalculatorMainWindow()
    window._on_calculate()

    assert window.metric_labels["annual_tons"].text() != "0 tons"
    assert window.metric_labels["revenue"].text() != "$0"
    assert window.projections_table.rowCount() == 10


def test_toggle_notes_attaches_and_toggles_dock(app, monkeypatch):
    """_toggle_notes attaches the optional notes dock when integration exists."""
    from financial_calculator.ui.pyqt6.main_window import FinancialCalculatorMainWindow

    class FakeDock:
        def __init__(self) -> None:
            self.visible = False

        def isVisible(self) -> bool:
            return self.visible

        def setVisible(self, visible: bool) -> None:
            self.visible = visible

    attached: list[object] = []
    notes_module = types.ModuleType("notes")
    integration_module = types.ModuleType("notes.integration")

    def attach_notes_dock(window, *, project_dir):  # noqa: ANN001
        assert project_dir.name == "financial_calculator"
        dock = FakeDock()
        attached.append(window)
        return dock

    integration_module.__dict__["attach_notes_dock"] = attach_notes_dock
    notes_module.__dict__["integration"] = integration_module
    monkeypatch.setitem(sys.modules, "notes", notes_module)
    monkeypatch.setitem(sys.modules, "notes.integration", integration_module)
    monkeypatch.setitem(sys.modules, "shared.python.notes", notes_module)
    monkeypatch.setitem(sys.modules, "shared.python.notes.integration", integration_module)

    window = FinancialCalculatorMainWindow()
    window._toggle_notes()
    window._toggle_notes()

    assert attached == [window]
    assert window._notes_dock is not None
    assert not window._notes_dock.isVisible()
