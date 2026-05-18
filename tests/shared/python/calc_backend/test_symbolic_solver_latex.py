import pytest
from calc_backend.app import app
from calc_backend.routers.symbolic_solver import SYMPY_AVAILABLE
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not SYMPY_AVAILABLE, reason="SymPy is required for these tests"
)

client = TestClient(app)


def test_solve_equation_latex_output():
    """Verify solve equation returns LaTeX fields."""
    response = client.post(
        "/api/calc/symbolic/solve",
        json={"equation": "x**2 - 4 = 0", "variable": "x"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert "error" not in data or data["error"] is None

    # We should have two solutions: 2 and -2
    assert len(data["solutions"]) == 2
    assert "-2" in data["solutions"]
    assert "2" in data["solutions"]

    # Check latex outputs
    assert len(data["latex_solutions"]) == 2
    assert data["latex"] is not None
    assert (
        "{" in data["latex"] or "[" in data["latex"]
    )  # Usually \left[ ... \right] or similar


def test_derivative_latex_output():
    """Verify derivative returns LaTeX field."""
    response = client.post(
        "/api/calc/symbolic/derivative",
        json={"expression": "x**3 + 2*x", "variable": "x"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert "error" not in data or data["error"] is None

    # Derivative is 3*x**2 + 2
    assert "3*x**2 + 2" in data["derivative"]

    # Check latex output
    assert data["latex"] is not None
    assert "x^{2}" in data["latex"]


def test_simplify_latex_output():
    """Verify simplify returns LaTeX field."""
    response = client.post(
        "/api/calc/symbolic/simplify",
        json={"expression": "(x**2 - 1)/(x - 1)"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert "error" not in data or data["error"] is None

    # Simplified is x + 1
    assert data["simplified"] == "x + 1"

    # Check latex output
    assert data["latex"] is not None
    assert "x + 1" in data["latex"]
