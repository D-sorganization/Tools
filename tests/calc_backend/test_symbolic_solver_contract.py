"""Contract tests for the Sidekick symbolic calculator backend."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from calc_backend.app import app  # noqa: E402
from calc_backend.symbolic import SymbolicMathService  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


def test_symbolic_solver_solves_equation_and_returns_latex() -> None:
    pytest.importorskip("sympy", reason="sympy not installed")

    response = client.post(
        "/api/calc/symbolic/solve",
        json={"equations": ["x**2 - 4"], "symbols": ["x"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["backend"] == "sympy"
    assert data["solutions"] == [{"x": "-2"}, {"x": "2"}]
    assert data["rendered"][0]["latex"] == "x^{2} - 4"
    assert data["rendered"][0]["display_text"] == "x**2 - 4"
    assert {workflow["id"] for workflow in data["workflows"]} >= {
        "equation",
        "system",
        "substitution",
    }


def test_symbolic_solver_reports_missing_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        if name == "sympy":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    service = SymbolicMathService()

    result = service.solve({"equations": ["x - 1"], "symbols": ["x"]})

    assert result.success is False
    assert result.backend == "unavailable"
    assert "Install optional dependency sympy" in result.message
    assert result.solutions == []


def test_symbolic_render_contract_escapes_display_text_and_renders_matrix() -> None:
    pytest.importorskip("sympy", reason="sympy not installed")

    response = client.post(
        "/api/calc/symbolic/render",
        json={"expressions": ["Matrix([[1, 2], [3, 4]])", "x < y"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert r"\begin{matrix}" in data["rendered"][0]["latex"]
    assert data["rendered"][1]["display_text"] == "x &lt; y"


def test_symbolic_solver_rejects_inputs_before_backend_execution() -> None:
    response = client.post(
        "/api/calc/symbolic/solve",
        json={
            "equations": ["x" * 260],
            "symbols": ["x"],
            "limits": {"max_expression_chars": 80},
        },
    )

    assert response.status_code == 422
    assert "expression exceeds max_expression_chars" in response.json()["detail"]
