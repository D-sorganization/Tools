"""Regression tests for shared-library GUI boundary leaks."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse(relative_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.AST = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _function_calls(tree: ast.Module, function_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {
                _call_name(child)
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
            }
    raise AssertionError(f"{function_name} not found")


def test_polynomial_generator_raises_domain_errors_not_dialogs() -> None:
    tree = _parse("src/shared/python/signal_toolkit/polynomial_generator.py")
    calls = _function_calls(tree, "_fit_polynomial_or_raise") | _function_calls(
        tree, "_generate_from_equation_or_raise"
    )

    assert "QtWidgets.QMessageBox.warning" not in calls
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "PolynomialFitError"
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "PolynomialGenerationError"
        for node in ast.walk(tree)
    )


def test_signal_generation_slot_does_not_open_message_box() -> None:
    tree = _parse("src/shared/python/signal_toolkit/widget_processing.py")
    calls = _function_calls(tree, "_generate_signal") | _function_calls(
        tree, "_generate_signal_or_raise"
    )

    assert "QMessageBox.warning" not in calls
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "SignalGenerationError"
        for node in ast.walk(tree)
    )


def test_model_explorer_requires_injected_file_selector() -> None:
    tree = _parse("src/shared/python/model_generation/explorer/model_explorer.py")
    calls = _function_calls(tree, "_load_from_file")

    assert "QFileDialog.getOpenFileName" not in calls
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "ModelFileSelectionRequiredError"
        for node in ast.walk(tree)
    )
