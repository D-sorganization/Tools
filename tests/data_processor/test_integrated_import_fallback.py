"""Regression tests for integrated app import compatibility."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def test_integrated_module_imports_without_legacy_r0(monkeypatch) -> None:
    """Integrated module should import even when legacy r0 module is absent."""
    pytest.importorskip("tkinter")
    pytest.importorskip("customtkinter")
    monkeypatch.delitem(sys.modules, "Data_Processor_r0", raising=False)
    module = importlib.import_module(
        "src.data_processing.data_processor.python.data_processor.Data_Processor_Integrated"
    )
    assert isinstance(module, ModuleType)
    assert hasattr(module, "IntegratedCSVProcessorApp")
