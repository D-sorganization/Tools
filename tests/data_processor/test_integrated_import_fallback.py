"""Regression tests for integrated app import compatibility.

NOTE: Data_Processor_Integrated.py has been archived to archive/ as part
of Phase 2 DRY consolidation. The modern PyQt6 version is the canonical
implementation. This test is retained as a skip sentinel.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="Legacy Data_Processor_Integrated.py archived to archive/; "
    "PyQt6 version is canonical."
)
def test_integrated_module_imports_without_legacy_r0() -> None:
    """Legacy test — module has been archived."""
