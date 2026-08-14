"""Architecture fitness tests for GH1696 — large __init__.py god module refactor.

Verifies that:
- pressure_drop_calculator __init__.py no longer contains calculation logic
- pressure_drop_calculator legacy API is preserved via _legacy.py
- signal_toolkit uses lazy imports (submodules not loaded at package import time)
- signal_toolkit public API is fully accessible after the lazy-import refactor
- __init__.py line-count regression guards prevent future accumulation

Story: GH1696 — Assessment: Large __init__.py files act as god modules (up to 611 lines)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
# Canonical package is ``shared.python.sidekick``; ``upstream_drift_tools`` is
# now an alias shim holding only an ``__init__.py``. These path constants were
# left behind by that rename, so the three filesystem-backed guards below have
# been failing against main ever since — invisibly, because changed-file test
# selection only ran this file when someone edited the file itself.
PDC_INIT = (
    REPO_ROOT
    / "src/shared/python/sidekick/process_calculators"
    / "pressure_drop_calculator/__init__.py"
)
PDC_LEGACY = (
    REPO_ROOT
    / "src/shared/python/sidekick/process_calculators"
    / "pressure_drop_calculator/_legacy.py"
)
SIGNAL_TOOLKIT_INIT = REPO_ROOT / "src/shared/python/signal_toolkit/__init__.py"
SIGNAL_TOOLKIT_LAZY_MAP = REPO_ROOT / "src/shared/python/signal_toolkit/_lazy_map.py"


# ---------------------------------------------------------------------------
# pressure_drop_calculator: logic-in-init regression guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pressure_drop_init_has_no_calculation_logic() -> None:
    """__init__.py must not define classes or functions inline.

    Calculation logic belongs in _legacy.py, not in __init__.py.
    """
    source = PDC_INIT.read_text()
    # The class definition should not appear in __init__.py any more
    assert "class PressureDropCalculator:" not in source, (
        "PressureDropCalculator class definition found in __init__.py — "
        "it must live in _legacy.py"
    )
    assert "class PressureDropResult:" not in source, (
        "PressureDropResult class definition found in __init__.py — "
        "it must live in _legacy.py"
    )


@pytest.mark.unit
def test_pressure_drop_legacy_module_exists() -> None:
    """_legacy.py must exist alongside __init__.py."""
    assert PDC_LEGACY.exists(), (
        f"_legacy.py not found at {PDC_LEGACY}. "
        "Legacy API must be extracted to this file."
    )


@pytest.mark.unit
def test_pressure_drop_legacy_api_importable() -> None:
    """All legacy symbols must be importable from the package top level."""
    from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
        PIPE_DIMENSIONS_SCH40,
        ROUGHNESS_VALUES,
        PressureDropCalculator,
        PressureDropResult,
    )

    assert isinstance(PIPE_DIMENSIONS_SCH40, dict)
    assert isinstance(ROUGHNESS_VALUES, dict)
    assert PressureDropCalculator is not None
    assert PressureDropResult is not None


@pytest.mark.unit
def test_pressure_drop_pipe_dimensions_values() -> None:
    """PIPE_DIMENSIONS_SCH40 must contain correct 4-inch Schedule 40 ID."""
    from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
        PIPE_DIMENSIONS_SCH40,
    )

    assert '4"' in PIPE_DIMENSIONS_SCH40
    assert abs(PIPE_DIMENSIONS_SCH40['4"'] - 0.10226) < 1e-6


@pytest.mark.unit
def test_pressure_drop_roughness_values() -> None:
    """ROUGHNESS_VALUES must contain commercial steel roughness."""
    from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
        ROUGHNESS_VALUES,
    )

    assert "Commercial Steel" in ROUGHNESS_VALUES
    assert abs(ROUGHNESS_VALUES["Commercial Steel"] - 0.000045) < 1e-9


@pytest.mark.unit
def test_pressure_drop_calculator_legacy_compute() -> None:
    """PressureDropCalculator.calculate_pressure_drop must return correct regime."""
    from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
        PressureDropCalculator,
    )

    calc = PressureDropCalculator()
    result = calc.calculate_pressure_drop(
        pipe_diameter_m=0.10226,  # 4" Sch40
        pipe_length_m=50.0,
        roughness_m=0.000045,
        flow_rate_kg_s=1.0,
        temperature_k=700.0,
        pressure_pa=10e5,
        molecular_weight_kg_mol=0.022,  # ~syngas
    )

    assert result.pressure_drop_pa >= 0.0
    assert result.flow_regime in {"Laminar", "Transitional", "Turbulent"}
    assert result.reynolds_number >= 0.0
    assert result.velocity >= 0.0


@pytest.mark.unit
def test_pressure_drop_calculator_raises_on_zero_diameter() -> None:
    """PressureDropCalculator must raise ValueError for non-positive diameter."""
    from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
        PressureDropCalculator,
    )

    calc = PressureDropCalculator()
    with pytest.raises(ValueError, match="pipe_diameter_m must be > 0"):
        calc.calculate_pressure_drop(
            pipe_diameter_m=0.0,
            pipe_length_m=10.0,
            roughness_m=0.000045,
            flow_rate_kg_s=1.0,
            temperature_k=300.0,
            pressure_pa=1e5,
            molecular_weight_kg_mol=0.029,
        )


@pytest.mark.unit
def test_pressure_drop_result_is_dataclass() -> None:
    """PressureDropResult must be constructable as a plain dataclass."""
    from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
        PressureDropResult,
    )

    r = PressureDropResult(
        pressure_drop_pa=100.0,
        reynolds_number=50000.0,
        friction_factor=0.02,
        velocity=5.0,
        flow_regime="Turbulent",
        density=1.2,
        viscosity=1.8e-5,
    )
    assert r.flow_regime == "Turbulent"
    assert r.pressure_drop_pa == 100.0


# ---------------------------------------------------------------------------
# pressure_drop_calculator: line-count regression guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pressure_drop_init_line_count() -> None:
    """pressure_drop_calculator/__init__.py must stay under 200 lines.

    The legacy logic (144 lines) now lives in _legacy.py. If someone adds
    it back inline, this test will catch the regression.
    """
    lines = PDC_INIT.read_text().splitlines()
    assert len(lines) <= 200, (
        f"pressure_drop_calculator/__init__.py has {len(lines)} lines — "
        "exceeded 200-line guard. Have calculation classes been re-added inline?"
    )


# ---------------------------------------------------------------------------
# signal_toolkit: lazy import tests
# ---------------------------------------------------------------------------


def _run_in_fresh_python(script: str) -> tuple[int, str, str]:
    """Run *script* in a fresh subprocess with the repo's src paths on PYTHONPATH."""
    import os
    import subprocess

    # Build PYTHONPATH from known src directories used by pytest.ini
    repo_root = Path(__file__).resolve().parents[2]
    extra_paths = [
        str(repo_root / "src" / "shared" / "python"),
        str(repo_root / "src"),
        str(repo_root / "src" / "python" / "src"),
        str(repo_root / "src" / "data_processing" / "data_processor" / "python"),
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.unit
def test_signal_toolkit_uses_lazy_import_pattern() -> None:
    """signal_toolkit must use a lazy-import dispatch table with __getattr__.

    After the issue-#1696 refactor the dispatch table lives in _lazy_map.py
    (as ``LAZY``) and is imported by ``__init__.py``, which still defines
    ``__getattr__`` backed by ``importlib.import_module``.  This test
    catches regressions where someone replaces the lazy pattern with eager
    imports or removes the dispatch table entirely.
    """
    init_source = SIGNAL_TOOLKIT_INIT.read_text()
    lazy_map_source = SIGNAL_TOOLKIT_LAZY_MAP.read_text()

    # The dispatch table may live in _lazy_map.py (preferred) or inline.
    has_lazy_in_init = "_LAZY" in init_source or "LAZY" in init_source
    has_lazy_in_map = "LAZY" in lazy_map_source
    assert has_lazy_in_init or has_lazy_in_map, (
        "signal_toolkit must contain a LAZY dispatch table in __init__.py "
        "or _lazy_map.py"
    )
    assert SIGNAL_TOOLKIT_LAZY_MAP.exists(), (
        "_lazy_map.py must exist alongside __init__.py (issue #1696 refactor)"
    )
    assert "def __getattr__" in init_source, (
        "signal_toolkit/__init__.py must define __getattr__ for lazy loading"
    )
    assert "importlib.import_module" in init_source, (
        "signal_toolkit/__init__.py must use importlib.import_module in __getattr__"
    )


@pytest.mark.unit
def test_signal_toolkit_lazy_attribute_loads_on_access() -> None:
    """Accessing a lazy attribute that is not pre-loaded must work correctly.

    Verifies the __getattr__ dispatch mechanism produces the correct value.
    After access, the attribute must be cached in the module namespace.
    """
    import signal_toolkit

    # Remove any cached value to force __getattr__ to run
    signal_toolkit.__dict__.pop("SeriesExpansion", None)

    # This name is in _LAZY and unlikely to be loaded by widget side-effects
    # (series module is not imported by any widget code)
    obj = signal_toolkit.SeriesExpansion
    assert obj is not None, "signal_toolkit.SeriesExpansion should not be None"

    # After access, should be cached in globals
    assert "SeriesExpansion" in signal_toolkit.__dict__, (
        "After access, SeriesExpansion must be cached in signal_toolkit.__dict__"
    )


@pytest.mark.unit
def test_signal_toolkit_all_exports_accessible() -> None:
    """All names in __all__ must be accessible (lazy or direct)."""
    import signal_toolkit

    for name in signal_toolkit.__all__:
        attr = getattr(signal_toolkit, name, None)
        # HAS_* flags and optional widgets may be None (no PyQt6 in CI)
        if name not in {"PolynomialGeneratorWidget", "SignalToolkitWidget"}:
            assert attr is not None, (
                f"signal_toolkit.{name} is None — lazy import may be broken"
            )


@pytest.mark.unit
def test_signal_toolkit_core_classes_importable() -> None:
    """Signal and SignalGenerator must be importable via the package."""
    from signal_toolkit import Signal, SignalGenerator

    assert Signal is not None
    assert SignalGenerator is not None


@pytest.mark.unit
def test_signal_toolkit_filter_functions_importable() -> None:
    """Filter factory functions must be importable via the package."""
    from signal_toolkit import apply_filter, create_butterworth_filter

    assert callable(create_butterworth_filter)
    assert callable(apply_filter)


@pytest.mark.unit
def test_signal_toolkit_unknown_attribute_raises() -> None:
    """Accessing an unknown attribute must raise AttributeError, not hang."""
    import signal_toolkit

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = signal_toolkit.this_does_not_exist_gh1696


# ---------------------------------------------------------------------------
# signal_toolkit: line-count regression guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_signal_toolkit_init_line_count() -> None:
    """signal_toolkit/__init__.py must stay under 300 lines.

    The file uses a _LAZY dispatch table which adds lines but eliminates
    eager submodule imports. This guard catches future accumulation.
    """
    lines = SIGNAL_TOOLKIT_INIT.read_text().splitlines()
    assert len(lines) <= 300, (
        f"signal_toolkit/__init__.py has {len(lines)} lines — "
        "exceeded 300-line guard. Are eager imports being re-added?"
    )
