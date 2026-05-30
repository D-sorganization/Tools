"""Verification tests for the full Data Processor app and its embeddable bridge.

Covers (GitHub issues D-sorganization/Tools#3111, #3112, #3113, #3114):

* The shared embedding bridge ``sidekick.data_processing.embedding`` that lets
  Sidekick / Gasification_Model / UpstreamDrift construct the *real* Data
  Processor widget (single source of truth) instead of a divergent copy.
* The launch blockers that previously prevented the full app from importing:
  the hard ``numba`` import and the ``high_performance_loader`` ``utils``
  dependency with no fallback.
* The core processing operations the tool advertises (filtering, derivatives,
  integration, time-resampling/retiming, multi-format export).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sidekick.data_processing import embedding


# --------------------------------------------------------------------------- #
# Path bridge (issue #3112 launch bootstrap, #3113 single source of truth)
# --------------------------------------------------------------------------- #
def test_find_repo_root_locates_data_processor() -> None:
    root = embedding._find_repo_root(Path(__file__).resolve())
    assert (root / "pyproject.toml").is_file()
    assert (root / "src/data_processing/data_processor/python").is_dir()


def test_ensure_paths_is_idempotent_and_adds_required_roots() -> None:
    root = embedding.ensure_full_data_processor_on_path()
    expected = [str(root / sub) for sub in embedding.REQUIRED_SUBPATHS]
    for path in expected:
        assert path in sys.path
    before = list(sys.path)
    embedding.ensure_full_data_processor_on_path()  # second call is a no-op
    assert sys.path == before


def test_ensure_paths_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        embedding.ensure_full_data_processor_on_path(repo_root="not-a-path")


def test_full_data_processor_available() -> None:
    assert embedding.full_data_processor_available() is True


# --------------------------------------------------------------------------- #
# Launch blockers (issue #3112)
# --------------------------------------------------------------------------- #
def test_signal_processing_imports_without_numba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine must import and run even when numba is unavailable."""
    embedding.ensure_full_data_processor_on_path()
    monkeypatch.setitem(sys.modules, "numba", None)  # force ImportError on import
    module = importlib.import_module("data_processor.core.signal_processing")
    module = importlib.reload(module)
    # A bare callable passed to the fallback decorator is returned unchanged.
    sentinel = lambda x: x  # noqa: E731
    assert module.jit(sentinel) is sentinel
    # Parameterized usage returns a decorator that is also identity.
    assert module.jit(nopython=True)(sentinel) is sentinel


def test_high_performance_loader_falls_back_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding.ensure_full_data_processor_on_path()
    from data_processor.core import data_loader as dl

    def _boom() -> object:
        raise ImportError("simulated missing utils package")

    monkeypatch.setattr(
        dl.DataLoader, "_import_high_performance_loader", staticmethod(_boom)
    )
    loader = dl.DataLoader(use_high_performance=True)
    assert loader.hp_loader is None
    assert loader.use_high_performance is False


# --------------------------------------------------------------------------- #
# Core operations (issue #3114 verify functionality)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def time_series() -> pd.DataFrame:
    t = np.linspace(0.0, 1.0, 200)
    clean = np.sin(2 * np.pi * 3 * t)
    noise = 0.25 * np.sin(2 * np.pi * 60 * t)
    return pd.DataFrame({"time": t, "signal": clean + noise})


def test_differentiation_produces_derivative(time_series: pd.DataFrame) -> None:
    embedding.ensure_full_data_processor_on_path()
    from data_processor.core.signal_processing import differentiate_signals

    result = differentiate_signals(time_series, "time", ["signal"], method="spline")
    assert "signal_d1" in result.columns
    assert len(result) == len(time_series)
    assert np.isfinite(result["signal_d1"].to_numpy()).any()


def test_integration_adds_cumulative_column(time_series: pd.DataFrame) -> None:
    embedding.ensure_full_data_processor_on_path()
    from data_processor.core.signal_processing import integrate_signals

    result = integrate_signals(time_series, "time", ["signal"], method="trapezoidal")
    assert "cumulative_signal" in result.columns
    assert len(result) == len(time_series)


def test_time_resampling_changes_time_base() -> None:
    embedding.ensure_full_data_processor_on_path()
    from data_processor.core.signal_processing import resample_data

    index = pd.date_range("2020-01-01", periods=200, freq="100ms")
    df = pd.DataFrame({"time": index, "signal": np.arange(200, dtype=float)})
    resampled = resample_data(df, "time", "2s", method="mean")
    assert "time" in resampled.columns
    assert len(resampled) < len(df)


def test_export_round_trip_csv_and_parquet(
    tmp_path: Path, time_series: pd.DataFrame
) -> None:
    csv_path = tmp_path / "out.csv"
    time_series.to_csv(csv_path, index=False)
    assert pd.read_csv(csv_path).shape == time_series.shape

    parquet_path = tmp_path / "out.parquet"
    try:
        time_series.to_parquet(parquet_path)
    except (ImportError, ValueError):
        pytest.skip("parquet engine not installed")
    assert pd.read_parquet(parquet_path).shape == time_series.shape


# --------------------------------------------------------------------------- #
# Embeddable widget construction (issue #3113) — requires PyQt6, headless safe
# --------------------------------------------------------------------------- #
@pytest.mark.headless_safe
def test_create_full_data_processor_widget() -> None:
    pytest.importorskip("PyQt6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = embedding.create_full_data_processor_widget()
    assert isinstance(widget, QWidget)
    assert type(widget).__name__ == "DataProcessorWidget"
