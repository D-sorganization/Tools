"""Tests for the experimental plant_simulator SCADADataset (issue #3295).

Verifies the silent random-data path is gone: the loader now reads a real
SQLite ``taglog`` table and raises ``ValueError`` on insufficient data unless
synthetic data is explicitly requested.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

from plant_simulator.dataset import SCADADataset  # noqa: E402


def _make_taglog_db(path: Path, n_timesteps: int, num_tags: int) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE taglog (timestamp INTEGER, tag_id TEXT, value REAL)")
        rows = []
        for t in range(n_timesteps):
            for tag in range(num_tags):
                rows.append((t, f"TAG_{tag}", float(t * 0.1 + tag)))
        conn.executemany("INSERT INTO taglog VALUES (?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_missing_db_raises_not_random(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(ValueError, match="not found"):
        SCADADataset(str(missing), sequence_length=5, num_tags=4)


def test_insufficient_data_raises(tmp_path: Path) -> None:
    db = tmp_path / "small.db"
    _make_taglog_db(db, n_timesteps=3, num_tags=4)
    with pytest.raises(ValueError, match="insufficient data"):
        SCADADataset(str(db), sequence_length=5, num_tags=4)


def test_loads_real_taglog(tmp_path: Path) -> None:
    db = tmp_path / "ok.db"
    _make_taglog_db(db, n_timesteps=50, num_tags=4)
    ds = SCADADataset(str(db), sequence_length=10, num_tags=4)
    assert ds.data.shape == (50, 4)
    # The pivot must reflect the seeded values, not random noise.
    assert ds.data[5, 2] == pytest.approx(5 * 0.1 + 2)
    assert len(ds) == 50 - 10


def test_window_shapes(tmp_path: Path) -> None:
    db = tmp_path / "ok2.db"
    _make_taglog_db(db, n_timesteps=30, num_tags=4)
    ds = SCADADataset(str(db), sequence_length=8, num_tags=4)
    x, y = ds[0]
    assert tuple(x.shape) == (8, 4)
    assert tuple(y.shape) == (4,)


def test_synthetic_requires_explicit_optin(tmp_path: Path) -> None:
    missing = tmp_path / "none.db"
    # Default: no silent fabrication.
    with pytest.raises(ValueError):
        SCADADataset(str(missing), sequence_length=5, num_tags=4)
    # Explicit opt-in produces data without raising.
    ds = SCADADataset(str(missing), sequence_length=5, num_tags=4, allow_synthetic=True)
    assert ds.data.shape[1] == 4
    assert ds.data.shape[0] >= 6


def test_invalid_params_rejected(tmp_path: Path) -> None:
    db = tmp_path / "ok3.db"
    _make_taglog_db(db, n_timesteps=20, num_tags=4)
    with pytest.raises(ValueError, match="sequence_length"):
        SCADADataset(str(db), sequence_length=0, num_tags=4)
    with pytest.raises(ValueError, match="num_tags"):
        SCADADataset(str(db), sequence_length=5, num_tags=0)
