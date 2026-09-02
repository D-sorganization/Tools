"""Shared shot-frame builder for the canonical launch-monitor suite.

UpstreamDrift's ``tests/unit/launch_monitor/test_analysis.py`` is a single
199-line file covering seven modules across nine test functions, one of which
(``test_dispersion_and_longitudinal_trend_capture_change``) covers two modules
at once. The ADR-0046 G1 port plan requires the file to be **split** so that
tests travel with the module they exercise, so the private ``_shots`` helper
every one of those cases builds on lands here once rather than being copied
into each split file.

The construction is UpstreamDrift's, unchanged — same seed, same columns, same
coefficients — so the split files assert against exactly the frames the
original suite asserted against.

``fixtures_dir`` serves the same purpose for step P9: UpstreamDrift's
``tests/unit/launch_monitor/test_importer.py`` splits across
:mod:`shared.python.launch_monitor.profiles` and
:mod:`shared.python.launch_monitor.importer`, and both halves read the six
synthetic vendor exports that travelled with them from
``tests/fixtures/launch_monitor/``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def build_shots(n: int = 80) -> pd.DataFrame:
    """Return the deterministic synthetic shot frame used across the suite."""
    rng = np.random.default_rng(42)
    club = np.linspace(35.0, 50.0, n)
    attack = rng.normal(-0.04, 0.025, n)
    ball = 1.47 * club + 3.0 * attack + rng.normal(0.0, 0.7, n)
    return pd.DataFrame(
        {
            "shot_id": [f"s{i}" for i in range(n)],
            "session_id": np.where(np.arange(n) < n / 2, "a", "b"),
            "monitor_vendor": np.where(np.arange(n) % 2, "Garmin", "TrackMan"),
            "captured_at": pd.date_range("2026-01-01", periods=n, freq="D"),
            "club_speed": club,
            "attack_angle": attack,
            "ball_speed": ball,
            "smash_factor": ball / club,
            "carry_distance": 3.4 * ball + rng.normal(0.0, 2.0, n),
            "lateral_carry": rng.normal(2.0, 8.0, n),
        }
    )


@pytest.fixture
def shots() -> Callable[..., pd.DataFrame]:
    """Expose :func:`build_shots` as a fixture for the split test files."""
    return build_shots


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the vendor-export fixtures that travelled with step P9."""
    return FIXTURE_DIR
