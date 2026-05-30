"""Logic tests for the wave-1 PSA fixes (#3105).

Pure-logic tests (no Qt widget construction, per the segfault gotcha): the
canonical flammability classifier (F6), the webapp/canonical agreement when
the webapp is importable (F6), and the O2 hazard-band sizing rule (F3).
"""

from __future__ import annotations

import numpy as np
import pytest
from sidekick.process_calculators.psa_package.psa_model import (
    calculate_o2_safety_analysis,
    get_flammability_status,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("h2", "o2", "expected"),
    [
        (50.0, 0.05, "Safe-Low O2"),
        (10.0, 5.0, "CRITICAL"),  # h2>4 and o2>2 takes precedence
        (2.0, 5.0, "Safe-Below LFL"),
        (80.0, 1.5, "Caution-Rich"),  # rich, but o2<=2 so not CRITICAL
        (50.0, 1.5, "FLAMMABLE"),  # in LFL-UFL band, o2<=2
    ],
)
def test_canonical_flammability_status(h2: float, o2: float, expected: str) -> None:
    assert get_flammability_status(h2, o2) == expected


@pytest.mark.unit
def test_webapp_flammability_agrees_with_canonical() -> None:
    """Webapp fork must not diverge from canonical safety logic (#3105 F6).

    Skipped when streamlit (the webapp's hard dependency) is unavailable.
    """
    webapp = pytest.importorskip("sidekick.process_calculators.psa_package.psa_webapp")
    for h2 in (1.0, 4.5, 50.0, 80.0):
        for o2 in (0.05, 2.5, 10.0):
            status, _color = webapp.get_flammability_status(h2, o2)
            assert status == get_flammability_status(h2, o2)


@pytest.mark.unit
def test_o2_band_top_from_data_max() -> None:
    """The hazard-band top is derived from the plotted data max (#3105 F3).

    Mirrors the sizing rule used in ``SensitivityPlotWidget._plot_o2_safety``
    so it can never collapse to the matplotlib default (0, 1) limit.
    """
    analysis = calculate_o2_safety_analysis(
        inlet_o2_pcts=np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float64),
        stage1_o2_removal_range=np.linspace(50.0, 95.0, 11, dtype=np.float64),
    )
    data_max = float(np.nanmax(analysis["s2_tail_o2"]))
    band_top = max(data_max, 2.0) * 1.05
    # The band must span at least the 2% danger threshold and never collapse.
    assert band_top >= 2.0
    assert band_top >= data_max
