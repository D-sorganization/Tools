"""Canonical multivariate tests (ADR-0046 G1 step P2).

The first case is UpstreamDrift's
``tests/unit/launch_monitor/test_analysis.py::test_pca_and_vif_expose_multicollinearity``,
which travels with the module per the port plan. The remaining cases pin the
input validation the module's public functions perform, which this repository's
design-by-contract rule requires of every ported public entry point.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from shared.python.launch_monitor.multivariate import compute_pca, compute_vif

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

METRICS = ("club_speed", "ball_speed", "carry_distance", "attack_angle")


def test_pca_and_vif_expose_multicollinearity(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Ported verbatim from the UpstreamDrift analysis suite."""
    frame = shots(100)
    pca = compute_pca(frame, metrics=METRICS)
    vif = compute_vif(frame, metrics=METRICS)
    assert pca.sample_count == 100
    assert pca.explained_variance_ratio.sum() == pytest.approx(1.0)
    assert pca.loadings.shape == (4, 4)
    assert vif.sample_count == 100
    assert vif.values["ball_speed"] > 5
    assert "ball_speed" in vif.warning_metrics


def test_multivariate_requires_at_least_two_metrics(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """One metric has no covariance structure to decompose."""
    frame = shots(20)
    with pytest.raises(ValueError, match=r"At least two metrics are required"):
        compute_pca(frame, metrics=("club_speed",))
    with pytest.raises(ValueError, match=r"At least two metrics are required"):
        compute_vif(frame, metrics=("club_speed",))


def test_multivariate_names_the_metrics_it_cannot_find(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """An absent metric is refused by name, never silently dropped."""
    frame = shots(20)
    with pytest.raises(ValueError, match=r"Metrics not present"):
        compute_pca(frame, metrics=("club_speed", "spin_rate"))


def test_multivariate_refuses_constant_metrics(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Standardizing a constant column would divide by zero."""
    frame = shots(20).assign(attack_angle=0.0)
    with pytest.raises(ValueError, match=r"Constant metrics cannot be analyzed"):
        compute_vif(frame, metrics=("club_speed", "ball_speed", "attack_angle"))


def test_multivariate_requires_enough_complete_rows(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Fewer complete rows than ``max(5, len(metrics) + 1)`` is refused."""
    frame = shots(20).head(4)
    with pytest.raises(ValueError, match=r"Insufficient complete rows"):
        compute_pca(frame, metrics=("club_speed", "ball_speed"))
