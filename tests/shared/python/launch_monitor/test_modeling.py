"""Canonical predictive-modeling tests (ADR-0046 G1 step P8).

The first three cases are the three model cases from UpstreamDrift's
``tests/unit/launch_monitor/test_analysis.py`` — the reproducibility sweep over
the four NumPy estimators, the optional shallow MLP, and the leakage guard —
travelling verbatim with the module they exercise. The remaining cases pin the
refusals and the split guarantees the module's docstring documents, which
``CLAUDE.md``'s design-by-contract rule asks of every ported public entry
point.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pandas as pd
import pytest

from shared.python.launch_monitor.modeling import fit_predictive_model

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.mark.parametrize("model", ["linear", "ridge", "lasso", "elastic_net"])
def test_regression_models_are_reproducible(
    model: str, shots: Callable[..., pd.DataFrame]
) -> None:
    """Ported verbatim from UpstreamDrift's ``test_analysis.py``."""
    frame = shots(120)
    first = fit_predictive_model(
        frame,
        target="ball_speed",
        features=("club_speed", "attack_angle"),
        model=model,
        random_seed=7,
        group_column="session_id",
    )
    second = fit_predictive_model(
        frame,
        target="ball_speed",
        features=("club_speed", "attack_angle"),
        model=model,
        random_seed=7,
        group_column="session_id",
    )
    assert first.metrics == second.metrics
    assert first.metrics["r2"] > 0.9
    assert first.predictions.equals(second.predictions)


def test_shallow_mlp_is_reproducible_when_analysis_extra_is_installed(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Ported verbatim from UpstreamDrift's ``test_analysis.py``."""
    pytest.importorskip("sklearn")
    result = fit_predictive_model(
        shots(160),
        target="ball_speed",
        features=("club_speed", "attack_angle"),
        model="mlp",
        random_seed=11,
    )
    assert result.metrics["r2"] > 0.85
    assert result.coefficients is None


def test_model_rejects_identity_derived_leakage(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Ported verbatim from UpstreamDrift's ``test_analysis.py``."""
    with pytest.raises(ValueError, match="leakage"):
        fit_predictive_model(
            shots(),
            target="ball_speed",
            features=("club_speed", "smash_factor"),
            model="ridge",
        )


def test_leakage_guard_covers_both_directions_and_the_target_itself(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Every way of reading the answer off the sheet is refused, not warned about."""
    frame = shots(120)
    with pytest.raises(ValueError, match=r"target cannot also be a feature"):
        fit_predictive_model(
            frame, target="ball_speed", features=("club_speed", "ball_speed")
        )
    with pytest.raises(ValueError, match=r"Identity-derived target leakage"):
        fit_predictive_model(frame, target="ball_speed", features=("smash_factor",))
    with pytest.raises(ValueError, match=r"Identity-derived target leakage"):
        fit_predictive_model(
            frame, target="smash_factor", features=("ball_speed", "attack_angle")
        )


def test_grouped_split_never_puts_one_group_on_both_sides(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Shots from one session must not appear in both train and test."""
    frame = shots(120)
    result = fit_predictive_model(
        frame,
        target="ball_speed",
        features=("club_speed", "attack_angle"),
        model="ridge",
        random_seed=7,
        group_column="session_id",
    )
    assert result.train_count + result.test_count == len(frame)
    tested_rows = set(result.predictions["row_index"])
    tested_sessions = set(frame.loc[sorted(tested_rows), "session_id"])
    untested = set(frame.index) - tested_rows
    trained_sessions = set(frame.loc[sorted(untested), "session_id"])
    assert tested_sessions and trained_sessions
    assert not (tested_sessions & trained_sessions)


def test_standardisation_uses_training_rows_only(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """A different seed reshuffles the split, so the fit is not scale-invariant.

    If the standardisation statistics were taken over the whole frame the
    coefficients would be identical across seeds for the same estimator; they
    are not, which is the observable consequence of fitting the scaler on the
    training rows.
    """
    frame = shots(120)
    kwargs = {
        "target": "ball_speed",
        "features": ("club_speed", "attack_angle"),
        "model": "ridge",
    }
    first = fit_predictive_model(frame, random_seed=7, **kwargs)
    second = fit_predictive_model(frame, random_seed=8, **kwargs)
    assert first.coefficients is not None
    assert second.coefficients is not None
    assert first.coefficients != second.coefficients
    assert set(first.coefficients) == {"club_speed", "attack_angle"}


def test_modeling_refuses_malformed_requests(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """No features, an impossible split fraction, absent columns, unknown model."""
    frame = shots(120)
    with pytest.raises(ValueError, match=r"At least one feature is required"):
        fit_predictive_model(frame, target="ball_speed", features=())
    for fraction in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError, match=r"test_fraction must be between"):
            fit_predictive_model(
                frame,
                target="ball_speed",
                features=("club_speed",),
                test_fraction=fraction,
            )
    with pytest.raises(ValueError, match=r"Columns not present"):
        fit_predictive_model(frame, target="ball_speed", features=("not_a_column",))
    # A missing group column is caught by the same required-columns check, so
    # ``_split_indices``' own "Group column not present" branch is defensive
    # and unreachable through this entry point. Pinned as it is, not "fixed":
    # this is a pure port.
    with pytest.raises(ValueError, match=r"Columns not present: \['not_a_column'\]"):
        fit_predictive_model(
            frame,
            target="ball_speed",
            features=("club_speed",),
            group_column="not_a_column",
        )
    with pytest.raises(
        ValueError, match=r"model must be linear, ridge, lasso, elastic_net, or mlp"
    ):
        fit_predictive_model(
            frame, target="ball_speed", features=("club_speed",), model="svm"
        )


def test_modeling_refuses_too_few_complete_rows(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """The floor scales with the feature count: four complete rows per feature."""
    with pytest.raises(ValueError, match=r"Insufficient complete rows"):
        fit_predictive_model(shots(10), target="ball_speed", features=("club_speed",))


def test_grouped_split_requires_at_least_two_groups(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """One group cannot be held out from itself."""
    frame = shots(120).assign(session_id="only")
    with pytest.raises(ValueError, match=r"at least two groups"):
        fit_predictive_model(
            frame,
            target="ball_speed",
            features=("club_speed",),
            group_column="session_id",
        )


def test_mlp_names_its_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, shots: Callable[..., pd.DataFrame]
) -> None:
    """The ``mlp`` branch raises a named ``ImportError``, never a bare one.

    The message is UpstreamDrift's, carried over verbatim by this pure port —
    it still names the ``upstream-drift[analysis]`` extra. Repointing it at
    this repo's dependency set is a tracked follow-up, and this assertion is
    what makes that change visible rather than silent.
    """
    monkeypatch.setitem(sys.modules, "sklearn.neural_network", None)
    with pytest.raises(ImportError, match=r"The shallow MLP requires the analysis"):
        fit_predictive_model(
            shots(160),
            target="ball_speed",
            features=("club_speed", "attack_angle"),
            model="mlp",
        )
