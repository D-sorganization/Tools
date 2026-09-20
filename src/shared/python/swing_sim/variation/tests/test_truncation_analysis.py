"""Tests for truncation mean-shift detection (#4253 item d)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.variation import (
    CATEGORY_LAUNCH,
    NoiseSpec,
    VariationDataset,
    VariationPlan,
)
from shared.python.swing_sim.variation.truncation_analysis import (
    detect_truncation_mean_shifts,
)

pytestmark = pytest.mark.physics


def test_unbounded_spec_has_no_truncation_shift() -> None:
    plan = VariationPlan(
        mode="launch",
        noise=(NoiseSpec(f"{CATEGORY_LAUNCH}.ball_speed_mph", scale=1.0),),
        n_runs=100,
        seed=42,
    )
    # 100 draws with mean 150.0 and scale 1.0 (unbounded)
    inputs = np.random.default_rng(42).normal(150.0, 1.0, size=(100, 1))
    dataset = VariationDataset(
        plan=plan,
        input_names=(f"{CATEGORY_LAUNCH}.ball_speed_mph",),
        inputs=inputs,
        output_names=("carry_m",),
        outputs=np.zeros((100, 1)),
        success=np.ones(100, dtype=bool),
    )
    notes = detect_truncation_mean_shifts(dataset)
    assert len(notes) == 0


def test_asymmetric_truncation_detects_mean_shift_and_clamp_counts() -> None:
    key = f"{CATEGORY_LAUNCH}.ball_speed_mph"
    # Base is 150.0, lower bound is 150.0 (one-sided truncation of standard normal)
    spec = NoiseSpec(key, scale=2.0, lower=150.0, upper=160.0)
    plan = VariationPlan(
        mode="launch",
        noise=(spec,),
        n_runs=200,
        seed=42,
    )
    raw = np.random.default_rng(42).normal(150.0, 2.0, size=200)
    clipped = np.clip(raw, 150.0, 160.0).reshape(-1, 1)
    dataset = VariationDataset(
        plan=plan,
        input_names=(key,),
        inputs=clipped,
        output_names=("carry_m",),
        outputs=np.zeros((200, 1)),
        success=np.ones(200, dtype=bool),
    )
    notes = detect_truncation_mean_shifts(dataset)
    assert len(notes) == 1
    note = notes[0]
    assert note.variable_key == key
    assert note.has_shift is True
    assert note.mean_shift > 0.5  # Mean shifted up because values < 150 were clipped
    assert note.truncated_lower_count > 50  # ~half the draws clipped at 150
    assert "shifted" in note.note
    assert "truncation" in note.note
