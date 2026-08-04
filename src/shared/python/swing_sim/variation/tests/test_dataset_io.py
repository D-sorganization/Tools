"""CSV/JSON dataset round-trips (#4120 V3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.python.swing_sim.variation import (
    CATEGORY_LAUNCH,
    NoiseSpec,
    VariationPlan,
    run_variation,
)
from shared.python.swing_sim.variation.dataset_io import (
    read_csv,
    read_json,
    write_csv,
    write_json,
)

pytestmark = pytest.mark.physics

_BALL = f"{CATEGORY_LAUNCH}.ball_speed_mph"


@pytest.fixture(scope="module")
def dataset():  # type: ignore[no-untyped-def]
    plan = VariationPlan(
        mode="launch",
        base_variables={_BALL: 0.8},  # some runs sample below zero and fail
        noise=(NoiseSpec(_BALL, scale=2.0),),
        n_runs=12,
        seed=4,
    )
    return run_variation(plan, n_workers=2)


class TestJsonRoundTrip:
    def test_lossless_including_plan_and_failures(  # type: ignore[no-untyped-def]
        self, dataset, tmp_path: Path
    ) -> None:
        path = tmp_path / "study.json"
        write_json(dataset, path)
        loaded = read_json(path)
        assert loaded.plan == dataset.plan
        assert loaded.input_names == dataset.input_names
        assert loaded.output_names == dataset.output_names
        np.testing.assert_array_equal(loaded.inputs, dataset.inputs)
        np.testing.assert_array_equal(loaded.outputs, dataset.outputs)
        np.testing.assert_array_equal(loaded.success, dataset.success)
        assert loaded.elapsed_s == dataset.elapsed_s


class TestCsvRoundTrip:
    def test_lossless_given_the_plan(self, dataset, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "study.csv"
        write_csv(dataset, path)
        loaded = read_csv(path, dataset.plan)
        np.testing.assert_array_equal(loaded.inputs, dataset.inputs)
        np.testing.assert_array_equal(loaded.outputs, dataset.outputs)
        np.testing.assert_array_equal(loaded.success, dataset.success)

    def test_rejects_a_mismatched_plan(self, dataset, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "study.csv"
        write_csv(dataset, path)
        other = VariationPlan(
            mode="launch",
            noise=(NoiseSpec(f"{CATEGORY_LAUNCH}.spin_rpm", scale=100.0),),
            n_runs=12,
            seed=4,
        )
        with pytest.raises(Exception, match="match plan"):
            read_csv(path, other)
