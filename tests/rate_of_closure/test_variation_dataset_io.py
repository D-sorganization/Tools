"""Unit tests for shared variation dataset IO (JSON, CSV, HDF5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.python.swing_sim.variation import (
    DATASET_HDF5_SCHEMA_ID,
    DATASET_HDF5_SCHEMA_VERSION,
    DATASET_JSON_SCHEMA_VERSION,
    NoiseSpec,
    VariationDataset,
    VariationPlan,
    from_json_dict,
    read_csv,
    read_hdf5,
    read_json,
    sample_inputs,
    to_json_dict,
    write_csv,
    write_hdf5,
    write_json,
)


@pytest.fixture
def sample_dataset() -> VariationDataset:
    input_name = "swing_sim.flight.launch.ball_speed_mph"
    plan = VariationPlan(
        mode="launch",
        noise=(NoiseSpec(input_name, scale=1.0),),
        n_runs=4,
        seed=42,
    )
    samples = sample_inputs(plan)
    outputs = np.column_stack((samples[:, 0] * 1.5, -samples[:, 0]))
    outputs[1] = np.nan
    success = np.array([True, False, True, True])

    return VariationDataset(
        plan=plan,
        input_names=(input_name,),
        inputs=samples,
        output_names=("carry_m", "lateral_m"),
        outputs=outputs,
        success=success,
        elapsed_s=0.5,
    )


def test_schema_constants() -> None:
    assert DATASET_JSON_SCHEMA_VERSION == 2
    assert DATASET_HDF5_SCHEMA_ID == "rate-of-closure/variation-dataset-hdf5"
    assert DATASET_HDF5_SCHEMA_VERSION == 1


def test_json_roundtrip(tmp_path: Path, sample_dataset: VariationDataset) -> None:
    json_path = tmp_path / "dataset.json"
    write_json(sample_dataset, json_path)
    loaded = read_json(json_path)

    np.testing.assert_array_equal(loaded.inputs, sample_dataset.inputs)
    np.testing.assert_allclose(loaded.outputs, sample_dataset.outputs, equal_nan=True)
    np.testing.assert_array_equal(loaded.success, sample_dataset.success)
    assert loaded.input_names == sample_dataset.input_names
    assert loaded.output_names == sample_dataset.output_names
    assert loaded.plan == sample_dataset.plan


def test_json_dict_roundtrip(sample_dataset: VariationDataset) -> None:
    d = to_json_dict(sample_dataset)
    assert d["schema_version"] == 2
    loaded = from_json_dict(d)

    np.testing.assert_array_equal(loaded.inputs, sample_dataset.inputs)
    np.testing.assert_allclose(loaded.outputs, sample_dataset.outputs, equal_nan=True)
    np.testing.assert_array_equal(loaded.success, sample_dataset.success)


def test_csv_roundtrip(tmp_path: Path, sample_dataset: VariationDataset) -> None:
    csv_path = tmp_path / "dataset.csv"
    write_csv(sample_dataset, csv_path)
    loaded = read_csv(csv_path, sample_dataset.plan)

    np.testing.assert_array_equal(loaded.inputs, sample_dataset.inputs)
    np.testing.assert_allclose(loaded.outputs, sample_dataset.outputs, equal_nan=True)
    np.testing.assert_array_equal(loaded.success, sample_dataset.success)


def test_hdf5_roundtrip(tmp_path: Path, sample_dataset: VariationDataset) -> None:
    pytest.importorskip("h5py")
    h5_path = tmp_path / "dataset.h5"
    write_hdf5(sample_dataset, h5_path)
    loaded = read_hdf5(h5_path)

    np.testing.assert_array_equal(loaded.inputs, sample_dataset.inputs)
    np.testing.assert_allclose(loaded.outputs, sample_dataset.outputs, equal_nan=True)
    np.testing.assert_array_equal(loaded.success, sample_dataset.success)
    assert loaded.input_names == sample_dataset.input_names
    assert loaded.output_names == sample_dataset.output_names
    assert loaded.plan == sample_dataset.plan
    assert loaded.elapsed_s == pytest.approx(sample_dataset.elapsed_s)
