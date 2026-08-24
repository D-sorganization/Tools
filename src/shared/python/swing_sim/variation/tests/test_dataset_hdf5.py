"""Versioned, self-contained HDF5 variation-dataset persistence (#4142)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from shared.python.contracts import (
    ContractLevel,
    ContractViolationError,
    get_contract_level,
    set_contract_level,
)
from shared.python.swing_sim.variation import (
    CATEGORY_LAUNCH,
    DATASET_HDF5_SCHEMA_ID,
    DATASET_HDF5_SCHEMA_VERSION,
    Hdf5UnavailableError,
    NoiseSpec,
    VariationPlan,
    dataset_hdf5,
    read_hdf5,
    run_variation,
    write_hdf5,
)

pytestmark = pytest.mark.physics


@pytest.fixture(scope="module")
def dataset():  # type: ignore[no-untyped-def]
    ball_speed = f"{CATEGORY_LAUNCH}.ball_speed_mph"
    plan = VariationPlan(
        mode="launch",
        base_variables={ball_speed: 0.8},
        noise=(NoiseSpec(ball_speed, scale=2.0),),
        n_runs=12,
        seed=4,
    )
    return run_variation(plan, n_workers=1)


def test_round_trip_is_lossless_and_self_contained(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"

    write_hdf5(dataset, path)
    loaded = read_hdf5(path)

    assert loaded.plan == dataset.plan
    assert loaded.input_names == dataset.input_names
    assert loaded.output_names == dataset.output_names
    np.testing.assert_array_equal(loaded.inputs, dataset.inputs)
    np.testing.assert_array_equal(loaded.outputs, dataset.outputs)
    np.testing.assert_array_equal(loaded.success, dataset.success)
    assert loaded.elapsed_s == dataset.elapsed_s


def test_archive_declares_public_schema_and_execution_document(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"
    write_hdf5(dataset, path)

    with h5py.File(path, "r") as archive:
        assert archive.attrs["schema_id"] == DATASET_HDF5_SCHEMA_ID
        assert archive.attrs["schema_version"] == DATASET_HDF5_SCHEMA_VERSION
        assert archive.attrs["content_sha256"]
        assert "plan_document_json" in archive
        assert "inputs" in archive
        assert "outputs" in archive
        assert "success" in archive


def test_write_refuses_to_replace_existing_evidence(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"
    path.write_bytes(b"reviewed evidence")

    with pytest.raises(FileExistsError, match="already exists"):
        write_hdf5(dataset, path)

    assert path.read_bytes() == b"reviewed evidence"


def test_reader_rejects_unsupported_schema_version(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"
    write_hdf5(dataset, path)
    with h5py.File(path, "r+") as archive:
        archive.attrs["schema_version"] = DATASET_HDF5_SCHEMA_VERSION + 1

    with pytest.raises(ContractViolationError, match="schema_version"):
        read_hdf5(path)


def test_reader_rejects_missing_required_member(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"
    write_hdf5(dataset, path)
    with h5py.File(path, "r+") as archive:
        del archive["success"]

    with pytest.raises(ContractViolationError, match="members mismatch"):
        read_hdf5(path)


def test_reader_rejects_numeric_content_substitution(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"
    write_hdf5(dataset, path)
    with h5py.File(path, "r+") as archive:
        archive["outputs"][0, 0] += 1.0

    with pytest.raises(ContractViolationError, match="content digest mismatch"):
        read_hdf5(path)


def test_integrity_check_cannot_be_disabled_with_dbc_level(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"
    write_hdf5(dataset, path)
    with h5py.File(path, "r+") as archive:
        archive["outputs"][0, 0] += 1.0

    previous = get_contract_level()
    try:
        set_contract_level(ContractLevel.OFF)
        with pytest.raises(ContractViolationError, match="content digest mismatch"):
            read_hdf5(path)
    finally:
        set_contract_level(previous)


def test_reader_rejects_invalid_array_shape_before_construction(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"
    write_hdf5(dataset, path)
    with h5py.File(path, "r+") as archive:
        del archive["outputs"]
        archive.create_dataset("outputs", data=np.zeros(dataset.plan.n_runs))

    with pytest.raises(ContractViolationError, match="outputs must be two-dimensional"):
        read_hdf5(path)


def test_reader_rejects_nonbinary_success_values(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"
    write_hdf5(dataset, path)
    with h5py.File(path, "r+") as archive:
        archive["success"][0] = 2

    with pytest.raises(ContractViolationError, match="zero or one"):
        read_hdf5(path)


def test_reader_rejects_execution_document_substitution(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "study.h5"
    write_hdf5(dataset, path)
    with h5py.File(path, "r+") as archive:
        archive["plan_document_json"][()] = "{}"

    with pytest.raises(ContractViolationError, match="content digest mismatch"):
        read_hdf5(path)


def test_missing_hdf5_dependency_has_actionable_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def unavailable(name: str):  # type: ignore[no-untyped-def]
        error = ModuleNotFoundError("No module named 'h5py'")
        error.name = "h5py"
        raise error

    monkeypatch.setattr(dataset_hdf5.importlib, "import_module", unavailable)

    with pytest.raises(Hdf5UnavailableError, match="variation-hdf5"):
        dataset_hdf5._require_h5py()
