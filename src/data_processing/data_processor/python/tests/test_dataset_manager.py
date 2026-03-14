"""Tests for DatasetManager — versioned dataset state management.

Covers load, save_version, undo/redo, close, and workspace persistence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from data_processor.core.dataset_manager import (
    DatasetHistory,
    DatasetManager,
    DatasetMetadata,
    DatasetVersion,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Simple 3-row dataset."""
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})


@pytest.fixture()
def csv_file(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    """Write sample_df to a CSV and return the path."""
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def mgr(tmp_path: Path) -> DatasetManager:
    """Manager with a temp workspace dir."""
    return DatasetManager(workspace_dir=tmp_path / "workspace")


# ── DatasetMetadata ──────────────────────────────────────────────────────────


class TestDatasetMetadata:
    """Tests for immutable metadata creation and serialization."""

    def test_create_generates_unique_id(self) -> None:
        """Each metadata instance should have a unique UUID."""
        m1 = DatasetMetadata.create(name="a")
        m2 = DatasetMetadata.create(name="b")
        assert m1.id != m2.id

    def test_round_trip_dict(self) -> None:
        """to_dict → from_dict should be lossless."""
        original = DatasetMetadata.create(
            name="test", operation="filtered", parameters={"k": 5}
        )
        restored = DatasetMetadata.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.operation == original.operation
        assert restored.parameters == original.parameters


# ── DatasetHistory ───────────────────────────────────────────────────────────


class TestDatasetHistory:
    """Tests for version history undo/redo."""

    def _make_version(self, name: str) -> DatasetVersion:
        meta = DatasetMetadata.create(name=name)
        return DatasetVersion(metadata=meta, data=pd.DataFrame({"v": [1]}))

    def test_empty_history_has_no_current(self) -> None:
        h = DatasetHistory()
        assert h.current is None
        assert not h.can_undo
        assert not h.can_redo

    def test_add_version_sets_current(self) -> None:
        h = DatasetHistory()
        v = self._make_version("v1")
        h.add_version(v)
        assert h.current is v

    def test_undo_redo_cycle(self) -> None:
        h = DatasetHistory()
        v1 = self._make_version("v1")
        v2 = self._make_version("v2")
        h.add_version(v1)
        h.add_version(v2)
        assert h.current is v2
        assert h.can_undo

        undone = h.undo()
        assert undone is v1
        assert h.can_redo

        redone = h.redo()
        assert redone is v2

    def test_add_after_undo_truncates_redo(self) -> None:
        h = DatasetHistory()
        v1 = self._make_version("v1")
        v2 = self._make_version("v2")
        v3 = self._make_version("v3")
        h.add_version(v1)
        h.add_version(v2)
        h.undo()
        h.add_version(v3)
        assert h.current is v3
        assert not h.can_redo

    def test_clear(self) -> None:
        h = DatasetHistory()
        h.add_version(self._make_version("v1"))
        h.clear()
        assert h.current is None


# ── DatasetManager ───────────────────────────────────────────────────────────


class TestDatasetManagerLoad:
    """Tests for loading datasets."""

    def test_load_from_file(self, mgr: DatasetManager, csv_file: Path) -> None:
        """Loading from CSV should create an active dataset."""
        ds_id = mgr.load_from_file(csv_file)
        assert ds_id in mgr.dataset_ids
        assert mgr.active_data is not None
        assert len(mgr.active_data) == 3

    def test_load_from_dataframe(
        self, mgr: DatasetManager, sample_df: pd.DataFrame
    ) -> None:
        ds_id = mgr.load_from_dataframe(sample_df, name="inline")
        assert ds_id in mgr.dataset_ids
        pd.testing.assert_frame_equal(mgr.active_data, sample_df)

    def test_load_nonexistent_file_raises(self, mgr: DatasetManager) -> None:
        with pytest.raises(FileNotFoundError):
            mgr.load_from_file("/nonexistent/file.csv")


class TestDatasetManagerVersioning:
    """Tests for save_version and undo/redo."""

    def test_save_version_and_undo(
        self, mgr: DatasetManager, sample_df: pd.DataFrame
    ) -> None:
        mgr.load_from_dataframe(sample_df, name="orig")
        modified = sample_df.copy()
        modified["z"] = [100, 200, 300]
        mgr.save_version(modified, operation="add_column", description="Added z")

        assert "z" in mgr.active_data.columns
        assert mgr.can_undo

        mgr.undo()
        assert "z" not in mgr.active_data.columns

    def test_redo_restores(self, mgr: DatasetManager, sample_df: pd.DataFrame) -> None:
        mgr.load_from_dataframe(sample_df, name="test")
        modified = sample_df.copy()
        modified["z"] = 999
        mgr.save_version(modified, operation="test")
        mgr.undo()
        mgr.redo()
        assert "z" in mgr.active_data.columns


class TestDatasetManagerClose:
    """Tests for closing datasets."""

    def test_close_removes_dataset(
        self, mgr: DatasetManager, sample_df: pd.DataFrame
    ) -> None:
        ds_id = mgr.load_from_dataframe(sample_df, name="to_close")
        mgr.close_dataset(ds_id)
        assert ds_id not in mgr.dataset_ids
        assert mgr.active_data is None


class TestDatasetManagerWorkspace:
    """Tests for workspace save/load round-trip."""

    def test_workspace_round_trip(
        self, mgr: DatasetManager, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Save workspace → load from fresh manager → data matches."""
        mgr.load_from_dataframe(sample_df, name="ws_test")
        ws_path = tmp_path / "ws_save"
        mgr.save_workspace(ws_path)

        mgr2 = DatasetManager()
        mgr2.load_workspace(ws_path)
        assert mgr2.active_data is not None
        np.testing.assert_array_almost_equal(
            mgr2.active_data["x"].values, sample_df["x"].values
        )
