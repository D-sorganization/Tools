"""Dataset state management with history tracking.

Provides functionality to save, load, and manage filtered datasets
without overwriting source data. Supports maintaining multiple
dataset versions and history for undo/redo operations.

Follows Clean Code principles:
- Single Responsibility: Each class has one purpose
- Small functions with clear names
- Immutable data where possible
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetMetadata:
    """Immutable metadata for a dataset version."""

    id: str
    name: str
    created_at: str
    source_file: str | None
    parent_id: str | None
    description: str
    operation: str
    parameters: dict[str, Any]

    @classmethod
    def create(
        cls,
        name: str,
        source_file: str | None = None,
        parent_id: str | None = None,
        description: str = "",
        operation: str = "created",
        parameters: dict[str, Any] | None = None,
    ) -> DatasetMetadata:
        """Factory method to create new metadata."""
        return cls(
            id=str(uuid4()),
            name=name,
            created_at=datetime.now().isoformat(),
            source_file=source_file,
            parent_id=parent_id,
            description=description,
            operation=operation,
            parameters=parameters or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "source_file": self.source_file,
            "parent_id": self.parent_id,
            "description": self.description,
            "operation": self.operation,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMetadata:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=data["created_at"],
            source_file=data.get("source_file"),
            parent_id=data.get("parent_id"),
            description=data.get("description", ""),
            operation=data.get("operation", "unknown"),
            parameters=data.get("parameters", {}),
        )


@dataclass
class DatasetVersion:
    """A versioned dataset with its data and metadata."""

    metadata: DatasetMetadata
    data: pd.DataFrame

    def copy(self) -> DatasetVersion:
        """Create a deep copy of this version."""
        return DatasetVersion(
            metadata=self.metadata,
            data=self.data.copy(),
        )


@dataclass
class DatasetHistory:
    """Tracks the history of operations on a dataset."""

    versions: list[DatasetVersion] = field(default_factory=list)
    current_index: int = -1

    @property
    def current(self) -> DatasetVersion | None:
        """Get the current dataset version."""
        if 0 <= self.current_index < len(self.versions):
            return self.versions[self.current_index]
        return None

    @property
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self.current_index > 0

    @property
    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self.current_index < len(self.versions) - 1

    def add_version(self, version: DatasetVersion) -> None:
        """Add a new version, truncating any redo history."""
        # Remove any versions after current (truncate redo history)
        self.versions = self.versions[: self.current_index + 1]
        self.versions.append(version)
        self.current_index = len(self.versions) - 1

    def undo(self) -> DatasetVersion | None:
        """Move to previous version."""
        if self.can_undo:
            self.current_index -= 1
            return self.current
        return None

    def redo(self) -> DatasetVersion | None:
        """Move to next version."""
        if self.can_redo:
            self.current_index += 1
            return self.current
        return None

    def clear(self) -> None:
        """Clear all history."""
        self.versions.clear()
        self.current_index = -1


class DatasetManager:
    """Manages datasets with history tracking and persistence.

    Provides:
    - Loading data without modifying source files
    - Saving filtered/processed datasets as new versions
    - Undo/redo support via version history
    - Export to various formats
    """

    def __init__(self, workspace_dir: Path | str | None = None) -> None:
        """Initialize the dataset manager.

        Args:
            workspace_dir: Directory for storing dataset versions.
                          If None, uses in-memory storage only.
        """
        self._workspace_dir = Path(workspace_dir) if workspace_dir else None
        self._datasets: dict[str, DatasetHistory] = {}
        self._active_dataset_id: str | None = None

        if self._workspace_dir:
            self._workspace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_dataset(self) -> DatasetVersion | None:
        """Get the currently active dataset."""
        if self._active_dataset_id and self._active_dataset_id in self._datasets:
            return self._datasets[self._active_dataset_id].current
        return None

    @property
    def active_data(self) -> pd.DataFrame | None:
        """Get the data from the currently active dataset."""
        dataset = self.active_dataset
        return dataset.data if dataset else None

    @property
    def dataset_ids(self) -> list[str]:
        """Get all dataset IDs."""
        return list(self._datasets.keys())

    @property
    def can_undo(self) -> bool:
        """Check if undo is available for active dataset."""
        if self._active_dataset_id:
            history = self._datasets.get(self._active_dataset_id)
            return history.can_undo if history else False
        return False

    @property
    def can_redo(self) -> bool:
        """Check if redo is available for active dataset."""
        if self._active_dataset_id:
            history = self._datasets.get(self._active_dataset_id)
            return history.can_redo if history else False
        return False

    def load_from_file(
        self,
        file_path: Path | str,
        name: str | None = None,
    ) -> str:
        """Load a dataset from file without modifying the source.

        Args:
            file_path: Path to the source file
            name: Optional name for the dataset

        Returns:
            Dataset ID for the loaded dataset
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = self._read_file(file_path)
        dataset_name = name or file_path.stem

        metadata = DatasetMetadata.create(
            name=dataset_name,
            source_file=str(file_path),
            operation="loaded",
            description=f"Loaded from {file_path.name}",
        )

        version = DatasetVersion(metadata=metadata, data=data)
        history = DatasetHistory()
        history.add_version(version)

        dataset_id = metadata.id
        self._datasets[dataset_id] = history
        self._active_dataset_id = dataset_id

        logger.info(f"Loaded dataset '{dataset_name}' with ID {dataset_id}")
        return dataset_id

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        name: str = "Untitled",
        source_file: str | None = None,
    ) -> str:
        """Load a dataset from an existing DataFrame.

        Args:
            df: DataFrame to load
            name: Name for the dataset
            source_file: Optional source file path

        Returns:
            Dataset ID for the loaded dataset
        """
        metadata = DatasetMetadata.create(
            name=name,
            source_file=source_file,
            operation="imported",
            description="Imported from DataFrame",
        )

        version = DatasetVersion(metadata=metadata, data=df.copy())
        history = DatasetHistory()
        history.add_version(version)

        dataset_id = metadata.id
        self._datasets[dataset_id] = history
        self._active_dataset_id = dataset_id

        logger.info(f"Imported dataset '{name}' with ID {dataset_id}")
        return dataset_id

    def save_version(
        self,
        data: pd.DataFrame,
        operation: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        dataset_id: str | None = None,
    ) -> str:
        """Save a new version of the dataset.

        This creates a new version in history without modifying
        the source file or previous versions.

        Args:
            data: The processed DataFrame
            operation: Name of the operation performed
            description: Human-readable description
            parameters: Parameters used in the operation
            dataset_id: ID of dataset to update (uses active if None)

        Returns:
            Version ID of the saved version
        """
        target_id = dataset_id or self._active_dataset_id
        if not target_id or target_id not in self._datasets:
            raise ValueError("No active dataset to save version for")

        history = self._datasets[target_id]
        current = history.current
        if not current:
            raise ValueError("No current version in history")

        metadata = DatasetMetadata.create(
            name=current.metadata.name,
            source_file=current.metadata.source_file,
            parent_id=current.metadata.id,
            operation=operation,
            description=description,
            parameters=parameters or {},
        )

        version = DatasetVersion(metadata=metadata, data=data.copy())
        history.add_version(version)

        logger.info(f"Saved new version: {operation}")
        return metadata.id

    def undo(self, dataset_id: str | None = None) -> pd.DataFrame | None:
        """Undo the last operation.

        Args:
            dataset_id: ID of dataset (uses active if None)

        Returns:
            DataFrame after undo, or None if not possible
        """
        target_id = dataset_id or self._active_dataset_id
        if not target_id or target_id not in self._datasets:
            return None

        history = self._datasets[target_id]
        version = history.undo()
        if version:
            logger.info(f"Undo: reverted to version {version.metadata.id}")
            return version.data
        return None

    def redo(self, dataset_id: str | None = None) -> pd.DataFrame | None:
        """Redo the previously undone operation.

        Args:
            dataset_id: ID of dataset (uses active if None)

        Returns:
            DataFrame after redo, or None if not possible
        """
        target_id = dataset_id or self._active_dataset_id
        if not target_id or target_id not in self._datasets:
            return None

        history = self._datasets[target_id]
        version = history.redo()
        if version:
            logger.info(f"Redo: restored version {version.metadata.id}")
            return version.data
        return None

    def set_active_dataset(self, dataset_id: str) -> None:
        """Set the active dataset by ID."""
        if dataset_id not in self._datasets:
            raise ValueError(f"Unknown dataset ID: {dataset_id}")
        self._active_dataset_id = dataset_id

    def get_dataset(self, dataset_id: str) -> DatasetVersion | None:
        """Get a specific dataset by ID."""
        history = self._datasets.get(dataset_id)
        return history.current if history else None

    def get_history(self, dataset_id: str | None = None) -> list[DatasetMetadata]:
        """Get the operation history for a dataset.

        Args:
            dataset_id: ID of dataset (uses active if None)

        Returns:
            List of metadata for all versions
        """
        target_id = dataset_id or self._active_dataset_id
        if not target_id or target_id not in self._datasets:
            return []

        history = self._datasets[target_id]
        return [v.metadata for v in history.versions]

    def export_dataset(
        self,
        output_path: Path | str,
        dataset_id: str | None = None,
        format: str | None = None,
    ) -> Path:
        """Export a dataset to file.

        Args:
            output_path: Path for the output file
            dataset_id: ID of dataset (uses active if None)
            format: Output format (inferred from extension if None)

        Returns:
            Path to the exported file
        """
        target_id = dataset_id or self._active_dataset_id
        if not target_id or target_id not in self._datasets:
            raise ValueError("No dataset to export")

        version = self._datasets[target_id].current
        if not version:
            raise ValueError("No current version to export")

        output_path = Path(output_path)
        format = format or output_path.suffix.lstrip(".")

        self._write_file(version.data, output_path, format)
        logger.info(f"Exported dataset to {output_path}")
        return output_path

    def save_workspace(self, workspace_path: Path | str | None = None) -> Path:
        """Save the entire workspace state to disk.

        Args:
            workspace_path: Directory to save to (uses default if None)

        Returns:
            Path to the saved workspace
        """
        save_dir = Path(workspace_path) if workspace_path else self._workspace_dir
        if not save_dir:
            raise ValueError("No workspace directory specified")

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save each dataset
        for dataset_id, history in self._datasets.items():
            dataset_dir = save_dir / dataset_id
            dataset_dir.mkdir(exist_ok=True)

            # Save metadata index
            metadata_list = [v.metadata.to_dict() for v in history.versions]
            index_path = dataset_dir / "index.json"
            with open(index_path, "w") as f:
                json.dump(
                    {
                        "versions": metadata_list,
                        "current_index": history.current_index,
                    },
                    f,
                    indent=2,
                )

            # Save data for each version
            for version in history.versions:
                data_path = dataset_dir / f"{version.metadata.id}.parquet"
                version.data.to_parquet(data_path)

        # Save workspace index
        workspace_index = {
            "active_dataset_id": self._active_dataset_id,
            "datasets": list(self._datasets.keys()),
        }
        with open(save_dir / "workspace.json", "w") as f:
            json.dump(workspace_index, f, indent=2)

        logger.info(f"Saved workspace to {save_dir}")
        return save_dir

    def load_workspace(self, workspace_path: Path | str) -> None:
        """Load workspace state from disk.

        Args:
            workspace_path: Directory containing saved workspace
        """
        load_dir = Path(workspace_path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Workspace not found: {load_dir}")

        workspace_index_path = load_dir / "workspace.json"
        if not workspace_index_path.exists():
            raise ValueError(f"Invalid workspace: missing workspace.json")

        with open(workspace_index_path) as f:
            workspace_index = json.load(f)

        self._datasets.clear()

        # Load each dataset
        for dataset_id in workspace_index["datasets"]:
            dataset_dir = load_dir / dataset_id
            index_path = dataset_dir / "index.json"

            with open(index_path) as f:
                index_data = json.load(f)

            history = DatasetHistory()
            history.current_index = index_data["current_index"]

            for meta_dict in index_data["versions"]:
                metadata = DatasetMetadata.from_dict(meta_dict)
                data_path = dataset_dir / f"{metadata.id}.parquet"
                data = pd.read_parquet(data_path)
                history.versions.append(DatasetVersion(metadata=metadata, data=data))

            self._datasets[dataset_id] = history

        self._active_dataset_id = workspace_index.get("active_dataset_id")
        logger.info(f"Loaded workspace from {load_dir}")

    def close_dataset(self, dataset_id: str | None = None) -> None:
        """Close a dataset and remove it from memory.

        Args:
            dataset_id: ID of dataset (uses active if None)
        """
        target_id = dataset_id or self._active_dataset_id
        if target_id and target_id in self._datasets:
            del self._datasets[target_id]
            if self._active_dataset_id == target_id:
                self._active_dataset_id = None
            logger.info(f"Closed dataset {target_id}")

    def _read_file(self, path: Path) -> pd.DataFrame:
        """Read data from various file formats."""
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif suffix == ".parquet":
            return pd.read_parquet(path)
        elif suffix == ".json":
            return pd.read_json(path)
        elif suffix in (".h5", ".hdf5"):
            return pd.read_hdf(path)
        elif suffix == ".feather":
            return pd.read_feather(path)
        else:
            # Try CSV as default
            return pd.read_csv(path)

    def _write_file(self, df: pd.DataFrame, path: Path, format: str) -> None:
        """Write data to various file formats."""
        format = format.lower()

        if format == "csv":
            df.to_csv(path, index=False)
        elif format in ("xlsx", "excel"):
            df.to_excel(path, index=False)
        elif format == "parquet":
            df.to_parquet(path)
        elif format == "json":
            df.to_json(path, orient="records", indent=2)
        elif format in ("h5", "hdf5"):
            df.to_hdf(path, key="data", mode="w")
        elif format == "feather":
            df.to_feather(path)
        else:
            df.to_csv(path, index=False)


__all__ = [
    "DatasetManager",
    "DatasetVersion",
    "DatasetHistory",
    "DatasetMetadata",
]
