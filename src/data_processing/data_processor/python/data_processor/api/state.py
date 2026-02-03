"""Application state management for the API."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class LoadedFile:
    """Represents a loaded file with its data."""

    file_id: str
    filename: str
    path: str
    dataframe: pd.DataFrame
    loaded_at: datetime
    size_bytes: int

    @property
    def row_count(self) -> int:
        """Get number of rows."""
        return len(self.dataframe)

    @property
    def column_count(self) -> int:
        """Get number of columns."""
        return len(self.dataframe.columns)


@dataclass
class AppState:
    """Application state for storing loaded files and data."""

    files: dict[str, LoadedFile] = field(default_factory=dict)

    def add_file(
        self, path: str, dataframe: pd.DataFrame, size_bytes: int
    ) -> LoadedFile:
        """Add a new file to the state."""
        file_id = str(uuid.uuid4())[:8]
        filename = Path(path).name
        loaded_file = LoadedFile(
            file_id=file_id,
            filename=filename,
            path=path,
            dataframe=dataframe,
            loaded_at=datetime.now(),
            size_bytes=size_bytes,
        )
        self.files[file_id] = loaded_file
        return loaded_file

    def get_file(self, file_id: str) -> LoadedFile | None:
        """Get a loaded file by ID."""
        return self.files.get(file_id)

    def remove_file(self, file_id: str) -> bool:
        """Remove a file from state."""
        if file_id in self.files:
            del self.files[file_id]
            return True
        return False

    def list_files(self) -> list[LoadedFile]:
        """List all loaded files."""
        return list(self.files.values())

    def clear(self) -> None:
        """Clear all loaded files."""
        self.files.clear()
