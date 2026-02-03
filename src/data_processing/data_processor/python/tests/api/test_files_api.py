"""Tests for the files API endpoints (TDD - RED phase)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for the API."""
    from data_processor.api.app import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({"time": [1, 2, 3], "signal_a": [1.0, 2.0, 3.0], "signal_b": [4.0, 5.0, 6.0]})
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health endpoint returns OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestLoadFileEndpoint:
    """Tests for file loading endpoint."""

    def test_load_valid_csv_returns_file_info(
        self, client: TestClient, sample_csv: Path
    ) -> None:
        """Loading a valid CSV returns file information."""
        response = client.post("/api/v1/files/load", json={"path": str(sample_csv)})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "file_id" in data
        assert data["file_info"]["row_count"] == 3
        assert data["file_info"]["column_count"] == 3

    def test_load_nonexistent_file_returns_error(self, client: TestClient) -> None:
        """Loading a nonexistent file returns an error."""
        response = client.post(
            "/api/v1/files/load", json={"path": "/nonexistent/file.csv"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] is not None

    def test_load_file_validates_extension(self, client: TestClient, tmp_path: Path) -> None:
        """Loading rejects files with invalid extensions."""
        bad_file = tmp_path / "test.exe"
        bad_file.write_text("malicious content")
        response = client.post("/api/v1/files/load", json={"path": str(bad_file)})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestListFilesEndpoint:
    """Tests for listing loaded files."""

    def test_list_files_empty_initially(self, client: TestClient) -> None:
        """File list is empty when no files loaded."""
        response = client.get("/api/v1/files")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["files"] == []

    def test_list_files_after_loading(
        self, client: TestClient, sample_csv: Path
    ) -> None:
        """File list contains loaded files."""
        # Load a file first
        client.post("/api/v1/files/load", json={"path": str(sample_csv)})

        response = client.get("/api/v1/files")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert len(data["files"]) == 1


class TestSignalsEndpoint:
    """Tests for getting signals from a file."""

    def test_get_signals_returns_signal_list(
        self, client: TestClient, sample_csv: Path
    ) -> None:
        """Getting signals returns list of signals in file."""
        # Load file first
        load_response = client.post(
            "/api/v1/files/load", json={"path": str(sample_csv)}
        )
        file_id = load_response.json()["file_id"]

        response = client.get(f"/api/v1/files/{file_id}/signals")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 3
        assert data["numeric_count"] == 3
        signal_names = [s["name"] for s in data["signals"]]
        assert "signal_a" in signal_names
        assert "signal_b" in signal_names

    def test_get_signals_invalid_file_returns_404(self, client: TestClient) -> None:
        """Getting signals for invalid file ID returns 404."""
        response = client.get("/api/v1/files/invalid-id/signals")
        assert response.status_code == 404


class TestDeleteFileEndpoint:
    """Tests for deleting loaded files."""

    def test_delete_file_removes_from_list(
        self, client: TestClient, sample_csv: Path
    ) -> None:
        """Deleting a file removes it from the list."""
        # Load a file
        load_response = client.post(
            "/api/v1/files/load", json={"path": str(sample_csv)}
        )
        file_id = load_response.json()["file_id"]

        # Delete it
        delete_response = client.delete(f"/api/v1/files/{file_id}")
        assert delete_response.status_code == 200

        # Verify it's gone
        list_response = client.get("/api/v1/files")
        assert list_response.json()["total_count"] == 0

    def test_delete_nonexistent_file_returns_404(self, client: TestClient) -> None:
        """Deleting a nonexistent file returns 404."""
        response = client.delete("/api/v1/files/nonexistent-id")
        assert response.status_code == 404
