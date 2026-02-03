"""Tests for the processing API endpoints (TDD - RED phase)."""

from __future__ import annotations

from pathlib import Path

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
    df = pd.DataFrame(
        {
            "time": range(100),
            "signal_a": [float(i) + 0.1 * (i % 10) for i in range(100)],
            "signal_b": [float(i) * 2 for i in range(100)],
        }
    )
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def loaded_file(client: TestClient, sample_csv: Path) -> str:
    """Load a file and return its ID."""
    response = client.post("/api/v1/files/load", json={"path": str(sample_csv)})
    return response.json()["file_id"]


class TestApplyFilterEndpoint:
    """Tests for the filter application endpoint."""

    def test_apply_moving_average_filter(
        self, client: TestClient, loaded_file: str
    ) -> None:
        """Applying moving average filter succeeds."""
        response = client.post(
            "/api/v1/processing/filter",
            json={
                "file_id": loaded_file,
                "filter_type": "Moving Average",
                "signals": ["signal_a"],
                "parameters": {"ma_window": 5},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "completed"
        assert "signal_a" in data["signals_processed"]

    def test_apply_butterworth_lowpass_filter(
        self, client: TestClient, loaded_file: str
    ) -> None:
        """Applying Butterworth low-pass filter succeeds."""
        response = client.post(
            "/api/v1/processing/filter",
            json={
                "file_id": loaded_file,
                "filter_type": "Butterworth Low-pass",
                "signals": ["signal_a", "signal_b"],
                "parameters": {"bw_order": 3, "bw_cutoff": 0.1},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_apply_filter_all_signals_when_empty(
        self, client: TestClient, loaded_file: str
    ) -> None:
        """Applying filter with empty signals list filters all numeric signals."""
        response = client.post(
            "/api/v1/processing/filter",
            json={
                "file_id": loaded_file,
                "filter_type": "Moving Average",
                "signals": [],
                "parameters": {"ma_window": 5},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # Should process all numeric signals
        assert len(data["signals_processed"]) >= 2

    def test_apply_filter_invalid_file_returns_404(self, client: TestClient) -> None:
        """Applying filter to invalid file returns 404."""
        response = client.post(
            "/api/v1/processing/filter",
            json={
                "file_id": "invalid-id",
                "filter_type": "Moving Average",
                "signals": [],
                "parameters": {},
            },
        )
        assert response.status_code == 404

    def test_apply_filter_invalid_type_returns_422(
        self, client: TestClient, loaded_file: str
    ) -> None:
        """Applying invalid filter type returns 422."""
        response = client.post(
            "/api/v1/processing/filter",
            json={
                "file_id": loaded_file,
                "filter_type": "Invalid Filter",
                "signals": [],
                "parameters": {},
            },
        )
        assert response.status_code == 422


class TestStatisticsEndpoint:
    """Tests for statistics calculation endpoint."""

    def test_get_statistics_returns_stats(
        self, client: TestClient, loaded_file: str
    ) -> None:
        """Getting statistics returns signal statistics."""
        response = client.post(
            "/api/v1/processing/statistics",
            json={"file_id": loaded_file, "signals": ["signal_a"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["statistics"]) == 1
        stats = data["statistics"][0]
        assert stats["name"] == "signal_a"
        assert stats["count"] == 100
        assert stats["mean"] is not None
        assert stats["std"] is not None
        assert stats["min"] is not None
        assert stats["max"] is not None

    def test_get_statistics_all_signals(
        self, client: TestClient, loaded_file: str
    ) -> None:
        """Getting statistics with empty signals list returns all."""
        response = client.post(
            "/api/v1/processing/statistics",
            json={"file_id": loaded_file, "signals": []},
        )
        assert response.status_code == 200
        data = response.json()
        # Should have stats for all numeric signals
        assert len(data["statistics"]) >= 2


class TestPreviewEndpoint:
    """Tests for data preview endpoint."""

    def test_preview_returns_data(self, client: TestClient, loaded_file: str) -> None:
        """Preview returns data rows."""
        response = client.post(
            "/api/v1/processing/preview",
            json={"file_id": loaded_file, "limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["columns"]) == 3
        assert len(data["data"]) == 10
        assert data["total_rows"] == 100
        assert data["limit"] == 10

    def test_preview_with_offset(self, client: TestClient, loaded_file: str) -> None:
        """Preview with offset skips rows."""
        response = client.post(
            "/api/v1/processing/preview",
            json={"file_id": loaded_file, "offset": 50, "limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["offset"] == 50
        assert len(data["data"]) == 10

    def test_preview_selected_signals(
        self, client: TestClient, loaded_file: str
    ) -> None:
        """Preview with selected signals only returns those columns."""
        response = client.post(
            "/api/v1/processing/preview",
            json={
                "file_id": loaded_file,
                "signals": ["signal_a"],
                "limit": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["columns"] == ["signal_a"]
        assert len(data["data"]) == 5


class TestExportEndpoint:
    """Tests for data export endpoint."""

    def test_export_csv_returns_file(
        self, client: TestClient, loaded_file: str, tmp_path: Path
    ) -> None:
        """Exporting to CSV creates a file."""
        output_path = tmp_path / "output.csv"
        response = client.post(
            "/api/v1/processing/export",
            json={
                "file_id": loaded_file,
                "format": "csv",
                "filename": str(output_path),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "csv"

    def test_export_parquet_returns_file(
        self, client: TestClient, loaded_file: str, tmp_path: Path
    ) -> None:
        """Exporting to Parquet creates a file."""
        output_path = tmp_path / "output.parquet"
        response = client.post(
            "/api/v1/processing/export",
            json={
                "file_id": loaded_file,
                "format": "parquet",
                "filename": str(output_path),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "parquet"
