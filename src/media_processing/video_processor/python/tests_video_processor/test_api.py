"""Tests for api.py — covers LoD-fixed endpoints in GH1474.

Tests validate that cancel_job and start_processing endpoints correctly
return job status values in error details and response bodies after
the LoD refactor (local variable extraction for job.status.value).
"""

import sys
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_cors_mock() -> ModuleType:
    """Create a mock cors module that satisfies api.py imports."""
    mock_cors = ModuleType("cors")
    mock_cors.add_cors_middleware = MagicMock()  # type: ignore[attr-defined]
    return mock_cors


@pytest.fixture(scope="module")
def api_module():  # type: ignore
    """Import api module with cors mocked out."""
    mock_cors = _make_cors_mock()
    with patch.dict(sys.modules, {"cors": mock_cors}):
        import importlib

        import video_processor_src.api as _api

        importlib.reload(_api)
        yield _api


@pytest.fixture()
def client(api_module):  # type: ignore
    """Return a TestClient for the video processor API."""
    return TestClient(api_module.app)


class TestCancelJob:
    """Tests for cancel_job endpoint — LoD fix at line 191."""

    def test_cancel_completed_job_returns_status_value_in_detail(  # type: ignore
        self, client, api_module
    ) -> None:
        """cancel_job on COMPLETED job returns status value in error detail."""
        JobStatus = api_module.JobStatus
        ProcessingJob = api_module.ProcessingJob
        jobs = api_module._jobs

        job_id = "cancel-completed-1474"
        job = ProcessingJob(
            job_id=job_id,
            filename="video.mp4",
            status=JobStatus.COMPLETED,
            created_at=time.time(),
        )
        jobs[job_id] = job
        try:
            response = client.post(f"/api/jobs/{job_id}/cancel")
            assert response.status_code == 400
            detail = response.json()["detail"]
            assert JobStatus.COMPLETED.value in detail
        finally:
            jobs.pop(job_id, None)

    def test_cancel_failed_job_returns_status_value_in_detail(  # type: ignore
        self, client, api_module
    ) -> None:
        """cancel_job on FAILED job returns status value in error detail."""
        JobStatus = api_module.JobStatus
        ProcessingJob = api_module.ProcessingJob
        jobs = api_module._jobs

        job_id = "cancel-failed-1474"
        job = ProcessingJob(
            job_id=job_id,
            filename="video.mp4",
            status=JobStatus.FAILED,
            created_at=time.time(),
        )
        jobs[job_id] = job
        try:
            response = client.post(f"/api/jobs/{job_id}/cancel")
            assert response.status_code == 400
            detail = response.json()["detail"]
            assert JobStatus.FAILED.value in detail
        finally:
            jobs.pop(job_id, None)

    def test_cancel_queued_job_succeeds(self, client, api_module) -> None:  # type: ignore
        """cancel_job on a QUEUED job succeeds."""
        JobStatus = api_module.JobStatus
        ProcessingJob = api_module.ProcessingJob
        jobs = api_module._jobs

        job_id = "cancel-queued-1474"
        job = ProcessingJob(
            job_id=job_id,
            filename="video.mp4",
            status=JobStatus.QUEUED,
            created_at=time.time(),
        )
        jobs[job_id] = job
        try:
            response = client.post(f"/api/jobs/{job_id}/cancel")
            assert response.status_code == 200
            assert response.json()["success"] is True
        finally:
            jobs.pop(job_id, None)


class TestStartProcessing:
    """Tests for start_processing endpoint — LoD fix at lines 265, 270."""

    def test_start_processing_non_queued_returns_status_value(  # type: ignore
        self, client, api_module
    ) -> None:
        """start_processing on non-QUEUED job returns status value in detail."""
        JobStatus = api_module.JobStatus
        ProcessingJob = api_module.ProcessingJob
        jobs = api_module._jobs

        job_id = "start-processing-1474"
        job = ProcessingJob(
            job_id=job_id,
            filename="video.mp4",
            status=JobStatus.PROCESSING,
            created_at=time.time(),
        )
        jobs[job_id] = job
        try:
            response = client.post(f"/api/process/{job_id}")
            assert response.status_code == 400
            detail = response.json()["detail"]
            assert JobStatus.PROCESSING.value in detail
        finally:
            jobs.pop(job_id, None)

    def test_start_processing_queued_job_returns_new_status(  # type: ignore
        self, client, api_module
    ) -> None:
        """start_processing on QUEUED job returns PROCESSING status in response."""
        JobStatus = api_module.JobStatus
        ProcessingJob = api_module.ProcessingJob
        jobs = api_module._jobs

        job_id = "start-queued-1474"
        job = ProcessingJob(
            job_id=job_id,
            filename="video.mp4",
            status=JobStatus.QUEUED,
            created_at=time.time(),
        )
        jobs[job_id] = job
        try:
            response = client.post(f"/api/process/{job_id}")
            assert response.status_code == 200
            body = response.json()
            assert body["success"] is True
            assert body["status"] == JobStatus.PROCESSING.value
        finally:
            jobs.pop(job_id, None)

    def test_start_processing_not_found_returns_404(self, client) -> None:  # type: ignore
        """start_processing on unknown job returns 404."""
        response = client.post("/api/process/nonexistent-job-id")
        assert response.status_code == 404
