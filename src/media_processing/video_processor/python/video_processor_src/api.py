"""FastAPI backend for Video Processor.

Provides endpoints for:
- File upload (video files)
- Processing status via Server-Sent Events (SSE)
- Job management (list, cancel)

See issue #626.

Usage:
    uvicorn video_processor_src.api:app --reload --port 8001
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from cors import add_cors_middleware
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path("output/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE_MB = 500
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Processing job status."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingJob(BaseModel):
    """A video processing job."""

    job_id: str = Field(description="Unique job identifier")
    filename: str = Field(description="Original filename")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: float = Field(description="Unix timestamp")
    message: str = Field(default="")
    output_path: str | None = Field(default=None)


class JobListResponse(BaseModel):
    """Response for listing jobs."""

    jobs: list[ProcessingJob]


class UploadResponse(BaseModel):
    """Response after successful upload."""

    job_id: str
    filename: str
    message: str


# ---------------------------------------------------------------------------
# In-memory job store (production would use Redis/DB)
# ---------------------------------------------------------------------------

_jobs: dict[str, ProcessingJob] = {}


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Video Processor API",
    description="Upload and process video files with progress tracking.",
    version="0.1.0",
)
add_cors_middleware(app)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "video-processor"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile) -> UploadResponse:
    """Upload a video file for processing.

    The file is saved to the upload directory and a processing job is created.
    Use the returned job_id to track progress via SSE.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed}",
        )

    # Generate unique job ID and save file
    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}{ext}"
    file_path = UPLOAD_DIR / safe_filename

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        msg = f"File too large ({size_mb:.1f} MB). Max: {MAX_UPLOAD_SIZE_MB} MB"
        raise HTTPException(status_code=413, detail=msg)

    file_path.write_bytes(content)

    # Create processing job
    job = ProcessingJob(
        job_id=job_id,
        filename=file.filename,
        status=JobStatus.QUEUED,
        created_at=time.time(),
        message="Upload complete, queued for processing",
    )
    _jobs[job_id] = job

    logger.info(
        "Video uploaded: %s -> %s (%.1f MB)",
        file.filename,
        safe_filename,
        size_mb,
    )

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        message="Upload successful. Connect to /api/progress/{job_id} for updates.",
    )


@app.get("/api/jobs", response_model=JobListResponse)
async def list_jobs() -> JobListResponse:
    """List all processing jobs."""
    return JobListResponse(jobs=list(_jobs.values()))


@app.get("/api/jobs/{job_id}", response_model=ProcessingJob)
async def get_job(job_id: str) -> ProcessingJob:
    """Get status of a specific job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _jobs[job_id]


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a processing job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]
    current_status = job.status
    if current_status in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {current_status.value}",
        )

    job.status = JobStatus.CANCELLED
    job.message = "Cancelled by user"
    return {"success": True, "job_id": job_id}


@app.get("/api/progress/{job_id}")
async def progress_stream(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream for job progress.

    Connect to this endpoint to receive real-time progress updates.
    The stream closes when the job completes, fails, or is cancelled.

    Event format::

        data: {"job_id": "...", "progress": 42.0}
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    async def event_generator() -> Any:
        """Generate SSE events for job progress."""
        job = _jobs[job_id]

        # Simulate processing if still queued
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.PROCESSING
            job.message = "Processing started"

        while job.status == JobStatus.PROCESSING:
            # Simulate progress (real implementation would poll actual processor)
            job.progress = min(job.progress + 2.0, 100.0)
            if job.progress >= 100.0:
                job.status = JobStatus.COMPLETED
                job.message = "Processing complete"

            event_data = job.model_dump_json()
            yield f"data: {event_data}\n\n"

            if job.status != JobStatus.PROCESSING:
                break

            await asyncio.sleep(0.5)

        # Send final status
        yield f"data: {job.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/process/{job_id}")
async def start_processing(job_id: str) -> dict[str, Any]:
    """Trigger processing for an uploaded video.

    In production this would dispatch to a task queue (Celery, etc.).
    For now it marks the job as processing so SSE can simulate progress.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]
    current_status = job.status
    if current_status != JobStatus.QUEUED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not queued (current status: {current_status.value})",
        )

    job.status = JobStatus.PROCESSING
    new_status = job.status
    job.message = "Processing started"
    return {"success": True, "job_id": job_id, "status": new_status.value}
