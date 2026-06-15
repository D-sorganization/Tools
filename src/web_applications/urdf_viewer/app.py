# ruff: noqa: E501
"""URDF Viewer web application — FastAPI backend.

Provides:
  - Static file serving for the Three.js viewer
  - Model upload / list / retrieve API
  - URDF generation API (reuses urdf_builder_gui core modules)
"""

import logging
import os
from pathlib import Path

from cors import add_cors_middleware
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.dependencies.utils import ensure_multipart_is_installed
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Security constants ──────────────────────────────────────────────────
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTENSIONS = {".urdf", ".xml"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="URDF Viewer")
add_cors_middleware(app)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic request model ──────────────────────────────────────────────


class URDFGenerateRequest(BaseModel):
    """Request body for URDF generation API."""

    robot_name: str = Field(default="humanoid", min_length=1)
    height_m: float = Field(default=1.75, gt=0, le=3.0)
    mass_kg: float = Field(default=70.0, gt=0, le=500.0)
    gender_factor: float = Field(default=0.5, ge=0, le=1.0)
    template: str = Field(default="Full Humanoid")
    geometry_type: str = Field(default="box")
    collision_geometry: str = Field(default="Same as Visual")
    inertia_mode: str = Field(default="Primitive")
    damping: float = Field(default=0.5, ge=0)
    friction: float = Field(default=0.0, ge=0)
    density: float = Field(default=1050.0, gt=0)


# ── Static Routes ───────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def read_root() -> FileResponse:
    """Serve the viewer index page."""
    return FileResponse(STATIC_DIR / "index.html")


# ── File Safety ─────────────────────────────────────────────────────────


def get_safe_path(filename: str) -> Path:
    """Sanitize and validate the filename to prevent path traversal."""
    safe_name = os.path.basename(filename)
    separators = {sep for sep in ("/", "\\", os.sep, os.path.altsep) if sep}
    if any(sep in filename for sep in separators):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_name or safe_name in [".", ".."]:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = MODELS_DIR / safe_name

    try:
        resolved_path = file_path.resolve()
        resolved_root = MODELS_DIR.resolve()

        if not resolved_path.is_relative_to(resolved_root):
            raise HTTPException(status_code=403, detail="Access denied")

        return file_path
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path") from None


# ── Model CRUD ──────────────────────────────────────────────────────────


def _python_multipart_available() -> bool:
    """Return whether FastAPI can register multipart upload routes."""
    try:
        ensure_multipart_is_installed()
    except RuntimeError as exc:
        logger.warning("Multipart upload support is unavailable: %s", exc)
        return False
    return True


async def _store_uploaded_file(
    file: UploadFile,
    overwrite: bool = Query(default=False),
) -> dict[str, str]:
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is missing")

        # Validate extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension '{ext}'. Only {ALLOWED_EXTENSIONS} allowed.",  # noqa: E501
            )

        file_path = get_safe_path(file.filename)

        # Prevent silent overwrite
        if file_path.exists() and not overwrite:
            raise HTTPException(
                status_code=409,
                detail="File already exists. Use ?overwrite=1 to replace.",
            )

        logger.info("Uploading file to %s", file_path)

        # Stream with size limit
        size = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = file.file.read(8192)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",  # noqa: E501
                    )
                buffer.write(chunk)

        return {"filename": file_path.name, "url": f"/api/models/{file_path.name}"}
    except HTTPException:
        raise
    except (PermissionError, OSError) as e:
        logger.error("Failed to upload file: %s", e)
        raise HTTPException(status_code=500, detail="Upload failed") from e


if _python_multipart_available():

    @app.post("/api/upload")
    async def upload_file(
        file: UploadFile,
        overwrite: bool = Query(default=False),
    ) -> dict[str, str]:
        """Upload a URDF or XML model file.

        Args:
            file: The uploaded file (max 25 MB, must be .urdf or .xml).
            overwrite: If True, allow overwriting an existing file.
        """
        return await _store_uploaded_file(file=file, overwrite=overwrite)

else:

    @app.post("/api/upload")
    async def upload_file_unavailable(
        overwrite: bool = Query(default=False),
    ) -> None:
        """Report missing multipart support without failing application import."""
        raise HTTPException(
            status_code=503,
            detail="File uploads require the python-multipart package.",
        )


@app.get("/api/models")
async def list_models() -> dict[str, list[str]]:
    """List available URDF models."""
    try:
        files = [
            f
            for f in os.listdir(MODELS_DIR)
            if (f.endswith(".urdf") or f.endswith(".xml"))
            and os.path.isfile(MODELS_DIR / f)
        ]
        return {"models": files}
    except (PermissionError, OSError) as e:
        logger.error("Failed to list models: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list models") from e


@app.get("/api/models/{filename}")
async def get_model(filename: str) -> FileResponse:
    """Retrieve a specific URDF model file."""
    file_path = get_safe_path(filename)

    # Only serve allowed extensions
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# ── URDF Generation API (reuses urdf_builder_gui core) ──────────────────


@app.post("/api/generate")
async def generate_urdf(request: URDFGenerateRequest) -> Response:
    """Generate a URDF XML file from parameters.

    Reuses the same core modules as the PyQt6 desktop GUI.
    """
    try:
        from urdf_builder_gui.anthropometric_model import URDFConfig
        from urdf_builder_gui.urdf_generator import (
            generate_urdf_xml,
            validate_urdf_structure,
        )

        config = URDFConfig(
            robot_name=request.robot_name,
            height_m=request.height_m,
            mass_kg=request.mass_kg,
            gender_factor=request.gender_factor,
            template=request.template,
            geometry_type=request.geometry_type,
            collision_geometry=request.collision_geometry,
            inertia_mode=request.inertia_mode,
            damping=request.damping,
            friction=request.friction,
            density=request.density,
        )

        urdf_xml = generate_urdf_xml(config)
        is_valid, errors = validate_urdf_structure(urdf_xml)

        if not is_valid:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Generated URDF failed validation",
                    "errors": errors,
                },
            )

        logger.info("Generated URDF for '%s'", request.robot_name)
        return Response(content=urdf_xml, media_type="application/xml")

    except HTTPException:
        raise
    except (ImportError, ValueError, TypeError) as e:
        logger.error("URDF generation failed: %s", e)
        raise HTTPException(status_code=500, detail="URDF generation failed") from e


@app.post("/api/preview")
async def preview_model(request: URDFGenerateRequest) -> dict[str, str]:
    """Generate a human-readable preview of the model structure."""
    try:
        from urdf_builder_gui.anthropometric_model import URDFConfig
        from urdf_builder_gui.preview_generator import generate_preview_text

        config = URDFConfig(
            robot_name=request.robot_name,
            height_m=request.height_m,
            mass_kg=request.mass_kg,
            gender_factor=request.gender_factor,
            template=request.template,
            geometry_type=request.geometry_type,
            collision_geometry=request.collision_geometry,
            inertia_mode=request.inertia_mode,
            damping=request.damping,
            friction=request.friction,
            density=request.density,
        )

        preview = generate_preview_text(config)
        return {"preview": preview}

    except (ImportError, ValueError, TypeError) as e:
        logger.error("Preview generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Preview generation failed") from e


@app.get("/api/templates")
async def list_templates() -> dict[str, list[str]]:
    """List available URDF templates."""
    try:
        from urdf_builder_gui.anthropometric_model import TEMPLATE_SEGMENTS

        return {"templates": list(TEMPLATE_SEGMENTS.keys())}
    except ImportError as e:
        logger.error("Failed to list templates: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load templates") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
