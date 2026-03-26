"""URDF Viewer web application — FastAPI backend.

Provides:
  - Static file serving for the Three.js viewer
  - Model upload / list / retrieve API
  - URDF generation API (reuses urdf_builder_gui core modules)
"""

import logging
import os
import shutil
import sys
from pathlib import Path

from cors import add_cors_middleware
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Ensure urdf_builder_gui is importable ───────────────────────────────
_URDF_BUILDER_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "urdf_builder_gui" / "python"
)
if _URDF_BUILDER_DIR not in sys.path:
    sys.path.insert(0, _URDF_BUILDER_DIR)

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


@app.post("/api/upload")
async def upload_file(file: UploadFile) -> dict[str, str]:
    """Upload a URDF file."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is missing")

        file_path = get_safe_path(file.filename)
        logger.info("Uploading file to %s", file_path)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": file_path.name, "url": f"/api/models/{file_path.name}"}
    except HTTPException:
        raise
    except (PermissionError, OSError) as e:
        logger.error("Failed to upload file: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/models")
async def list_models() -> dict[str, list[str]]:
    """List available URDF models."""
    try:
        files = [
            f
            for f in os.listdir(MODELS_DIR)
            if (f.endswith(".urdf") or f.endswith(".xml")) and os.path.isfile(MODELS_DIR / f)
        ]
        return {"models": files}
    except (PermissionError, OSError) as e:
        logger.error("Failed to list models: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/models/{filename}")
async def get_model(filename: str) -> FileResponse:
    """Retrieve a specific URDF model file."""
    file_path = get_safe_path(filename)
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
    except Exception as e:
        logger.error("URDF generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


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

    except Exception as e:
        logger.error("Preview generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/templates")
async def list_templates() -> dict[str, list[str]]:
    """List available URDF templates."""
    try:
        from urdf_builder_gui.anthropometric_model import TEMPLATE_SEGMENTS

        return {"templates": list(TEMPLATE_SEGMENTS.keys())}
    except Exception as e:
        logger.error("Failed to list templates: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104  # nosec B104
