import logging
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="URDF Viewer")

# Restrict CORS to known local development origins.
# Override with CORS_ORIGINS env var (comma-separated) if needed.
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
_env_origins = os.environ.get("CORS_ORIGINS")
_cors_origins = _env_origins.split(",") if _env_origins else _DEFAULT_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def get_safe_path(filename: str) -> Path:
    """
    Sanitize and validate the filename to prevent path traversal.
    """
    # 1. Sanitize the filename to remove directory components
    safe_name = os.path.basename(filename)

    # 2. Basic validation
    if not safe_name or safe_name in [".", ".."]:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # 3. Construct the full path
    file_path = MODELS_DIR / safe_name

    # 4. Canonicalize paths to check for traversal
    try:
        resolved_path = file_path.resolve()
        resolved_root = MODELS_DIR.resolve()

        # 5. Verify containment
        if not resolved_path.is_relative_to(resolved_root):
            raise HTTPException(status_code=403, detail="Access denied")

        return file_path
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path") from None


@app.post("/api/upload")
async def upload_file(file: UploadFile) -> dict[str, str]:
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is missing")

        file_path = get_safe_path(file.filename)
        logger.info(f"Uploading file to {file_path}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": file_path.name, "url": f"/api/models/{file_path.name}"}
    except HTTPException:
        raise
    except (PermissionError, OSError) as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/models")
async def list_models() -> dict[str, list[str]]:
    try:
        files = [
            f
            for f in os.listdir(MODELS_DIR)
            if (f.endswith(".urdf") or f.endswith(".xml"))
            and os.path.isfile(MODELS_DIR / f)
        ]
        return {"models": files}
    except (PermissionError, OSError) as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/models/{filename}")
async def get_model(filename: str) -> FileResponse:
    file_path = get_safe_path(filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
