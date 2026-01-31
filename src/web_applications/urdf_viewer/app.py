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

# Allow CORS for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
async def read_root():
    return FileResponse(STATIC_DIR / "index.html")

def get_safe_path(filename: str) -> Path:
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name in ['.', '..']:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = MODELS_DIR / safe_name
    # Ensure path is within MODELS_DIR
    try:
        # Resolve to absolute paths to check containment
        if not file_path.resolve().is_relative_to(MODELS_DIR.resolve()):
             raise HTTPException(status_code=403, detail="Access denied")
    except ValueError:
         raise HTTPException(status_code=400, detail="Invalid path")

    return file_path

@app.post("/api/upload")
async def upload_file(file: UploadFile):
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
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/models")
async def list_models():
    try:
        files = [f for f in os.listdir(MODELS_DIR) if (f.endswith(".urdf") or f.endswith(".xml")) and os.path.isfile(MODELS_DIR / f)]
        return {"models": files}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/models/{filename}")
async def get_model(filename: str):
    file_path = get_safe_path(filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
