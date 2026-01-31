# URDF Viewer

A web-based viewer for URDF (Unified Robot Description Format) models, built with FastAPI, React, and Three.js.

## Features

- **Upload and View**: Upload URDF files and view them instantly in the browser.
- **Interactive 3D View**: Orbit, zoom, and pan around the model using Three.js.
- **Model Library**: Keeps a list of uploaded models for quick access.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Navigate to the tool directory:
   ```bash
   cd src/web_applications/urdf_viewer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the server:
   ```bash
   python app.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn app:app --reload
   ```

2. Open your browser and navigate to:
   http://localhost:8000

## Architecture

- **Backend**: FastAPI application serving static files and API endpoints for file management.
- **Frontend**: React application using `urdf-loader` and `three.js` loaded via ES modules (no build step required).
- **Storage**: Uploaded models are stored in the `models/` directory.
