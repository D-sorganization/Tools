#!/usr/bin/env python3
"""Launch script for the Data Processor API server."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description="Data Processor API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install uvicorn")
        sys.exit(1)

    from data_processor.api.app import create_app

    app = create_app()

    print(f"Starting Data Processor API at http://{args.host}:{args.port}")
    print(f"API docs available at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
