#!/usr/bin/env python3
"""Unified launcher for the Data Processor.

Usage:
    python -m data_processor.launch gui      # Launch PyQt6 GUI
    python -m data_processor.launch tk       # Launch Tkinter GUI (fallback)
    python -m data_processor.launch api      # Launch API server
    python -m data_processor.launch --help   # Show help
"""

from __future__ import annotations

import argparse
import sys


def launch_pyqt_gui() -> None:
    """Launch the PyQt6 GUI."""
    try:
        from data_processor.gui.main_window import main

        main()
    except ImportError as e:
        print(f"Error: PyQt6 GUI failed to load: {e}")
        print("Falling back to Tkinter GUI...")
        launch_tkinter_gui()


def launch_tkinter_gui() -> None:
    """Launch the Tkinter GUI (fallback)."""
    try:
        from data_processor.gui_refactored import main

        main()
    except ImportError as e:
        print(f"Error: Tkinter GUI failed to load: {e}")
        sys.exit(1)


def launch_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch the API server."""
    try:
        import uvicorn

        from data_processor.api.app import create_app

        app = create_app()
        print(f"Starting Data Processor API at http://{host}:{port}")
        print(f"API docs available at http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port)
    except ImportError as e:
        print(f"Error: API server failed to start: {e}")
        print("Install with: pip install uvicorn fastapi")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Data Processor Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mode",
        choices=["gui", "tk", "api"],
        nargs="?",
        default="gui",
        help="Launch mode: gui (PyQt6), tk (Tkinter), api (server)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")

    args = parser.parse_args()

    if args.mode == "gui":
        launch_pyqt_gui()
    elif args.mode == "tk":
        launch_tkinter_gui()
    elif args.mode == "api":
        launch_api_server(args.host, args.port)


if __name__ == "__main__":
    main()
