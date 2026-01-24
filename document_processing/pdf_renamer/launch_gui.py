"""Launcher script for PDF Renamer GUI."""

import logging

# Use shared logging utility
try:
    from utils.logging_utils import init_default_logging
except ImportError:
    # Fallback
    def init_default_logging():
        logging.basicConfig(level=logging.INFO)
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
init_default_logging()s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_renamer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

if __name__ == "__main__":
    from pdf_renamer.gui import main

    main()
