"""Launcher script for PDF Renamer GUI."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import shared logging configuration
from utils.logging_utils import DEFAULT_FORMAT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=DEFAULT_FORMAT,
    handlers=[
        logging.FileHandler("pdf_renamer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

if __name__ == "__main__":
    from pdf_renamer.gui import main

    main()
