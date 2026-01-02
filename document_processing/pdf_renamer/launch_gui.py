"""Launcher script for PDF Renamer GUI."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_renamer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

if __name__ == "__main__":
    from pdf_renamer.gui import main

    main()
