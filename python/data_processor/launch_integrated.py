#!/usr/bin/env python3
"""Launch script for the Integrated Data Processor.

This script launches the integrated version that includes the compiler converter functionality.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from Data_Processor_Integrated import IntegratedCSVProcessorApp

    if __name__ == "__main__":
        logging.info("Starting Integrated Data Processor...")
        logging.info("This version includes:")
        logging.info("- Original CSV processing functionality")
        logging.info("- Format converter with support for multiple file formats")
        logging.info("- Parquet file analyzer")
        logging.info("- All existing plotting and analysis features")
        logging.info("")

        app = IntegratedCSVProcessorApp()
        app.mainloop()

except ImportError as e:
    logging.exception(f"Error importing required modules: {e}")
    logging.exception("Please ensure all dependencies are installed:")
    logging.exception(
        "pip install customtkinter pandas numpy scipy matplotlib openpyxl "
        "Pillow simpledbf pyarrow tables feather-format"
    )
    input("Press Enter to exit...")
except Exception as e:
    logging.exception(f"Error starting application: {e}")
    input("Press Enter to exit...")
