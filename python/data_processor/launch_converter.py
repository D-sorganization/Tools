#!/usr/bin/env python3
"""Launch script for the data converter application."""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from data_processor.Data_Processor_Integrated import DataProcessorApp
    from PyQt6.QtWidgets import QApplication

    def main() -> None:
        """Main entry point for the data converter application.

        """
        app = QApplication(sys.argv)
        window = DataProcessorApp()
        window.show()
        sys.exit(app.exec())

    if __name__ == "__main__":
        main()

except ImportError as e:
    logging.error(f"Error importing required modules: {e}")
    logging.error("Please ensure all dependencies are installed:")
    logging.error("pip install PyQt6 pandas numpy")
    sys.exit(1)
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    sys.exit(1)
