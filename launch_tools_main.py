#!/usr/bin/env python3
"""Main Tools Launcher - Robust launcher with error handling and dependency management.

.. deprecated::
    This legacy launcher is deprecated. Use ``launch.py`` (unified CLI entry
    point) or ``UnifiedToolsLauncher.py`` (PyQt6 GUI) instead::

        python launch.py --list          # see all tools
        python launch.py --tool <name>   # launch a specific tool

This script launches the integrated Tools application with proper error handling.
"""

import logging
import os
import sys
import traceback
from pathlib import Path
from tkinter import Tk, messagebox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("tools_launcher.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Bootstrap imports for development mode (before pip install -e .)
_REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

REPO_ROOT = ensure_paths(_REPO_ROOT)


try:
    from tools.dependency_utils import check_dependencies, install_packages
except ImportError:
    # Fallback to internal dummy if tools package not found
    def check_dependencies(packages: list[str]) -> list[str]:
        return []

    def install_packages(packages: list[str]) -> bool:
        return True


def create_constants_file() -> bool:
    """Create a minimal constants file if it doesn't exist."""
    constants_path = Path("src/data_processing/data_processor/archive/constants.py")

    if not constants_path.exists():
        logger.info("Creating missing constants.py file...")

        constants_content = '''"""
Constants for the Data Processor application.
"""

# File processing constants
MAX_FILE_SIZE_MB = 500
CHUNK_SIZE = 10000
DEFAULT_ENCODING = 'utf-8'

# UI constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FONT_SIZE = 12

# Processing constants
DEFAULT_SAMPLE_RATE = 1000
MAX_PLOT_POINTS = 10000

# Export constants
DEFAULT_DPI = 300
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.json', '.parquet']

# Logging constants
LOG_LEVEL = 'INFO'
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
'''

        try:
            constants_path.parent.mkdir(parents=True, exist_ok=True)
            constants_path.write_text(constants_content)
            logger.info(f"Created constants file: {constants_path}")
            return True
        except PermissionError as e:
            logger.error(f"Permission denied creating constants file: {e}")
            return False
        except OSError as e:
            logger.error(f"OS error creating constants file: {e}")
            return False

    return True


def _set_app_icon(app: object) -> None:
    """Set the application icon from available locations.

    Args:
        app: The application instance with iconbitmap method.
    """
    icon_paths = [
        Path("../../../tools_icon.ico"),
        Path("tools_icon.ico"),
    ]
    for icon_path in icon_paths:
        if icon_path.exists():
            try:
                app.iconbitmap(str(icon_path))  # type: ignore[attr-defined]
                logger.info("✓ Applied tools_icon.ico to Integrated Data Processor")
                return
            except (OSError, AttributeError) as e:
                logger.warning(f"Could not set tools_icon from {icon_path}: {e}")


def _log_available_tabs() -> None:
    """Log the available tabs in the integrated launcher."""
    logger.info("Starting Integrated Tools Launcher...")
    logger.info("Available tabs:")
    logger.info("- Data Processing & Analysis")
    logger.info("- Format Converter")
    logger.info("- Folder Tool")
    logger.info("- DAT File Import")
    logger.info("- Plotting & Visualization")
    logger.info("- Help & Documentation")


def launch_integrated_app() -> bool:
    """Launch the integrated Tools application."""
    app_dir = Path("src/data_processing/data_processor/archive")
    original_cwd = os.getcwd()

    try:
        if app_dir.exists():
            os.chdir(app_dir)
            logger.info(f"Changed directory to: {app_dir.resolve()}")

        from Data_Processor_Integrated import IntegratedCSVProcessorApp

        _log_available_tabs()
        app = IntegratedCSVProcessorApp()
        _set_app_icon(app)
        app.mainloop()
        return True

    except ImportError as e:
        logger.error(f"Import error launching integrated app: {e}")
        logger.error(traceback.format_exc())
        return False
    except (OSError, RuntimeError) as e:
        logger.error(f"Failed to launch integrated app: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        os.chdir(original_cwd)


def launch_fallback_app() -> bool:
    """Launch a fallback simple launcher if the main app fails."""
    try:
        logger.info("Launching fallback launcher...")

        # Try the refactored GUI
        from gui_refactored import DataProcessorGUI

        app = DataProcessorGUI()
        app.mainloop()

        return True

    except (ImportError, OSError, RuntimeError) as e:
        logger.error(f"Fallback launcher also failed: {e}")
        return False


def show_error_dialog(message: str) -> None:
    """Show error dialog to user."""
    try:
        log_path = os.path.abspath("tools_launcher.log")
        enhanced_message = f"{message}\n\nLog file location: {log_path}"

        root = Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror("Tools Launcher Error", enhanced_message)
        root.destroy()
    except (OSError, RuntimeError):
        # If GUI fails, just print to console
        logger.error(f"ERROR: {message}")


def _handle_missing_dependencies(missing_packages: list[str]) -> bool:
    """Handle missing package dependencies by prompting user to install.

    Args:
        missing_packages: List of missing package names.

    Returns:
        True if dependencies were resolved, False otherwise.
    """
    logger.warning(f"Missing packages: {missing_packages}")
    try:
        root = Tk()
        root.withdraw()
        install = messagebox.askyesno(
            "Missing Dependencies",
            f"The following packages are missing:\n{', '.join(missing_packages)}\n\n"
            "Would you like to install them automatically?",
        )
        root.destroy()

        if install:
            return install_packages(missing_packages)  # type: ignore[no-any-return]
        return False

    except (OSError, RuntimeError) as e:
        logger.error(f"Dependency installation dialog failed: {e}")
        show_error_dialog(
            f"Missing required packages: {', '.join(missing_packages)}\n\n"
            "Please install them manually:\n"
            f"pip install {' '.join(missing_packages)}"
        )
        return False


def _try_launch_apps() -> bool:
    """Try to launch the main app, falling back to alternative if needed.

    Returns:
        True if any app launched successfully, False otherwise.
    """
    if launch_integrated_app():
        logger.info("Tools Launcher completed successfully")
        return True

    logger.warning("Main app failed, trying fallback...")
    if launch_fallback_app():
        logger.info("Fallback launcher completed successfully")
        return True

    return False


def main() -> bool:
    """Main launcher function."""
    logger.info("=" * 60)
    logger.info("Starting Main Tools Launcher")
    logger.info("=" * 60)

    try:
        # Path setup is handled at module level now

        if not create_constants_file():
            raise RuntimeError("Failed to create required constants file")

        required_packages = ["customtkinter", "pandas", "numpy", "matplotlib", "PIL"]
        missing_packages = check_dependencies(required_packages)
        if missing_packages and not _handle_missing_dependencies(missing_packages):
            raise RuntimeError("Required packages are missing")

        if not _try_launch_apps():
            raise RuntimeError("All launcher attempts failed")

        return True

    except Exception as e:
        error_msg = f"Tools Launcher failed to start: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        show_error_dialog(
            f"{error_msg}\n\n"
            "Please check the log file 'tools_launcher.log' for details.\n\n"
            "Common solutions:\n"
            "1. Install missing Python packages\n"
            "2. Check Python installation\n"
            "3. Run from the correct directory"
        )

        return False


if __name__ == "__main__":
    success = main()
    if not success:
        input("Press Enter to exit...")
