#!/usr/bin/env python3
"""
Main Tools Launcher - Robust launcher with error handling and dependency management.
This script launches the integrated Tools application with proper error handling.
"""

import logging
import os
import sys
import traceback
from pathlib import Path
from tkinter import Tk, messagebox

# Configure logging
try:
    from tools.logger import setup_logging

    logger = setup_logging(__name__, "tools_launcher.log")
except ImportError:
    # Fallback if tools package issue (e.g. during very early bootstrap)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("tools_launcher.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)


# Use shared path setup utility
try:
    from utils.path_setup import get_repo_root, setup_python_path

    # Standard setup
    REPO_ROOT = get_repo_root()
    setup_python_path(repo_root=REPO_ROOT)

except ImportError:
    # Fallback if unimportable (should rarely happen if structure is valid)
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir / "src" / "python" / "src"))
    try:
        from utils.path_setup import get_repo_root, setup_python_path

        REPO_ROOT = get_repo_root()
        setup_python_path(repo_root=REPO_ROOT)
    except ImportError:
        logger.warning("Could not import utils.path_setup even after path patch.")


def check_dependencies() -> list[str]:
    """Check if required dependencies are available."""
    try:
        from tools.dependency_utils import check_dependencies as check_deps

        required_packages = [
            "customtkinter",
            "pandas",
            "numpy",
            "matplotlib",
            "PIL",
        ]
        return check_deps(required_packages)
    except ImportError:
        logger.warning("Could not import tools.dependency_utils")
        # Fallback minimal check
        return []


def install_missing_packages(packages: list[str]) -> bool:
    """Attempt to install missing packages."""
    try:
        from tools.dependency_utils import install_packages

        return install_packages(packages)
    except ImportError:
        logger.error("Could not import tools.dependency_utils for installation")
        return False


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
        from tools.ui_utils import set_tk_icon

        if app_dir.exists():
            os.chdir(app_dir)
            logger.info(f"Changed directory to: {app_dir.resolve()}")

        from Data_Processor_Integrated import IntegratedCSVProcessorApp

        _log_available_tabs()
        app = IntegratedCSVProcessorApp()

        # Determine if app is a root or we need to set icon on it roughly
        # IntegratedCSVProcessorApp is likely a CTk/Tk subclass
        set_tk_icon(app)

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


def show_error_dialog(message: str) -> None:
    """Show error dialog to user."""
    try:
        root = Tk()
        root.withdraw()
        messagebox.showerror("Tools Launcher Error", message)
        root.destroy()
    except (OSError, RuntimeError):
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
            return install_missing_packages(missing_packages)
        return False

    except (OSError, RuntimeError) as e:
        logger.error(f"Dependency installation dialog failed: {e}")
        show_error_dialog(
            f"Missing required packages: {', '.join(missing_packages)}\n\n"
            "Please install them manually:\n"
            f"pip install {' '.join(missing_packages)}"
        )
        return False


def main() -> bool:
    """Main launcher function."""
    logger.info("=" * 60)
    logger.info("Starting Main Tools Launcher")
    logger.info("=" * 60)

    try:
        # Path setup is handled at module level now

        missing_packages = check_dependencies()
        if missing_packages and not _handle_missing_dependencies(missing_packages):
            raise RuntimeError("Required packages are missing")

        if not launch_integrated_app():
            raise RuntimeError("Integrated app launch failed")

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
