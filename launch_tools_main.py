#!/usr/bin/env python3
"""
Main Tools Launcher - Robust launcher with error handling and dependency management.
This script launches the integrated Tools application with proper error handling.
"""

import logging

# Use shared logging utility
try:
    from utils.logging_utils import init_default_logging
except ImportError:
    # Fallback
    def init_default_logging():
        init_default_logging()
import os
import sys
import traceback
from pathlib import Path

# Use shared subprocess utility
try:
    from utils.subprocess_utils import run_command
except ImportError:
    # Fallback
    import subprocess
    run_command = subprocess.run
from tkinter import Tk, messagebox

# Configure logging
init_default_logging()s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("tools_launcher.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_python_path() -> None:
    """Setup Python path for all required modules."""
    current_dir = Path(__file__).resolve().parent

    # Add paths for different components
    paths_to_add = [
        current_dir,
        current_dir / "data_processing" / "data_processor" / "archive",
        current_dir
        / "data_processing"
        / "data_processor"
        / "python"
        / "data_processor",
        current_dir / "replicants" / "python" / "folder_tool",
        current_dir / "tools",
        current_dir / "python" / "src",
    ]

    for path in paths_to_add:
        if path.exists():
            sys.path.insert(0, str(path))
            logger.info(f"Added to Python path: {path}")

    # Also set PYTHONPATH environment variable
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_paths = [str(p) for p in paths_to_add if p.exists()]

    if existing_pythonpath:
        new_pythonpath = os.pathsep.join(new_paths + [existing_pythonpath])
    else:
        new_pythonpath = os.pathsep.join(new_paths)

    os.environ["PYTHONPATH"] = new_pythonpath
    logger.info(f"Set PYTHONPATH: {new_pythonpath}")


def check_dependencies() -> list[str]:
    """Check if required dependencies are available."""
    required_packages = [
        "customtkinter",
        "pandas",
        "numpy",
        "matplotlib",
        "PIL",  # Pillow
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")

    return missing_packages


def install_missing_packages(packages: list[str]) -> bool:
    """Attempt to install missing packages."""
    if not packages:
        return True

    logger.info(f"Attempting to install missing packages: {packages}")

    try:
        import subprocess

        # Map package names to pip names if different
        pip_names = {
            "PIL": "Pillow",
            "customtkinter": "customtkinter",
            "pandas": "pandas",
            "numpy": "numpy",
            "matplotlib": "matplotlib",
        }

        for package in packages:
            pip_name = pip_names.get(package, package)
            logger.info(f"Installing {pip_name}...")

            result = run_command(
                [sys.executable, "-m", "pip", "install", pip_name],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info(f"✓ Successfully installed {pip_name}")
            else:
                logger.error(f"✗ Failed to install {pip_name}: {result.stderr}")
                return False

        return True

    except Exception as e:
        logger.error(f"Error installing packages: {e}")
        return False


def create_constants_file() -> bool:
    """Create a minimal constants file if it doesn't exist."""
    constants_path = Path("data_processing/data_processor/archive/constants.py")

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
        except Exception as e:
            logger.error(f"Failed to create constants file: {e}")
            return False

    return True


def launch_integrated_app() -> bool:
    """Launch the integrated Tools application."""
    try:
        # Change to the correct directory
        app_dir = Path("data_processing/data_processor/archive")
        original_cwd = os.getcwd()

        if app_dir.exists():
            os.chdir(app_dir)
            logger.info(f"Changed directory to: {app_dir.resolve()}")

        # Import and launch the application
        from Data_Processor_Integrated import IntegratedCSVProcessorApp

        logger.info("Starting Integrated Tools Launcher...")
        logger.info("Available tabs:")
        logger.info("- Data Processing & Analysis")
        logger.info("- Format Converter")
        logger.info("- Folder Tool")
        logger.info("- DAT File Import")
        logger.info("- Plotting & Visualization")
        logger.info("- Help & Documentation")

        app = IntegratedCSVProcessorApp()

        # Set the new tools_icon for the application
        try:
            tools_icon_path = Path("../../../tools_icon.ico")
            if tools_icon_path.exists():
                app.iconbitmap(str(tools_icon_path))
                logger.info("✓ Applied tools_icon.ico to Integrated Data Processor")
            else:
                # Try relative path
                tools_icon_path = Path("tools_icon.ico")
                if tools_icon_path.exists():
                    app.iconbitmap(str(tools_icon_path))
                    logger.info("✓ Applied tools_icon.ico to Integrated Data Processor")
        except Exception as e:
            logger.warning(f"Could not set tools_icon: {e}")

        app.mainloop()

        return True

    except Exception as e:
        logger.error(f"Failed to launch integrated app: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Restore original directory
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

    except Exception as e:
        logger.error(f"Fallback launcher also failed: {e}")
        return False


def show_error_dialog(message: str) -> None:
    """Show error dialog to user."""
    try:
        root = Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror("Tools Launcher Error", message)
        root.destroy()
    except Exception:
        # If GUI fails, just print to console
        print(f"ERROR: {message}")


def main() -> bool:
    """Main launcher function."""
    logger.info("=" * 60)
    logger.info("Starting Main Tools Launcher")
    logger.info("=" * 60)

    try:
        # Setup Python path
        setup_python_path()

        # Create missing constants file
        if not create_constants_file():
            raise Exception("Failed to create required constants file")

        # Check dependencies
        missing_packages = check_dependencies()

        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")

            # Ask user if they want to install
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
                    if not install_missing_packages(missing_packages):
                        raise Exception("Failed to install required packages")
                else:
                    raise Exception("Required packages are missing")

            except Exception as e:
                logger.error(f"Dependency installation failed: {e}")
                show_error_dialog(
                    f"Missing required packages: {', '.join(missing_packages)}\n\n"
                    "Please install them manually:\n"
                    f"pip install {' '.join(missing_packages)}"
                )
                return False

        # Try to launch the main integrated app
        if launch_integrated_app():
            logger.info("Tools Launcher completed successfully")
            return True

        # If main app fails, try fallback
        logger.warning("Main app failed, trying fallback...")
        if launch_fallback_app():
            logger.info("Fallback launcher completed successfully")
            return True

        # If everything fails
        raise Exception("All launcher attempts failed")

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
