"""Verify PDF Renamer installation and dependencies."""

import importlib
import os
import re
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check_dependency(name: str, import_statement: str) -> bool:
    """Check if a dependency is installed."""
    try:
        # Handle 'assert' statements (e.g., version checks) directly
        if import_statement.startswith("assert"):
            # Parse "assert sys.version_info >= (3, 11)" style checks
            if "sys.version_info" in import_statement:
                match = re.search(r"\((\d+),\s*(\d+)\)", import_statement)
                if match:
                    major, minor = int(match.group(1)), int(match.group(2))
                    assert sys.version_info >= (major, minor)
                else:
                    raise ValueError(
                        f"Cannot parse version assertion: {import_statement}"
                    )
            else:
                raise ValueError(f"Unsupported assertion: {import_statement}")
        # Handle 'from X import Y' statements
        elif import_statement.startswith("from "):
            match = re.match(r"from\s+([\w.]+)\s+import\s+(\w+)", import_statement)
            if match:
                module_path, attr_name = match.group(1), match.group(2)
                mod = importlib.import_module(module_path)
                getattr(mod, attr_name)
            else:
                raise ValueError(f"Cannot parse from-import: {import_statement}")
        # Handle 'import X' statements
        elif import_statement.startswith("import "):
            module_name = import_statement.replace("import ", "").strip()
            importlib.import_module(module_name)
        else:
            raise ValueError(f"Unsupported statement: {import_statement}")

        logger.info(f"{GREEN}[OK]{RESET} {name}")
        return True
    except ImportError as e:
        logger.error(f"{RED}[FAIL]{RESET} {name} - {e}")
        return False


def main() -> None:
    """Run installation verification."""
    logger.info(f"\n{BOLD}PDF Renamer - Installation Verification{RESET}\n")
    logger.info("=" * 50)

    all_good = True

    # Core dependencies
    logger.info(f"\n{BOLD}Core Dependencies:{RESET}")
    all_good &= check_dependency("Python 3.11+", "assert sys.version_info >= (3, 11)")
    all_good &= check_dependency("PyPDF", "import pypdf")
    all_good &= check_dependency("PyMuPDF (fitz)", "import fitz")
    all_good &= check_dependency("PyQt6", "from PyQt6.QtWidgets import QApplication")

    # Optional dependencies
    logger.info(f"\n{BOLD}Optional Dependencies:{RESET}")
    llm_ok = check_dependency("Google Generative AI", "import google.generativeai")
    check_dependency("pdfplumber", "import pdfplumber")

    # Project modules
    logger.info(f"\n{BOLD}Project Modules:{RESET}")
    # We need to make sure the project root is in path for these to work if run from this file
    sys.path.append(os.path.abspath(Path(Path(__file__).parent, "../../..")))

    all_good &= check_dependency("extractors", "from src.pdf_renamer import extractors")
    all_good &= check_dependency("core", "from src.pdf_renamer import core")
    all_good &= check_dependency("worker", "from src.pdf_renamer import worker")
    all_good &= check_dependency("cache", "from src.pdf_renamer import cache")
    all_good &= check_dependency("deduper", "from src.pdf_renamer import deduper")
    all_good &= check_dependency(
        "transaction_log", "from src.pdf_renamer import transaction_log"
    )
    all_good &= check_dependency("utils", "from src.pdf_renamer import utils")
    all_good &= check_dependency("gui", "from src.pdf_renamer import gui")

    # Environment checks
    logger.info(f"\n{BOLD}Environment:{RESET}")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        logger.info(f"{GREEN}[OK]{RESET} GEMINI_API_KEY set (AI features available)")
    else:
        logger.warning(
            f"{YELLOW}[WARN]{RESET} GEMINI_API_KEY not set (AI features disabled)"
        )

    # Summary
    logger.info("\n" + "=" * 50)
    if all_good:
        logger.info(
            f"\n{GREEN}{BOLD}[SUCCESS] All required dependencies installed!{RESET}"
        )
        logger.info(f"\n{BOLD}Ready to use:{RESET}")
        logger.info("  - GUI: python launch_gui.py")
        logger.info("  - CLI: python -m src.pdf_renamer.cli /path/to/pdfs --dry-run")
        if not llm_ok:
            print(
                f"\n{YELLOW}Note:{RESET} AI features require google-generativeai (optional)"
            )
        if not gemini_key and llm_ok:
            print(
                f"{YELLOW}Note:{RESET} Set GEMINI_API_KEY environment variable to enable AI"
            )
    else:
        logger.error(f"\n{RED}{BOLD}[FAILED] Some dependencies are missing!{RESET}")
        logger.info(f"\n{BOLD}To fix:{RESET}")
        logger.info("  pip install -r requirements.txt")

    print()


if __name__ == "__main__":
    main()
