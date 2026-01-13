"""Verify PDF Renamer installation and dependencies."""

import os
import sys

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check_dependency(name: str, import_statement: str) -> bool:
    """Check if a dependency is installed."""
    try:
        # We execute in a local scope, but we need standard modules like sys available if referenced
        exec(import_statement, {"sys": sys, "os": os})
        print(f"{GREEN}[OK]{RESET} {name}")
        return True
    except ImportError as e:
        print(f"{RED}[FAIL]{RESET} {name} - {e}")
        return False
    except Exception as e:
        print(f"{YELLOW}[WARN]{RESET} {name} - {e}")
        return False


def main() -> None:
    """Run installation verification."""
    print(f"\n{BOLD}PDF Renamer - Installation Verification{RESET}\n")
    print("=" * 50)

    all_good = True

    # Core dependencies
    print(f"\n{BOLD}Core Dependencies:{RESET}")
    all_good &= check_dependency("Python 3.11+", "assert sys.version_info >= (3, 11)")
    all_good &= check_dependency("PyPDF", "import pypdf")
    all_good &= check_dependency("PyMuPDF (fitz)", "import fitz")
    all_good &= check_dependency("PyQt6", "from PyQt6.QtWidgets import QApplication")

    # Optional dependencies
    print(f"\n{BOLD}Optional Dependencies:{RESET}")
    llm_ok = check_dependency("Google Generative AI", "import google.generativeai")
    check_dependency("pdfplumber", "import pdfplumber")

    # Project modules
    print(f"\n{BOLD}Project Modules:{RESET}")
    # We need to make sure the project root is in path for these to work if run from this file
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    )

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
    print(f"\n{BOLD}Environment:{RESET}")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        print(f"{GREEN}[OK]{RESET} GEMINI_API_KEY set (AI features available)")
    else:
        print(f"{YELLOW}[WARN]{RESET} GEMINI_API_KEY not set (AI features disabled)")

    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print(f"\n{GREEN}{BOLD}[SUCCESS] All required dependencies installed!{RESET}")
        print(f"\n{BOLD}Ready to use:{RESET}")
        print("  - GUI: python launch_gui.py")
        print("  - CLI: python -m src.pdf_renamer.cli /path/to/pdfs --dry-run")
        if not llm_ok:
            print(
                f"\n{YELLOW}Note:{RESET} AI features require google-generativeai (optional)"
            )
        if not gemini_key and llm_ok:
            print(
                f"{YELLOW}Note:{RESET} Set GEMINI_API_KEY environment variable to enable AI"
            )
    else:
        print(f"\n{RED}{BOLD}[FAILED] Some dependencies are missing!{RESET}")
        print(f"\n{BOLD}To fix:{RESET}")
        print("  pip install -r requirements.txt")

    print()


if __name__ == "__main__":
    main()
