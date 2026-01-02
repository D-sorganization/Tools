"""Interactive API key setup for PDF Renamer."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_renamer.config import setup_api_key_interactive, get_api_key, _find_key_location


def main():
    """Run interactive API key setup."""
    print("\n" + "="*60)
    print("PDF Renamer - API Key Configuration")
    print("="*60)

    # Check current status
    existing_key = get_api_key()

    if existing_key:
        print(f"\n✓ API key is already configured!")
        print(f"  Location: {_find_key_location()}")
        print(f"  Key preview: {existing_key[:8]}...{existing_key[-4:]}")
        print("\nAI features are enabled and ready to use.")

        response = input("\nDo you want to reconfigure? (y/N): ").strip().lower()
        if response != 'y':
            print("\nNo changes made. Exiting.")
            return
    else:
        print("\n⚠ No API key found.")
        print("\nAI features require a Gemini API key for title extraction.")
        print("Without it, the tool will use local extraction only (metadata + heuristics).")

    # Run interactive setup
    if setup_api_key_interactive():
        print("\n" + "="*60)
        print("Setup Complete!")
        print("="*60)
        print("\n✓ API key configured successfully")
        print("✓ AI features are now enabled")
        print("\nYou can now use the PDF Renamer with AI-powered extraction:")
        print("  • GUI: python launch_gui.py")
        print("  • CLI: python -m src.pdf_renamer.cli /path/to/pdfs --provider gemini")
    else:
        print("\n" + "="*60)
        print("Setup Cancelled or Failed")
        print("="*60)
        print("\nYou can still use the PDF Renamer with local extraction only.")
        print("To set up the API key later, run this script again:")
        print("  python setup_api_key.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
