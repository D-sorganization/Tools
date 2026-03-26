"""Interactive API key setup for PDF Renamer."""

import logging

logger = logging.getLogger(__name__)

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)

from pdf_renamer.config import (  # noqa: E402
    _find_key_location,
    get_api_key,
    setup_api_key_interactive,
)


def main() -> None:
    """Run interactive API key setup."""
    logger.info("\n" + "=" * 60)
    logger.info("PDF Renamer - API Key Configuration")
    logger.info("=" * 60)

    # Check current status
    existing_key = get_api_key()

    if existing_key:
        logger.info("\n✓ API key is already configured!")
        logger.info(f"  Location: {_find_key_location()}")
        logger.info("  Key: [hidden for security]")
        logger.info("\nAI features are enabled and ready to use.")

        response = input("\nDo you want to reconfigure? (y/N): ").strip().lower()
        if response != "y":
            logger.info("\nNo changes made. Exiting.")
            return
    else:
        logger.info("\n⚠ No API key found.")
        logger.info("\nAI features require a Gemini API key for title extraction.")

    # Run interactive setup
    if setup_api_key_interactive():
        logger.info("\n" + "=" * 60)
        logger.info("Setup Complete!")
        logger.info("=" * 60)
        logger.info("\n✓ API key configured successfully")
        logger.info("✓ AI features are now enabled")
        logger.info("\nYou can now use the PDF Renamer with AI-powered extraction:")
        logger.info("  • GUI: python launch_gui.py")
        logger.info("  • CLI: python -m src.pdf_renamer.cli /path/to/pdfs --provider gemini")
    else:
        logger.info("\n" + "=" * 60)
        logger.error("Setup Cancelled or Failed")
        logger.info("=" * 60)
        logger.info("\nYou can still use the PDF Renamer with local extraction only.")
        logger.info("To set up the API key later, run this script again:")
        logger.info("  python setup_api_key.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        import traceback

        traceback.print_exc()
