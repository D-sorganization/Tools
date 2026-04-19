"""Configuration management for PDF Renamer with secure API key handling."""

import logging
import os
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)

try:
    from utils.file_utils import safe_read_json, safe_write_json
except ImportError:
    import json

    # Final fallback - inline implementations
    def safe_read_json(file_path: Path | str, default: Any = None) -> Any:
        """Fallback safe JSON reader."""
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        path = Path(file_path)
        if not path.exists():
            return default
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return default

    def safe_write_json(
        file_path: Path | str,
        data: Any,
        indent: int = 2,
        create_parents: bool = True,
    ) -> bool:
        """Fallback safe JSON writer."""
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        path = Path(file_path)
        try:
            if create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            return True
        except (TypeError, OSError):
            return False


KEYRING_SERVICE = "pdf_renamer"
KEYRING_USERNAME = "gemini"
API_KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _get_keyring() -> Any | None:
    """Return the optional keyring module when it is available."""
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def get_api_key(key_name: str = "GEMINI_API_KEY") -> str | None:
    """
    Get API key from secure sources with priority order.

    Priority:
    1. Environment variable (current session) - checks both GEMINI_API_KEY and
       GOOGLE_API_KEY
    2. OS keyring entry for the pdf_renamer Gemini credential

    Args:
        key_name: Name of the API key (default: GEMINI_API_KEY)

    Returns:
        API key string or None if not found
    """
    for candidate_name in (key_name, *API_KEY_NAMES):
        api_key = os.environ.get(candidate_name)
        if api_key:
            return api_key

    keyring = _get_keyring()
    if keyring is not None:
        try:
            return cast(
                "str | None", keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
            )
        except Exception as exc:  # noqa: BLE001 - keyring backends vary widely.
            logger.warning("Unable to read Gemini API key from keyring: %s", exc)

    return None


def setup_api_key_interactive() -> bool:
    """
    Interactive setup for API key. Prompts user and stores it in OS keyring.

    Returns:
        True if API key was set up successfully, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("API Key Setup")
    logger.info("=" * 60)

    existing_key = get_api_key()
    if existing_key:
        logger.info("\n✓ API key already configured!")
        logger.info(f"  Found in: {_find_key_location()}")

        response = input("\nDo you want to update it? (y/N): ").strip().lower()
        if response != "y":
            return True

    logger.info("\nTo use AI-powered title extraction, you need a Gemini API key.")
    logger.info("Get your free API key at: https://makersuite.google.com/app/apikey")

    response = (
        input("\nWould you like to set up your API key now? (y/N): ").strip().lower()
    )
    if response != "y":
        logger.info("\nSkipping API key setup. You can set it later by:")
        logger.info("  1. Setting environment variable: GEMINI_API_KEY=your_key")
        logger.info("  2. Installing keyring and rerunning this setup helper")
        return False

    api_key = input("\nEnter your Gemini API key: ").strip()
    if not api_key:
        logger.info("\n✗ No API key entered. Setup cancelled.")
        return False

    keyring = _get_keyring()
    if keyring is None:
        logger.error("\n✗ keyring is not installed; API key was not saved.")
        logger.info("Set GEMINI_API_KEY in your shell environment instead.")
        return False

    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, api_key)
        logger.info("\n✓ API key saved to the OS keyring.")
        logger.info("\nAI features are now enabled!")
        return True

    except Exception as exc:  # noqa: BLE001 - keyring backends vary widely.
        logger.error("\n✗ Failed to save API key to keyring: %s", exc)
        logger.info("Set GEMINI_API_KEY in your shell environment instead.")
        return False


def _find_key_location() -> str:
    """Find where the API key is configured."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "Environment variable"

    keyring = _get_keyring()
    if keyring is not None:
        try:
            if keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME):
                return "OS keyring"
        except Exception as exc:  # noqa: BLE001 - keyring backends vary widely.
            logger.warning("Unable to inspect Gemini API key in keyring: %s", exc)

    return "Unknown"


def get_config_dir() -> Path:
    """Get the configuration directory for storing user preferences."""
    config_dir = Path.home() / ".pdf_renamer"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_user_preferences() -> dict[str, Any]:
    """Load user preferences from config file."""
    config_file = get_config_dir() / "preferences.json"
    default_prefs: dict[str, Any] = {
        "last_directory": str(Path.home()),
        "default_style": "standard",
        "default_workers": 4,
        "remember_settings": True,
        "create_failed_folder": True,
        "failed_folder_name": "failed_renames",
    }

    prefs = safe_read_json(config_file, default=None)
    if prefs is None:
        save_user_preferences(default_prefs)
        return default_prefs

    # Merge with defaults to handle new settings
    for key, value in default_prefs.items():
        if key not in prefs:
            prefs[key] = value
    return dict(prefs)


def save_user_preferences(preferences: dict[str, Any]) -> None:
    """Save user preferences to config file."""
    config_file = get_config_dir() / "preferences.json"
    if not safe_write_json(config_file, preferences):
        logger.error("Failed to save preferences")


def update_last_directory(directory: str) -> None:
    """Update the last used directory in preferences."""
    prefs = get_user_preferences()
    prefs["last_directory"] = directory
    save_user_preferences(prefs)
