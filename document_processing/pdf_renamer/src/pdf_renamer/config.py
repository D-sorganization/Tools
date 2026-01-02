"""Configuration management for PDF Renamer with secure API key handling."""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to load from .env file if available
try:
    from dotenv import load_dotenv

    # Search for .env file in multiple locations
    env_locations = [
        Path(__file__).parent.parent.parent / ".env",  # Project root
        Path.cwd() / ".env",  # Current working directory
        Path.home() / ".pdf_renamer" / ".env",  # User home directory
        Path("c:/Users/diete/Repositories/Tools/document_processing/pdf_renamer/.env"),  # Tools version
    ]

    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    # python-dotenv not installed, will fall back to environment variables
    pass


def get_api_key(key_name: str = "GEMINI_API_KEY") -> str | None:
    """
    Get API key from multiple sources with priority order.

    Priority:
    1. Environment variable (current session)
    2. .env file in project root
    3. .env file in Tools folder
    4. .env file in user home (~/.pdf_renamer/.env)

    Args:
        key_name: Name of the API key (default: GEMINI_API_KEY)

    Returns:
        API key string or None if not found
    """
    # First check environment variable
    api_key = os.environ.get(key_name)
    if api_key:
        return api_key

    # If not in environment, try to manually load from known locations
    env_locations = [
        Path(__file__).parent.parent.parent / ".env",  # Project root
        Path("c:/Users/diete/Repositories/Tools/document_processing/pdf_renamer/.env"),  # Tools version
        Path.home() / ".pdf_renamer" / ".env",  # User home
    ]

    for env_path in env_locations:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            var_name, var_value = line.split('=', 1)
                            if var_name.strip() == key_name:
                                return var_value.strip().strip('"').strip("'")
            except Exception:
                continue

    return None


def setup_api_key_interactive() -> bool:
    """
    Interactive setup for API key. Prompts user and saves to .env file.

    Returns:
        True if API key was set up successfully, False otherwise
    """
    print("\n" + "="*60)
    print("API Key Setup")
    print("="*60)

    existing_key = get_api_key()
    if existing_key:
        print("\n✓ API key already configured!")
        print(f"  Found in: {_find_key_location()}")

        response = input("\nDo you want to update it? (y/N): ").strip().lower()
        if response != 'y':
            return True

    print("\nTo use AI-powered title extraction, you need a Gemini API key.")
    print("Get your free API key at: https://makersuite.google.com/app/apikey")
    print("\nNote: AI features are optional. You can skip this and use local extraction only.")

    response = input("\nWould you like to set up your API key now? (y/N): ").strip().lower()
    if response != 'y':
        print("\nSkipping API key setup. You can set it later by:")
        print("  1. Creating a .env file with: GEMINI_API_KEY=your_key")
        print("  2. Setting environment variable: GEMINI_API_KEY=your_key")
        return False

    api_key = input("\nEnter your Gemini API key: ").strip()
    if not api_key:
        print("\n✗ No API key entered. Setup cancelled.")
        return False

    # Choose save location
    print("\nWhere should I save the API key?")
    print("  1. Project folder (Playground/PDFRenamer/.env)")
    print("  2. Tools folder (Tools/document_processing/pdf_renamer/.env)")
    print("  3. User home (~/.pdf_renamer/.env)")

    choice = input("\nChoice (1-3, default=1): ").strip() or "1"

    save_locations = {
        "1": Path(__file__).parent.parent.parent / ".env",
        "2": Path("c:/Users/diete/Repositories/Tools/document_processing/pdf_renamer/.env"),
        "3": Path.home() / ".pdf_renamer" / ".env",
    }

    env_path = save_locations.get(choice, save_locations["1"])

    # Create directory if needed
    env_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to .env file
    try:
        # Read existing content
        existing_content = []
        if env_path.exists():
            with open(env_path) as f:
                existing_content = [line for line in f if not line.strip().startswith('GEMINI_API_KEY')]

        # Write new content
        with open(env_path, 'w') as f:
            # Write header
            f.write("# PDF Renamer Configuration\n")
            f.write("# Auto-generated API key configuration\n\n")

            # Write API key
            f.write(f"GEMINI_API_KEY={api_key}\n")

            # Write back other existing variables
            if existing_content:
                f.write("\n# Other settings\n")
                for line in existing_content:
                    if line.strip() and not line.strip().startswith('#'):
                        f.write(line)

        print(f"\n✓ API key saved to: {env_path}")
        print("  File is gitignored and secure.")
        print("\nAI features are now enabled!")
        return True

    except Exception as e:
        print(f"\n✗ Failed to save API key: {e}")
        print(f"\nYou can manually create {env_path} with:")
        print(f"  GEMINI_API_KEY={api_key}")
        return False


def _find_key_location() -> str:
    """Find where the API key is configured."""
    if os.environ.get("GEMINI_API_KEY"):
        return "Environment variable"

    env_locations = [
        (Path(__file__).parent.parent.parent / ".env", "Project folder"),
        (Path("c:/Users/diete/Repositories/Tools/document_processing/pdf_renamer/.env"), "Tools folder"),
        (Path.home() / ".pdf_renamer" / ".env", "User home"),
    ]

    for env_path, location in env_locations:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        if line.strip().startswith('GEMINI_API_KEY='):
                            return f"{location} ({env_path})"
            except Exception:
                continue

    return "Unknown"


def get_config_dir() -> Path:
    """Get the configuration directory for storing user preferences."""
    config_dir = Path.home() / ".pdf_renamer"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_user_preferences() -> dict[str, Any]:
    """Load user preferences from config file."""
    config_file = get_config_dir() / "preferences.json"
    default_prefs = {
        "last_directory": str(Path.home()),
        "default_style": "standard",
        "default_workers": 4,
        "remember_settings": True,
        "create_failed_folder": True,
        "failed_folder_name": "failed_renames"
    }

    if not config_file.exists():
        save_user_preferences(default_prefs)
        return default_prefs

    try:
        with open(config_file, encoding='utf-8') as f:
            prefs = json.load(f)
            # Merge with defaults to handle new settings
            for key, value in default_prefs.items():
                if key not in prefs:
                    prefs[key] = value
            return prefs
    except Exception as e:
        logger.warning(f"Failed to load preferences: {e}. Using defaults.")
        return default_prefs


def save_user_preferences(preferences: dict[str, Any]) -> None:
    """Save user preferences to config file."""
    config_file = get_config_dir() / "preferences.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save preferences: {e}")


def update_last_directory(directory: str) -> None:
    """Update the last used directory in preferences."""
    prefs = get_user_preferences()
    prefs["last_directory"] = directory
    save_user_preferences(prefs)


# Auto-load .env on import
try:
    from dotenv import load_dotenv
    for env_path in [
        Path(__file__).parent.parent.parent / ".env",
        Path("c:/Users/diete/Repositories/Tools/document_processing/pdf_renamer/.env"),
        Path.home() / ".pdf_renamer" / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass
