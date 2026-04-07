#!/usr/bin/env python3
"""
Launcher Verification Script
This script helps identify which launcher file is being executed.
"""

import os
import sys
from datetime import datetime
from pathlib import Path


def _emit_stdout(message: str = "") -> None:
    """Write a single line to stdout for this diagnostic CLI."""
    sys.stdout.write(f"{message}\n")


def _print_environment_info(current_file: Path) -> None:
    """Print current environment information.

    Args:
        current_file: Resolved path to this script file.

    Raises:
        TypeError: If current_file is not a Path.
    """
    if not isinstance(current_file, Path):
        raise TypeError(f"current_file must be a Path, got {type(current_file)}")
    _emit_stdout("=" * 60)
    _emit_stdout("🔍 LAUNCHER VERIFICATION SCRIPT")
    _emit_stdout("=" * 60)
    _emit_stdout(f"📁 Current working directory: {os.getcwd()}")
    _emit_stdout(f"📄 This script location: {current_file}")
    _emit_stdout(f"🐍 Python executable: {sys.executable}")
    _emit_stdout(f"📋 Python version: {sys.version}")


def _check_launcher_file(launcher_path: Path) -> None:
    """Check and display info about a launcher file.

    Args:
        launcher_path: Path to the launcher file to inspect.

    Raises:
        TypeError: If launcher_path is not a Path.
    """
    if not isinstance(launcher_path, Path):
        raise TypeError(f"launcher_path must be a Path, got {type(launcher_path)}")
    if not launcher_path.exists():
        _emit_stdout(f"❌ {launcher_path.name} - NOT FOUND")
        return

    stat = launcher_path.stat()
    size = stat.st_size
    modified = stat.st_mtime
    mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S")

    _emit_stdout(f"✅ {launcher_path.name}")
    _emit_stdout(f"   📏 Size: {size:,} bytes")
    _emit_stdout(f"   📅 Modified: {mod_time}")

    try:
        with open(launcher_path, encoding="utf-8") as f:
            first_lines = [f.readline().strip() for _ in range(5)]
            for i, line in enumerate(first_lines, 1):
                if line and not line.startswith("#"):
                    _emit_stdout(f"   📝 Line {i}: {line[:60]}...")
                    break
    except (OSError, UnicodeDecodeError) as e:
        _emit_stdout(f"   ❌ Could not read file: {e}")
    _emit_stdout()


def _print_recommendations(current_dir: Path) -> None:
    """Print recommendations for which launcher to use.

    Args:
        current_dir: The repository root directory to check.

    Raises:
        TypeError: If current_dir is not a Path.
    """
    if not isinstance(current_dir, Path):
        raise TypeError(f"current_dir must be a Path, got {type(current_dir)}")
    _emit_stdout("=" * 60)
    _emit_stdout("💡 RECOMMENDATIONS")
    _emit_stdout("=" * 60)

    unified_launcher = current_dir / "UnifiedToolsLauncher.py"
    if unified_launcher.exists():
        _emit_stdout("✅ Use 'UnifiedToolsLauncher.py' - This is the PRIMARY launcher with:")
        _emit_stdout("   • Modern PyQt6 GUI interface")
        _emit_stdout("   • Full plugin system support")
        _emit_stdout("   • Comprehensive error handling")
        _emit_stdout("   • Tool path validation and security")
        _emit_stdout("   • Output/error capture")
        _emit_stdout()
        _emit_stdout(f"🚀 To launch: python {unified_launcher}")
    else:
        _emit_stdout("❌ UnifiedToolsLauncher.py not found!")

    _emit_stdout()
    _emit_stdout("⚠️  Note: 'tools_launcher.py' does not exist.")
    _emit_stdout("   Any references to it are outdated. Use UnifiedToolsLauncher.py instead.")


def _check_desktop_shortcuts() -> None:
    """Check for desktop shortcuts."""
    _emit_stdout()
    _emit_stdout("=" * 60)
    _emit_stdout("🔗 DESKTOP SHORTCUTS")
    _emit_stdout("=" * 60)

    desktop_path = Path.home() / "OneDrive" / "Desktop"
    if not desktop_path.exists():
        desktop_path = Path.home() / "Desktop"

    shortcuts = list(desktop_path.glob("*Tools*Launcher*.lnk"))
    if shortcuts:
        _emit_stdout("Found desktop shortcuts:")
        for shortcut in shortcuts:
            _emit_stdout(f"🔗 {shortcut.name}")
    else:
        _emit_stdout("❌ No Tools Launcher shortcuts found on desktop")
        _emit_stdout("💡 Run 'create_launcher_shortcut.ps1' to create one")


def main() -> None:
    """Main verification function."""
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    _print_environment_info(current_file)

    _emit_stdout()
    _emit_stdout("=" * 60)
    _emit_stdout("🚀 AVAILABLE LAUNCHER FILES")
    _emit_stdout("=" * 60)

    launcher_files = ["launch.py", "UnifiedToolsLauncher.py"]
    for launcher in launcher_files:
        _check_launcher_file(current_dir / launcher)

    _print_recommendations(current_dir)
    _check_desktop_shortcuts()


if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
