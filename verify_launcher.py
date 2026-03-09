#!/usr/bin/env python3
"""
Launcher Verification Script
This script helps identify which launcher file is being executed.
"""

import os
import sys
from datetime import datetime
from pathlib import Path


def _print_environment_info(current_file: Path) -> None:
    """Print current environment information."""
    print("=" * 60)
    print("🔍 LAUNCHER VERIFICATION SCRIPT")
    print("=" * 60)
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📄 This script location: {current_file}")
    print(f"🐍 Python executable: {sys.executable}")
    print(f"📋 Python version: {sys.version}")


def _check_launcher_file(launcher_path: Path) -> None:
    """Check and display info about a launcher file."""
    if not launcher_path.exists():
        print(f"❌ {launcher_path.name} - NOT FOUND")
        return

    size = launcher_path.stat().st_size
    modified = launcher_path.stat().st_mtime
    mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S")

    print(f"✅ {launcher_path.name}")
    print(f"   📏 Size: {size:,} bytes")
    print(f"   📅 Modified: {mod_time}")

    try:
        with open(launcher_path, encoding="utf-8") as f:
            first_lines = [f.readline().strip() for _ in range(5)]
            for i, line in enumerate(first_lines, 1):
                if line and not line.startswith("#"):
                    print(f"   📝 Line {i}: {line[:60]}...")
                    break
    except (OSError, UnicodeDecodeError) as e:
        print(f"   ❌ Could not read file: {e}")
    print()


def _print_recommendations(current_dir: Path) -> None:
    """Print recommendations for which launcher to use."""
    print("=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)

    unified_launcher = current_dir / "UnifiedToolsLauncher.py"
    if unified_launcher.exists():
        print("✅ Use 'UnifiedToolsLauncher.py' - This is the PRIMARY launcher with:")
        print("   • Modern PyQt6 GUI interface")
        print("   • Full plugin system support")
        print("   • Comprehensive error handling")
        print("   • Tool path validation and security")
        print("   • Output/error capture")
        print(f"\n🚀 To launch: python {unified_launcher}")
    else:
        print("❌ UnifiedToolsLauncher.py not found!")

    print("\n⚠️  Note: 'tools_launcher.py' does not exist.")
    print("   Any references to it are outdated. Use UnifiedToolsLauncher.py instead.")


def _check_desktop_shortcuts() -> None:
    """Check for desktop shortcuts."""
    print("\n" + "=" * 60)
    print("🔗 DESKTOP SHORTCUTS")
    print("=" * 60)

    desktop_path = Path.home() / "OneDrive" / "Desktop"
    if not desktop_path.exists():
        desktop_path = Path.home() / "Desktop"

    shortcuts = list(desktop_path.glob("*Tools*Launcher*.lnk"))
    if shortcuts:
        print("Found desktop shortcuts:")
        for shortcut in shortcuts:
            print(f"🔗 {shortcut.name}")
    else:
        print("❌ No Tools Launcher shortcuts found on desktop")
        print("💡 Run 'create_launcher_shortcut.ps1' to create one")


def main() -> None:
    """Main verification function."""
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    _print_environment_info(current_file)

    print("\n" + "=" * 60)
    print("🚀 AVAILABLE LAUNCHER FILES")
    print("=" * 60)

    launcher_files = ["launch.py", "UnifiedToolsLauncher.py"]
    for launcher in launcher_files:
        _check_launcher_file(current_dir / launcher)

    _print_recommendations(current_dir)
    _check_desktop_shortcuts()


if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
