#!/usr/bin/env python3
"""
Launcher Verification Script
This script helps identify which launcher file is being executed.
"""

import os
import sys
from pathlib import Path


def main() -> None:
    print("=" * 60)
    print("🔍 LAUNCHER VERIFICATION SCRIPT")
    print("=" * 60)

    # Get current script info
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📄 This script location: {current_file}")
    print(f"🐍 Python executable: {sys.executable}")
    print(f"📋 Python version: {sys.version}")

    print("\n" + "=" * 60)
    print("🚀 AVAILABLE LAUNCHER FILES")
    print("=" * 60)

    # Check for different launcher files
    launcher_files = ["UnifiedToolsLauncher.py", "launch_tools_main.py"]

    for launcher in launcher_files:
        launcher_path = current_dir / launcher
        if launcher_path.exists():
            size = launcher_path.stat().st_size
            modified = launcher_path.stat().st_mtime
            from datetime import datetime

            mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S")

            print(f"✅ {launcher}")
            print(f"   📏 Size: {size:,} bytes")
            print(f"   📅 Modified: {mod_time}")

            # Check first few lines to identify the launcher
            try:
                with open(launcher_path, encoding="utf-8") as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                    for i, line in enumerate(first_lines, 1):
                        if line and not line.startswith("#"):
                            print(f"   📝 Line {i}: {line[:60]}...")
                            break
            except Exception as e:
                print(f"   ❌ Could not read file: {e}")
            print()
        else:
            print(f"❌ {launcher} - NOT FOUND")

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


if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
