"""
Script to create a desktop shortcut for the Tools Launcher.

This script detects the user's Desktop path (standard or OneDrive),
locates the Python executable (preferring pythonw.exe to avoid console windows),
and creates a Windows shortcut (.lnk) pointing to the PyQt6 UnifiedToolsLauncher.
"""
import os
import sys

import win32com.client


def create_shortcut():
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")

    # If standard Desktop doesn't exist, check OneDrive
    if not os.path.exists(desktop):
        # OneDrive case
        desktop = os.path.join(os.environ["USERPROFILE"], "OneDrive", "Desktop")

    # If the OneDrive Desktop also does not exist, fail early with a clear error
    if not os.path.exists(desktop):
        raise FileNotFoundError(
            f"Could not find a Desktop folder under USERPROFILE. "
            f"Tried: "
            f"{os.path.join(os.environ['USERPROFILE'], 'Desktop')} and "
            f"{os.path.join(os.environ['USERPROFILE'], 'OneDrive', 'Desktop')}."
        )

    path = os.path.join(desktop, "Tools Launcher.lnk")
    # Point to the Qt6 launcher
    target = os.path.join(os.getcwd(), "UnifiedToolsLauncher.py")
    w_dir = os.getcwd()
    icon = os.path.join(os.getcwd(), "tools_icon_alt.ico")

    # Check if python/pythonw is in path or use sys.executable
    # Usually we want pythonw to avoid console
    # Using sys.executable is safest for environment, but we want the 'w' version if possible.
    python_exe = sys.executable
    if python_exe.endswith("python.exe"):
        pythonw = python_exe.replace("python.exe", "pythonw.exe")
        if os.path.exists(pythonw):
            python_exe = pythonw

    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(path)
    shortcut.TargetPath = python_exe
    shortcut.Arguments = f'"{target}"'
    shortcut.WorkingDirectory = w_dir
    shortcut.IconLocation = icon
    shortcut.Description = "Launch Professional Tools Launcher (Antigravity Unified)"
    shortcut.Save()
    print(f"Shortcut created at {path}")

if __name__ == "__main__":
    try:
        create_shortcut()
    except Exception as e:
        print(f"Failed to create shortcut: {e}")
