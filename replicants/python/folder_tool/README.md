# Folder Fix - Enhanced Folder Processor Tool

## Overview
Folder Fix is a comprehensive Windows application for advanced folder processing tasks including file combining, organization, deduplication, and archive extraction.

## Features
- **Multiple Processing Modes:**
  - Combine & Copy files from multiple sources
  - Flatten & Tidy deeply nested folder structures
  - Copy & Prune empty folders
  - Deduplicate files (in-place)
  - Analyze & Report only (no changes)

- **Advanced Options:**
  - File filtering by extension and size
  - Organization by file type or date
  - Bulk archive extraction (.zip, .rar, .7z)
  - Preview mode for safe testing
  - Automatic backup creation
  - ZIP output generation

- **Windows Integration:**
  - Custom paper plane icon in window and taskbar
  - Modern Windows visual styles
  - DPI awareness for high-resolution displays
  - Proper taskbar grouping

## Build Information
- **Executable:** `FolderFix.exe`
- **Original Source:** `Folders_Tool_r0.py`
- **Icon:** Paper plane design with ICO format for crisp display
- **Framework:** Python + Tkinter GUI
- **Packaging:** PyInstaller with custom manifest

## Usage
1. Run `FolderFix.exe` or use `Launch_FolderFix.bat`
2. Select source folder(s) to process
3. Choose destination folder (if applicable)
4. Configure filtering and organization options
5. Select processing mode
6. Run the operation

## Files Included
- `FolderFix.exe` - Main executable
- `Launch_FolderFix.bat` - Quick launch script
- `paper_plane_icon.ico` - Application icon
- `FolderFix.exe.manifest` - Windows integration manifest

## Build Files (Development)
- `Folders_Tool_r0.py` - Source code
- `FolderFix.spec` - PyInstaller configuration
- `requirements.txt` - Python dependencies
- `paper_plane_icon.png` - Original icon source

## Building from Source
1. Install dependencies: `pip install -r requirements.txt`
2. Build executable: `pyinstaller FolderFix.spec --clean`
3. Find output in `dist/FolderFix.exe`

## Version
- **Version:** 2.0
- **Build Date:** January 2025
- **Compatibility:** Windows 10/11

## Support
This tool provides comprehensive logging and error handling. Check the generated log files for detailed operation information.
