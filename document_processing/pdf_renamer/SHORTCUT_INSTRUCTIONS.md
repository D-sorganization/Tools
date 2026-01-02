# Desktop Shortcut Instructions

## Quick Method (Automated)

### Windows

**Option 1: Run the VBScript (Recommended)**

1. Double-click `create_shortcut.vbs` in this folder
2. Click "OK" when prompted
3. A shortcut named "PDF Renamer" will appear on your desktop

**Option 2: Use the Batch File Directly**

1. Right-click `PDF_Renamer.bat`
2. Select "Send to" → "Desktop (create shortcut)"
3. Rename the shortcut to "PDF Renamer"

## Manual Method

### Windows

1. Right-click on your desktop
2. Select "New" → "Shortcut"
3. Click "Browse..." and navigate to:
   ```
   c:\Users\diete\Repositories\Playground\PDFRenamer\PDF_Renamer.bat
   ```
4. Click "Next"
5. Name it: `PDF Renamer`
6. Click "Finish"
7. (Optional) Right-click the shortcut → "Properties" → "Change Icon"
   - Browse to: `C:\Windows\System32\shell32.dll`
   - Select an icon (icon #71 is PDF-like)

### Linux/Mac

Create a desktop entry or symbolic link:

**Linux (.desktop file):**
```bash
cat > ~/Desktop/pdf-renamer.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=PDF Renamer
Comment=AI-Powered PDF Renaming Tool
Exec=/usr/bin/python3 /c/Users/diete/Repositories/Playground/PDFRenamer/launch_gui.py
Icon=application-pdf
Terminal=false
Categories=Utility;FileTools;
EOF

chmod +x ~/Desktop/pdf-renamer.desktop
```

**Mac (Automator Application):**
1. Open Automator
2. Create new "Application"
3. Add "Run Shell Script" action
4. Script: `cd /Users/diete/Repositories/Playground/PDFRenamer && python3 launch_gui.py`
5. Save as "PDF Renamer" in Applications folder
6. Drag to Dock

## What the Shortcut Does

When you double-click the shortcut:

1. ✅ Checks if Python is installed
2. ✅ Checks if required dependencies (PyQt6, etc.) are installed
3. ✅ Auto-installs missing dependencies if needed
4. ✅ Launches the PDF Renamer GUI
5. ✅ Shows error messages if something goes wrong

## Troubleshooting

### "Python is not installed or not in PATH"

**Solution:**
1. Install Python 3.11+ from https://www.python.org/
2. During installation, check "Add Python to PATH"
3. Restart your computer
4. Try the shortcut again

### "Failed to install dependencies"

**Solution:**
```bash
cd c:\Users\diete\Repositories\Playground\PDFRenamer
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Shortcut icon doesn't look right

**Solution:**
1. Right-click shortcut → Properties
2. Click "Change Icon"
3. Browse to `C:\Windows\System32\shell32.dll`
4. Choose icon #71 (document icon) or any you prefer
5. Click OK

### GUI doesn't start

**Check PyQt6 installation:**
```bash
python -c "from PyQt6.QtWidgets import QApplication"
```

If error, reinstall:
```bash
pip uninstall PyQt6
pip install PyQt6
```

## Advanced: Pinning to Taskbar/Start Menu

### Windows 10/11

**Taskbar:**
1. Double-click shortcut to launch the app
2. Right-click the app icon in taskbar while running
3. Select "Pin to taskbar"

**Start Menu:**
1. Right-click the desktop shortcut
2. Select "Pin to Start"

### Windows 11 Start Menu Alternative

1. Press Win+R
2. Type: `shell:programs`
3. Copy `PDF_Renamer.bat` to this folder
4. It will appear in Start Menu → All Apps

## Customization

### Change Icon

Download a custom icon (.ico file) and:

1. Right-click shortcut → Properties
2. Click "Change Icon"
3. Browse to your .ico file
4. Click OK

### Recommended Icon Sources
- https://icon-icons.com (search "PDF")
- https://icons8.com (free for personal use)
- Create your own with https://www.favicon-generator.org/

### Change Shortcut Name

Simply right-click → Rename

## Uninstalling the Shortcut

Just delete the desktop shortcut - it won't affect the application.
