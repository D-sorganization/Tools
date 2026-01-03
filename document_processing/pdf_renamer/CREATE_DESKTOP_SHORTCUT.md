# Create Desktop Shortcut - Quick Start

## ⚡ Fastest Method (Choose One)

### Method 1: Enhanced Shortcut Creator (Recommended) ⭐

1. **Double-click** `CREATE_SHORTCUT.bat` in this folder
   - Automatically creates a professional shortcut with custom icon
   - Includes detailed setup instructions and troubleshooting
   - Works with the existing tools icon from the main project

**OR** run the PowerShell script directly:
1. **Right-click** on `create_pdf_renamer_shortcut.ps1` in this folder
2. Select **"Run with PowerShell"**
3. If prompted about execution policy, type `Y` and press Enter
4. Done! Look for "PDF Renamer" icon on your desktop

### Method 2: Basic PowerShell Script

1. **Right-click** on `create_shortcut.ps1` in this folder
2. Select **"Run with PowerShell"**
3. If prompted about execution policy, type `Y` and press Enter
4. Done! Look for "PDF Renamer" icon on your desktop

### Method 3: VBScript

1. **Double-click** `create_shortcut.vbs` in this folder
2. Click **"OK"** when the success message appears
3. Done! Look for "PDF Renamer" icon on your desktop

### Method 4: Manual (If scripts don't work)

1. **Right-click** on `PDF_Renamer.bat` in this folder
2. Select **"Send to"** → **"Desktop (create shortcut)"**
3. **Rename** the shortcut on your desktop to: `PDF Renamer`
4. Done!

---

## 🚀 Using the Shortcut

Once created, simply **double-click** the "PDF Renamer" icon on your desktop to launch the application.

The shortcut will:
- ✅ Check for Python installation
- ✅ Auto-install missing dependencies if needed
- ✅ Launch the PDF Renamer GUI
- ✅ Display helpful error messages if issues occur

---

## ⚠️ Troubleshooting

### PowerShell: "Execution policy" error

**Fix:**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Then run the script again.

### Python not found

**Fix:**
1. Install Python from: https://www.python.org/downloads/
2. During installation, check **"Add Python to PATH"**
3. Restart your computer
4. Run the shortcut script again

### Still having issues?

See [SHORTCUT_INSTRUCTIONS.md](SHORTCUT_INSTRUCTIONS.md) for detailed troubleshooting and alternative methods.

---

## 📌 Pinning to Taskbar

After creating the desktop shortcut:

1. **Double-click** the shortcut to launch PDF Renamer
2. While it's running, **right-click** the taskbar icon
3. Select **"Pin to taskbar"**

Now you can launch it directly from your taskbar!

---

## 🎨 Icon Features

The enhanced shortcut creator automatically applies:
- ✅ **Custom Tools Icon**: Uses the project's professional tools icon
- ✅ **Taskbar Support**: Icon appears properly in taskbar when pinned
- ✅ **High Quality**: Uses the best available icon resolution
- ✅ **Fallback**: Uses system document icon if custom icon unavailable

### Manual Icon Customization (Optional)

If you want to change the icon:
1. **Right-click** the desktop shortcut
2. Select **"Properties"**
3. Click **"Change Icon..."**
4. Browse to choose a different icon file
5. Click **"OK"** twice

---

## 🗑️ Removing the Shortcut

Just **delete** the desktop shortcut - it won't affect the application files.

To recreate it later, run the script again!
