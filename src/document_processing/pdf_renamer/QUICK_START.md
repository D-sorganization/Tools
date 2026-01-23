# Quick Start Guide

## ✅ Installation Fixed!

All dependencies have been installed successfully, including PyMuPDF which was causing errors.

## 🚀 Launch the Application

### Option 1: Desktop Shortcut (Easiest)
1. Run `create_shortcut.ps1` or `create_shortcut.vbs`
2. Double-click "PDF Renamer" icon on your desktop
3. Done!

### Option 2: Command Line
```bash
# GUI Mode (Recommended)
python launch_gui.py

# CLI Mode
python -m src.pdf_renamer.cli /path/to/pdfs --dry-run
```

### Option 3: Batch File
```bash
# Double-click or run:
PDF_Renamer.bat
```

## ✨ Verified Installation

All core dependencies are now installed:
- ✅ Python 3.13
- ✅ PyPDF (metadata extraction)
- ✅ **PyMuPDF (layout analysis)** - FIXED!
- ✅ PyQt6 (GUI framework)
- ✅ Google Generative AI (optional, for AI features)
- ✅ pdfplumber (optional, enhanced extraction)

## 🎯 Using the Application

### GUI Features
1. **Browse** - Select folder with PDFs
2. **Settings** - Configure naming style:
   - ☑ Include Author (for "Author - Title.pdf" format)
   - Choose: Standard / Snake Case / Kebab Case
3. **Options**:
   - Dry Run (preview only)
   - Delete Duplicates
   - Use AI (Gemini)
   - Include Subfolders
4. **Start Processing** - Watch real-time progress!

### Naming Styles

| Style | Without Author | With Author |
|-------|---------------|-------------|
| **Standard** | `Machine Learning Basics.pdf` | `Smith - Machine Learning Basics.pdf` |
| **Snake Case** | `machine_learning_basics.pdf` | `smith_machine_learning_basics.pdf` |
| **Kebab Case** | `machine-learning-basics.pdf` | `smith-machine-learning-basics.pdf` |

### Title Case Rules
- Major words: Capitalized (Machine, Learning, Analysis)
- Minor words: Lowercase (of, the, and, for, to, at, by, in)
- First/Last: Always capitalized

**Examples:**
```
introduction to machine learning → Introduction to Machine Learning
the lord of the rings → The Lord of the Rings
a tale of two cities → A Tale of Two Cities
```

## 🔧 Verify Installation Anytime

```bash
python verify_installation.py
```

This will check all dependencies and show what's working.

## 💡 Tips

1. **Always start with Dry Run** - Preview changes before actual renaming
2. **Use AI for difficult PDFs** - Enable "Use AI (Gemini)" for best results
   - Requires GEMINI_API_KEY environment variable
   - Get free key at: https://makersuite.google.com/app/apikey
3. **Parallel Workers** - Adjust based on your CPU (default: 4)
4. **Transaction Logs** - Automatic rollback support via `pdf_renamer_transactions.jsonl`

## 🐛 Troubleshooting

### PyMuPDF Errors (NOW FIXED!)
Previously caused errors - now resolved by installing `pymupdf` package.

### GUI Won't Start
```bash
# Check PyQt6
python -c "from PyQt6.QtWidgets import QApplication"

# Reinstall if needed
pip install --force-reinstall PyQt6
```

### Missing Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Or individually
pip install pypdf pymupdf PyQt6 google-generativeai pdfplumber
```

### AI Features Not Working
```bash
# Set API key (Windows PowerShell)
$env:GEMINI_API_KEY="your_api_key_here"

# Or permanently in System Environment Variables
```

## 📁 Project Structure

```
PDFRenamer/
├── launch_gui.py          ← Launch GUI
├── PDF_Renamer.bat        ← Windows launcher
├── verify_installation.py ← Check dependencies
├── requirements.txt       ← All dependencies
├── src/pdf_renamer/
│   ├── gui.py            ← PyQt6 interface
│   ├── worker.py         ← Thread-safe processing
│   ├── core.py           ← Extraction pipeline
│   ├── extractors.py     ← Layer 0-2 extraction
│   ├── deduper.py        ← SHA256 duplicate detection
│   ├── transaction_log.py ← Rollback support
│   └── utils.py          ← Helper functions
└── README.md             ← Full documentation
```

## 🎓 Next Steps

1. **Create Desktop Shortcut** - See [CREATE_DESKTOP_SHORTCUT.md](CREATE_DESKTOP_SHORTCUT.md)
2. **Read Full Docs** - See [README.md](README.md)
3. **Test with Sample PDFs** - Use dry-run mode first!
4. **Enable AI Features** - Set GEMINI_API_KEY for best results

## 🆘 Need Help?

- **Full Documentation**: [README.md](README.md)
- **Feature Details**: [AUTHOR_TITLE_FEATURE.md](AUTHOR_TITLE_FEATURE.md)
- **Implementation Notes**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

**Status**: ✅ **READY TO USE!**

All dependencies installed, PyMuPDF errors resolved, application is fully functional.
