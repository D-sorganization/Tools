# API Key Setup Guide

## Overview

The PDF Renamer supports AI-powered title extraction using Google's Gemini API. This guide explains how to securely configure your API key.

## Security Features

✅ **Multiple secure storage locations**
✅ **Automatic .gitignore protection**
✅ **Never committed to version control**
✅ **Shared across Tools and Playground folders**
✅ **Environment variable support**

---

## Quick Setup

### Option 1: Interactive Setup (Easiest)

```bash
python setup_api_key.py
```

This will:

1. Check if you already have an API key configured
2. Guide you through getting a key from Google
3. Ask where you want to save it
4. Automatically configure it for you

### Option 2: Manual Setup

Create a `.env` file in one of these locations:

**Option A: Project Folder (Playground)**

```bash
# Create file: Playground/PDFRenamer/.env
GEMINI_API_KEY=your_actual_api_key_here
```

**Option B: Tools Folder (Shared)**

```bash
# Create file: Tools/document_processing/pdf_renamer/.env
GEMINI_API_KEY=your_actual_api_key_here
```

**Option C: User Home (Global)**

```bash
# Create file: ~/.pdf_renamer/.env
GEMINI_API_KEY=your_actual_api_key_here
```

### Option 3: Environment Variable

**Windows (PowerShell) - Temporary**

```powershell
$env:GEMINI_API_KEY="your_actual_api_key_here"
python launch_gui.py
```

**Windows (PowerShell) - Permanent**

```powershell
[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your_actual_api_key_here', 'User')
```

**Windows (Command Prompt) - Temporary**

```cmd
set GEMINI_API_KEY=your_actual_api_key_here
python launch_gui.py
```

**Linux/Mac - Temporary**

```bash
export GEMINI_API_KEY="your_actual_api_key_here"
python launch_gui.py
```

**Linux/Mac - Permanent** (add to ~/.bashrc or ~/.zshrc)

```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

---

## Getting Your API Key

1. Go to **Google AI Studio**: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (starts with "AIza...")
5. Use it in setup

**Note**: The free tier is generous and suitable for personal use!

---

## How It Works

### API Key Priority

The system checks for your API key in this order:

1. **Environment variable** (current session)
2. **Project .env file** (Playground/PDFRenamer/.env)
3. **Tools .env file** (Tools/document_processing/pdf_renamer/.env)
4. **User home .env file** (~/.pdf_renamer/.env)

This means:

- Set it once, use it everywhere
- Environment variables override .env files
- Tools and Playground can share the same key

### Automatic Loading

The `config.py` module automatically:

- Searches all locations on startup
- Loads the first key it finds
- Makes it available to the LLM layer
- Never exposes it in logs or errors

---

## Security Best Practices

### ✅ DO:

- Store API keys in `.env` files (auto-gitignored)
- Use environment variables for temporary testing
- Keep API keys in secure locations
- Use the Tools folder .env to share across projects
- Rotate keys periodically

### ❌ DON'T:

- Commit `.env` files to git (already in .gitignore)
- Share API keys in screenshots or logs
- Hardcode keys in source files
- Email or message keys in plain text
- Use production keys for testing

---

## Verification

### Check if API Key is Configured

```bash
python -c "from src.pdf_renamer.config import get_api_key; print('✓ Found' if get_api_key() else '✗ Not found')"
```

### Find Where Key is Located

```bash
python -c "from src.pdf_renamer.config import _find_key_location; print(_find_key_location())"
```

### Run Full Verification

```bash
python verify_installation.py
```

Look for the "Environment" section showing API key status.

---

## Sharing Between Tools and Playground

### Recommended Setup: Use Tools Folder

Since you have both Tools and Playground versions:

1. **Create .env in Tools folder:**

   ```
   c:\Users\diete\Repositories\Tools\document_processing\pdf_renamer\.env
   ```

2. **Add your key:**

   ```
   GEMINI_API_KEY=your_actual_key_here
   ```

3. **Both versions automatically find it!**
   - Playground checks Tools folder as fallback
   - No duplication needed
   - Update once, works everywhere

### Verify Sharing Works

```bash
# In Playground folder
cd c:\Users\diete\Repositories\Playground\PDFRenamer
python -c "from src.pdf_renamer.config import get_api_key, _find_key_location; print(f'Key found: {bool(get_api_key())}'); print(f'Location: {_find_key_location()}')"
```

Should show: `Location: Tools folder`

---

## Troubleshooting

### API Key Not Found

**Symptom**: Warning message about missing API key

**Solutions**:

1. Run `python setup_api_key.py`
2. Check file exists: `dir .env` or `ls -la .env`
3. Check file content (no quotes needed):
   ```
   GEMINI_API_KEY=AIza...your_key
   ```
4. Verify permissions: file should be readable

### API Key Invalid

**Symptom**: LLM extraction fails with authentication error

**Solutions**:

1. Verify key at: https://makersuite.google.com/app/apikey
2. Check for extra spaces/quotes in .env file
3. Regenerate key if compromised
4. Check API quota/limits

### GUI Shows "AI Not Available"

**Solutions**:

1. Install: `pip install google-generativeai`
2. Check API key is configured
3. Test connection: `python -c "import google.generativeai as genai; print('OK')"`

### Permission Denied on .env File

**Solutions**:

1. Check file permissions
2. Try saving to user home instead: `~/.pdf_renamer/.env`
3. Run as administrator (Windows) or with sudo (Linux/Mac)

---

## Migration from Tools to Playground

If you already have a `.env` file in the Tools folder, the Playground version will automatically find and use it. No migration needed!

### Optional: Copy to Playground

If you want a separate key for Playground:

```bash
# Windows PowerShell
Copy-Item "c:\Users\diete\Repositories\Tools\document_processing\pdf_renamer\.env" "c:\Users\diete\Repositories\Playground\PDFRenamer\.env"

# Linux/Mac
cp ~/Repositories/Tools/document_processing/pdf_renamer/.env ~/Repositories/Playground/PDFRenamer/.env
```

---

## Advanced: Multiple API Keys

You can use different keys for different environments:

### Development Key (Playground)

```
Playground/PDFRenamer/.env:
GEMINI_API_KEY=development_key_here
```

### Production Key (Tools)

```
Tools/document_processing/pdf_renamer/.env:
GEMINI_API_KEY=production_key_here
```

Since Playground checks its own .env first, it will use the development key. Tools will use the production key.

---

## Environment Files (.env) Format

### Basic Format

```bash
# Comments start with #
GEMINI_API_KEY=your_key_here

# No quotes needed (but they work)
GEMINI_API_KEY="your_key_here"
GEMINI_API_KEY='your_key_here'

# Spaces around = are OK
GEMINI_API_KEY = your_key_here
```

### Multiple Variables

```bash
# You can add other settings too
GEMINI_API_KEY=your_key
DEFAULT_STYLE=snake_case
MAX_WORKERS=8
DRY_RUN=true
```

### Example .env File

```bash
# PDF Renamer Configuration
# Created: 2026-01-02

# Gemini API Key for AI-powered extraction
GEMINI_API_KEY=AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Default settings
# DEFAULT_STYLE=standard
# WORKERS=4
# DRY_RUN=false
```

---

## Summary

**Best Practice**: Create `.env` file in Tools folder and both versions will automatically use it.

**Quick Commands**:

```bash
# Setup interactively
python setup_api_key.py

# Verify configuration
python verify_installation.py

# Test API key works
python -c "from src.pdf_renamer.config import get_api_key; print('✓ Configured' if get_api_key() else '✗ Missing')"
```

Your API key is now secure, shared, and ready to use! 🔐
