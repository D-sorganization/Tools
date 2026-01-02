# PDF Renamer Setup and Usage Guide

This guide will help you set up and use the advanced PDF Renamer tool, which features a 3-layer extraction system (Metadata -> Heuristic -> AI) to accurately rename your PDF files.

## Prerequisites

1.  **Python 3.10+**: Ensure Python is installed and added to your PATH.
    *   Verify with: `python --version`
2.  **Gemini API Key**: Required if you want to use the AI fallback layer for difficult PDFs.
    *   Get a key here: [Google AI Studio](https://aistudio.google.com/app/apikey)

## Installation

1.  **Navigate to the project directory**:
    The tool is located in `Repositories\Playground\PDFRenamer`.

2.  **Install Dependencies**:
    Run the following command in your terminal to install the required Python packages (`pypdf`, `pymupdf`, `google-generativeai`, etc.).

    ```powershell
    pip install -r requirements.txt
    ```

## Configuration

### Setting the Gemini API Key (Optional but Recommended)

For the AI layer to work, you need to set the `GEMINI_API_KEY` environment variable.

**PowerShell:**
```powershell
$env:GEMINI_API_KEY="your_actual_api_key_here"
```

**Command Prompt (cmd):**
```cmd
set GEMINI_API_KEY=your_actual_api_key_here
```

**Bash:**
```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

## Detailed Usage

The tool is run via the command line. You must run it as a module from the `src` directory.

### Basic Syntax

```powershell
python -m pdf_renamer.main [DIRECTORY] [OPTIONS]
```

### Common Scenarios

**1. Dry Run (Preview changes without renaming)**
This is always the safest first step. It will print what *would* happen.
```powershell
python -m src.pdf_renamer.main "C:\Path\To\My\Papers" --dry-run
```

**2. Standard Rename with AI Fallback**
This will process files, using metadata where possible, heuristics for others, and Gemini for the hardest ones.
```powershell
python -m src.pdf_renamer.main "C:\Path\To\My\Papers" --provider gemini
```

**3. Using a Specific Naming Style**
Available styles: `standard` (default), `snake_case`, `kebab_case`.
```powershell
python -m src.pdf_renamer.main "C:\Path\To\My\Papers" --style snake_case
```

**4. Concurrent Processing**
Speed up processing by using more workers (default is 4).
```powershell
python -m src.pdf_renamer.main "C:\Path\To\My\Papers" --workers 8
```

## Troubleshooting

*   **`ModuleNotFoundError`**: Ensure you are running the command from the repository root (parent of `src`) and that you have installed requirements.
*   **AI Errors**: Check your internet connection and ensure your API key is correct. The tool will fall back to local heuristics if the AI fails.
*   **Database Locking**: The tool uses SQLite (`pdf_titles.sqlite`). If you run multiple instances, you might see locking errors.
