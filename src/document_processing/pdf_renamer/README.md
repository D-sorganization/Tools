# PDF Renamer Pro - Professional AI-Powered Document Management

A professional-grade tool to intelligently rename PDF files using a layered extraction approach with AI fallback, featuring a modern PyQt6 GUI, parallel processing, comprehensive duplicate detection, and advanced workflow management.

## 🚀 New Professional Features

### 🤖 API-Only Mode

- **Manual Review Workflow**: Generate AI-powered rename proposals without automatic execution
- **Approval System**: Review, approve, or reject each rename individually
- **Custom Names**: Override AI suggestions with custom filenames
- **CSV Export**: Export proposals for external review and documentation
- **Batch Execution**: Execute only approved renames with full control

### 📁 Smart Failed File Management

- **Automatic Segregation**: Move files that couldn't be renamed to a dedicated subfolder
- **Configurable Folder**: Customize the name of the failed files folder
- **Clean Organization**: Keep successfully renamed files separate from problematic ones
- **Transaction Logging**: Full audit trail of all file movements

### 💾 User Preferences & Memory

- **Last Folder Memory**: Automatically remembers and defaults to the last used directory
- **Persistent Settings**: Save and restore user preferences between sessions
- **Professional Defaults**: Intelligent default settings for enterprise use
- **Cross-Session Continuity**: Seamless workflow across application restarts

### 🎨 Enhanced Professional Interface

- **Tabbed Interface**: Separate tabs for Batch Processing, API Mode, and Settings
- **Modern Design**: Professional styling with icons and improved visual hierarchy
- **Real-time Status**: Enhanced progress tracking and status indicators
- **Comprehensive Logging**: Color-coded execution logs with detailed information

## Features

### 🔄 Dual Processing Modes

#### Batch Processing Mode

- **Automated Workflow**: Traditional batch processing with full automation
- **Dry-run Preview**: Safe preview mode before making actual changes
- **Parallel Processing**: Multi-threaded processing for large document sets
- **Real-time Progress**: Live progress tracking and detailed logging

#### API-Only Mode (New!)

- **Manual Review**: Generate AI proposals without automatic execution
- **Approval Workflow**: Review and approve each rename individually
- **Custom Override**: Modify AI suggestions with custom names
- **Export Capability**: Export proposals to CSV for external review
- **Controlled Execution**: Execute only approved renames

### Core Capabilities

- **Multi-Layer Title Extraction**

  - **Layer 0**: PDF metadata extraction (fast, free)
  - **Layer 1**: Layout-aware heuristic analysis using PyMuPDF (robust, intelligent)
  - **Layer 2**: AI fallback using Google Gemini (highest accuracy for difficult PDFs)

- **Advanced Duplicate Detection**

  - SHA256-based content hashing (cryptographically secure)
  - Size-based pre-filtering for performance
  - Automatic or manual duplicate deletion
  - Recursive subfolder scanning

- **Smart File Processing**

  - Intelligent caching to avoid re-processing files
  - Thread-safe parallel processing with configurable workers
  - Transaction logging for rollback capability
  - Multiple naming styles (standard, snake_case, kebab-case)
  - **Failed File Management**: Automatic segregation of problematic files
  - **User Preferences**: Persistent settings and folder memory

- **Production-Ready Reliability**
  - Thread-safe file operations with locking
  - Windows reserved filename protection
  - Unicode normalization and sanitization
  - Path length validation
  - Comprehensive error handling
  - **Audit Trail**: Complete transaction logging for compliance

### User Interface

- **Modern PyQt6 GUI**

  - **Tabbed Interface**: Separate modes for different workflows
  - **Batch Processing Tab**: Traditional automated processing
  - **API Mode Tab**: Manual review and approval workflow
  - **Settings Tab**: User preferences and API configuration
  - Real-time progress tracking
  - Color-coded log output
  - Dry-run preview mode
  - Cancellable operations
  - **Folder Memory**: Remembers last used directories
  - **Professional Styling**: Modern design with icons and improved UX

- **CLI Interface**
  - Full command-line support
  - Scriptable workflows
  - Batch processing

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: AI Features

To enable AI-powered title extraction (Layer 2), set up your Gemini API key:

```bash
# Linux/Mac
export GEMINI_API_KEY="your_api_key_here"

# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"

# Windows Command Prompt
set GEMINI_API_KEY=your_api_key_here
```

Get your free API key at: https://makersuite.google.com/app/apikey

## Usage

### 🖥️ GUI Mode (Recommended)

Launch the professional interface:

```bash
python launch_gui.py
```

#### Batch Processing Tab

- **Traditional Workflow**: Automated batch processing with full control
- Browse and select target directory (remembers last location)
- Configure processing options and naming styles
- Enable/disable AI features and duplicate handling
- **Failed File Management**: Automatically move problematic files to subfolder
- Real-time progress tracking and detailed logging
- Dry-run preview before making actual changes

#### API Mode Tab (New!)

- **Professional Workflow**: Manual review and approval process
- Generate AI-powered rename proposals without automatic execution
- Review each proposal with confidence scores
- Approve, reject, or modify individual suggestions
- Export proposals to CSV for documentation
- Execute only approved renames with full audit trail

#### Settings Tab

- Configure user preferences and defaults
- Set up and test Gemini API key securely
- Customize failed file folder names
- Enable persistent settings across sessions

### 📋 API-Only Workflow Example

1. **Generate Proposals**:

   - Select directory in API Mode tab
   - Choose naming style and options
   - Click "Generate Proposals" to create AI suggestions
   - Review proposals with confidence scores

2. **Review and Approve**:

   - Examine each proposed rename
   - Approve good suggestions with ✅ button
   - Reject problematic ones with ❌ button
   - Modify names directly in the table if needed

3. **Export and Execute**:
   - Export proposals to CSV for documentation
   - Execute only approved renames
   - Full transaction logging for audit compliance

### CLI Mode

For scripting and automation:

```bash
# Basic dry run (preview only)
python -m src.pdf_renamer.cli /path/to/pdfs --dry-run

# Process with AI fallback
python -m src.pdf_renamer.cli /path/to/pdfs --provider gemini

# Snake case naming with 8 workers
python -m src.pdf_renamer.cli /path/to/pdfs --style snake_case --workers 8

# Custom cache database
python -m src.pdf_renamer.cli /path/to/pdfs --db my_titles.sqlite

# Full example: recursive, delete duplicates, AI enabled
python -m src.pdf_renamer.cli /path/to/pdfs \
  --provider gemini \
  --style kebab_case \
  --workers 4 \
  --dry-run
```

### 📁 Failed File Management

Files that cannot be processed are automatically moved to a dedicated subfolder:

```
/your/pdf/directory/
├── Successfully Renamed File 1.pdf
├── Successfully Renamed File 2.pdf
└── failed_renames/
    ├── problematic_file_1.pdf
    ├── encrypted_document.pdf
    └── corrupted_file.pdf
```

**Benefits:**

- Clean separation of successful vs. problematic files
- Easy identification of files needing manual attention
- Configurable folder name (default: "failed_renames")
- Full transaction logging for audit trails

### 💾 User Preferences

The application now remembers your preferences:

```json
{
  "last_directory": "/path/to/your/pdfs",
  "default_style": "standard",
  "default_workers": 4,
  "remember_settings": true,
  "create_failed_folder": true,
  "failed_folder_name": "failed_renames"
}
```

**Stored securely in**: `~/.pdf_renamer/preferences.json`

## Architecture

### Extraction Layers

```
┌─────────────────────────────────────┐
│  Layer 0: Metadata Extraction       │
│  - Fast, deterministic              │
│  - Confidence: 0.95 if valid        │
└──────────┬──────────────────────────┘
           │ Failed or low confidence
           ↓
┌─────────────────────────────────────┐
│  Layer 1: Heuristic Analysis        │
│  - Layout-aware (font size, pos)    │
│  - Confidence: 0.7-0.9              │
└──────────┬──────────────────────────┘
           │ Failed or conf < 0.7
           ↓
┌─────────────────────────────────────┐
│  Layer 2: AI (Gemini)               │
│  - Native PDF understanding         │
│  - Confidence: 0.0-1.0              │
└─────────────────────────────────────┘
```

### Thread-Safe Processing

- File operations protected by global lock (prevents TOCTOU races)
- Parallel PDF reading and extraction
- Atomic rename operations
- Transaction logging for audit trail

### Caching Strategy

All extraction results are cached by SHA256 hash:

- Avoids redundant API calls
- Persistent across runs
- Survives file renames

## Security Features

### Robust Hashing

- Uses SHA256 instead of MD5 (cryptographically secure)
- Prevents collision-based duplicate detection failures

### Filename Sanitization

- Removes invalid characters
- Handles Windows reserved names (CON, PRN, AUX, etc.)
- Unicode normalization (NFC)
- Path length validation
- Control character filtering

### Safe Operations

- Symlink detection to prevent infinite loops
- Permission checking before operations
- Dry-run mode for safe testing
- Transaction logs for rollback

## Configuration

### Environment Variables

| Variable         | Description                           | Default           |
| ---------------- | ------------------------------------- | ----------------- |
| `GEMINI_API_KEY` | Google Gemini API key for AI features | None (local only) |

### Database

Results are cached in `pdf_titles.sqlite` (configurable):

```sql
CREATE TABLE results (
    sha256 TEXT PRIMARY KEY,
    file_path TEXT,
    title TEXT,
    confidence REAL,
    method TEXT,
    provider TEXT,
    model TEXT,
    timestamp DATETIME,
    error TEXT
);
```

### Transaction Logs

All operations are logged to `pdf_renamer_transactions.jsonl`:

```json
{
  "session_id": "20260102_143022",
  "timestamp": "2026-01-02T14:30:22.123456",
  "operation": "rename",
  "original_path": "/path/to/old.pdf",
  "new_path": "/path/to/New Title.pdf",
  "success": true,
  "error": ""
}
```

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'PyQt6'"**

```bash
pip install PyQt6
```

**2. "GEMINI_API_KEY not found"**

- Set environment variable (see Installation)
- Or use local extraction only (no AI)

**3. "Permission denied" errors**

- Check file/folder permissions
- Run with appropriate user privileges
- Avoid system directories

**4. GUI doesn't start**

```bash
# Check Qt installation
python -c "from PyQt6.QtWidgets import QApplication"

# Try CLI mode instead
python -m src.pdf_renamer.cli /path/to/pdfs --dry-run
```

### Performance Tuning

**Optimize Worker Count:**

- CPU-bound: `workers = CPU_count`
- I/O-bound: `workers = CPU_count * 2`
- Default: 4 workers

**Large File Sets:**

- Process in batches
- Increase chunk size in hash functions
- Use SSD for cache database

## Development

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src/pdf_renamer --cov-report=html

# Specific test file
pytest tests/test_utils.py -v
```

### Code Quality

```bash
# Type checking
mypy src/

# Linting
ruff check src/

# Formatting
black src/
```

## Limitations

- AI features require internet connection and API key
- Very large PDFs (>100MB) may be slow to process
- OCR not supported (text must be extractable)
- Encrypted PDFs require password (not supported)

## License

This project is provided as-is for educational and professional use.

## Credits

- **PyMuPDF**: Fast PDF parsing and layout analysis
- **Google Gemini**: AI-powered title extraction
- **PyQt6**: Modern cross-platform GUI framework

## Version History

### v2.0 (2026-01-02) - Hybrid Release

- Complete rewrite with best features from two codebases
- Added PyQt6 GUI
- Implemented SHA256 hashing for security
- Added transaction logging for rollback
- Enhanced filename sanitization
- Thread-safe parallel processing
- Comprehensive error handling
- Production-ready reliability

### v1.0 (Previous)

- Initial release
- Basic metadata and heuristic extraction
- CLI interface only

### Naming Styles

Choose from three naming conventions:

1. **Standard** (default): `Title Case Here.pdf`
2. **Snake Case**: `title_case_here.pdf`
3. **Kebab Case**: `title-case-here.pdf`

## 🔒 Security Features

### API Key Management

- **Secure Storage**: API keys stored in `.env` files, never in code
- **Multiple Locations**: Supports project, user, and global configurations
- **Environment Priority**: Environment variables take precedence
- **Automatic Detection**: Finds keys in multiple standard locations
- **Interactive Setup**: Guided API key configuration with validation

### File Operation Safety

- **Transaction Logging**: Complete audit trail of all operations
- **Atomic Operations**: Thread-safe file operations prevent corruption
- **Collision Handling**: Smart duplicate name resolution
- **Permission Checking**: Validates file access before operations
- **Rollback Capability**: Transaction logs enable operation reversal
