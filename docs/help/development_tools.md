# Development Tools

The Development Tools category contains utilities for project management, file organization, and automation tasks.

## Folder Packer Pro

Project archiving and distribution tool for creating clean, shareable project packages.

### Features

- **Selective Packaging**: Choose specific files and folders
- **Exclusion Patterns**: Automatically exclude unwanted files
- **Multiple Formats**: ZIP, TAR, TAR.GZ, 7z
- **Integrity Verification**: Checksums for validation
- **Size Estimation**: Preview package size before creating

### Exclusion Presets

Pre-configured patterns for common development scenarios:

#### Python Projects

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv/
env/
.eggs/
*.egg-info/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/
```

#### Node.js Projects

```
node_modules/
.npm/
dist/
build/
.next/
.nuxt/
coverage/
```

#### IDE and Editor Files

```
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store
Thumbs.db
```

#### Version Control

```
.git/
.gitignore
.svn/
.hg/
```

### Usage

1. **Select Source**: Choose the folder to package
2. **Configure Exclusions**: Select presets or add custom patterns
3. **Preview**: Review files that will be included
4. **Set Output**: Choose format and destination
5. **Create Package**: Generate the archive

### Custom Exclusion Patterns

Add custom patterns using glob syntax:

| Pattern | Matches |
|---------|---------|
| `*.log` | All .log files |
| `temp/*` | Everything in temp folder |
| `**/*.bak` | All .bak files recursively |
| `data/[0-9]*` | Data folders starting with numbers |

### Output Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| ZIP | .zip | Universal compatibility |
| TAR | .tar | Unix standard, no compression |
| TAR.GZ | .tar.gz | Good compression |
| 7z | .7z | Best compression |

---

## Folder Fix Pro

Automated folder structure cleanup and organization tool.

### Features

- **Empty Folder Detection**: Find and remove empty directories
- **Duplicate Detection**: Identify files with identical content
- **Structure Validation**: Compare against templates
- **Batch Renaming**: Apply naming patterns
- **Dry-Run Mode**: Preview changes before applying
- **Undo Support**: Revert changes if needed

### Operations

#### Empty Folder Cleanup

Recursively find and optionally remove empty folders:

1. Scan directory tree
2. Identify empty folders
3. Preview list of empty folders
4. Apply removal (with confirmation)

**Options**:
- Keep folders matching patterns (e.g., `.gitkeep`)
- Only remove if all children are also empty

#### Duplicate File Detection

Find files with identical content using hash comparison:

1. Scan files and compute hashes
2. Group files by hash
3. Review duplicate groups
4. Choose which copies to keep/delete

**Hash Methods**:
- MD5 (fast, less secure)
- SHA256 (recommended)
- SHA512 (most secure)

#### Structure Validation

Compare folder structure against a template:

1. Define expected structure (JSON or YAML)
2. Scan actual folder
3. Report differences:
   - Missing folders
   - Extra folders
   - Missing files
   - Extra files

**Template Example**:

```yaml
structure:
  src:
    - __init__.py
    - main.py
    modules:
      - module1.py
      - module2.py
  tests:
    - test_main.py
  docs:
    - README.md
```

#### Batch Renaming

Apply consistent naming patterns to files:

**Pattern Types**:
| Pattern | Description | Example |
|---------|-------------|---------|
| Prefix | Add text before name | `project_` + `file.py` |
| Suffix | Add text before extension | `file` + `_v2` + `.py` |
| Replace | Substitute text | `old` -> `new` |
| Case | Change capitalization | `File.py` -> `file.py` |
| Sequence | Add numbers | `file_001.py` |
| Date | Add timestamp | `file_20240215.py` |

### Safety Features

#### Dry-Run Mode

Preview all changes without modifying files:

```
[DRY RUN] Would delete: empty_folder/
[DRY RUN] Would rename: old_name.py -> new_name.py
[DRY RUN] Would delete duplicate: file_copy.txt
```

#### Undo Support

All operations create a transaction log:

1. Operations recorded to log file
2. Each change includes original state
3. Undo command reverses changes
4. Log expires after configurable period

---

## PDF Renamer

Batch rename PDF files based on content or metadata.

### Extraction Methods

#### From Metadata

Extract information from PDF metadata:

| Field | Description |
|-------|-------------|
| Title | Document title |
| Author | Document author |
| Subject | Document subject |
| Keywords | Document keywords |
| Created | Creation date |
| Modified | Last modification date |

#### From Content

Extract information from PDF text:

| Method | Description |
|--------|-------------|
| First Heading | First large text block |
| First Line | First line of text |
| Pattern Match | Regex pattern extraction |
| Page Number | Specific page text |

### Naming Patterns

Build filenames from extracted data:

```
{title} - {author} ({year})
{date}_{subject}
{author}_{title}
```

**Pattern Variables**:
- `{title}` - Document title
- `{author}` - Document author
- `{date}` - Creation date (YYYYMMDD)
- `{year}` - Creation year
- `{subject}` - Document subject
- `{pages}` - Page count
- `{original}` - Original filename

### Collision Handling

When renamed file would overwrite existing:

| Option | Behavior |
|--------|----------|
| Skip | Keep original, don't rename |
| Append Number | Add `_1`, `_2`, etc. |
| Append Date | Add timestamp |
| Overwrite | Replace existing (careful!) |

### Transaction Log

All renames are logged for potential undo:

```json
{
  "timestamp": "2024-02-15T10:30:00",
  "original": "scan001.pdf",
  "renamed": "Invoice_2024_Acme_Corp.pdf",
  "path": "/documents/invoices/"
}
```

### API Mode

Use PDF Renamer programmatically:

```python
from pdf_renamer import rename_pdfs

results = rename_pdfs(
    source_dir="/path/to/pdfs",
    pattern="{title}_{date}",
    dry_run=True
)
```

### Tips

- **Test with Dry Run**: Always preview before applying
- **Backup First**: Keep copies of originals
- **Text PDFs Work Best**: Scanned images need OCR first
- **Check Encoding**: Some PDFs have encoding issues

---

## Common Workflows

### Preparing a Project for Distribution

1. Use **Folder Packer Pro** to create archive
2. Select Python preset exclusions
3. Add any project-specific exclusions
4. Review file list
5. Create ZIP package

### Cleaning Up a Messy Project

1. Use **Folder Fix Pro** in dry-run mode
2. Remove empty folders
3. Find and handle duplicates
4. Apply consistent naming
5. Validate against template

### Organizing Document Collections

1. Use **PDF Renamer** to standardize names
2. Extract dates and titles from metadata
3. Apply consistent naming pattern
4. Review and approve changes
5. Generate rename log for records

---

## Tips for Developers

### Version Control Integration

- Run cleanup before commits
- Exclude tool artifacts from git
- Keep transaction logs in separate location

### Automation

- Create batch scripts for common operations
- Schedule regular cleanup tasks
- Integrate with CI/CD pipelines

### Best Practices

1. **Always dry-run first**: Preview changes before applying
2. **Keep backups**: Especially for irreversible operations
3. **Document patterns**: Record custom patterns used
4. **Review logs**: Check transaction logs periodically

---

For more detailed documentation, see the [User Manual](../USER_MANUAL.md).
