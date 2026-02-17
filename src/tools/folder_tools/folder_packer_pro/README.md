# 📦 Folder Packer Pro v2.0

**Professional Project Packaging Tool - Enhanced Edition**

A comprehensive, modern project packing application designed to securely package and transport development projects. This is a complete professional rebuild of the original Folder Packer tool with encryption, compression options, git integration, and enterprise-grade security.

## 🌟 What's New in v2.0 (Enhanced Pro Edition)

### Major Enhancements Over v1.1

#### **1. Advanced Security**

- **AES-256 Encryption**: Military-grade encryption for sensitive projects
- **Password Protection**: Secure packages with strong passwords
- **PBKDF2 Key Derivation**: 100,000 iterations for password hashing
- **Encrypted Storage**: Files encrypted before writing to disk

#### **2. Flexible Compression**

- **Multiple Levels**: None, Fast, Balanced, Best
- **Gzip Compression**: Industry-standard compression
- **Smart Selection**: Choose speed vs. size trade-off
- **Compression Stats**: See exact space savings

#### **3. Modern Professional UI**

- **Tabbed Interface**: Pack, Unpack, Preview, and Log tabs
- **Dark/Light Themes**: Professional color schemes
- **Real-time Preview**: See files before packing
- **Syntax Highlighting**: Basic code syntax detection
- **Progress Tracking**: Detailed progress with file counts

#### **4. Git Integration**

- **Preserve .git Folder**: Include repository history
- **Smart Exclusions**: Automatically exclude build artifacts
- **Repository Packaging**: Transport complete git repos
- **Version Control Ready**: Maintain all git metadata

#### **5. Smart File Management**

- **Advanced Exclusions**: Customizable exclusion patterns
- **File Type Detection**: Automatic categorization
- **Preview System**: Tree view with file details
- **Content Preview**: View file contents before packing

#### **6. Professional Package Format**

- **Custom .fpp Format**: Folder Packer Package format
- **JSON-based Structure**: Human-readable metadata
- **Manifest Files**: Optional detailed file lists
- **Integrity Checking**: Verify packages after creation

#### **7. Enhanced User Experience**

- **Batch Operations**: Pack multiple projects
- **Drag-and-Drop Support**: Easy file selection
- **Operation Logs**: Color-coded detailed logging
- **Export Capabilities**: Save logs and manifests
- **Error Recovery**: Robust error handling

## 🎯 Key Features

### Core Functionality

✅ **Pack/Unpack**: Complete folder packaging system
✅ **AES-256 Encryption**: Military-grade security
✅ **Gzip Compression**: Multiple compression levels
✅ **Git Integration**: Preserve repository structure
✅ **Smart Filtering**: Exclude build artifacts automatically
✅ **Manifest Generation**: Optional file catalogs
✅ **Package Verification**: Integrity checking

### Professional Features

✅ **File Preview**: Tree view with metadata
✅ **Syntax Detection**: Basic code highlighting
✅ **Progress Tracking**: Real-time updates
✅ **Operation Logs**: Comprehensive logging
✅ **Theme Support**: Dark and light modes
✅ **Export Reports**: Manifests and logs
✅ **Package Inspection**: View package contents

### Security Features

✅ **Password Protection**: Strong password support
✅ **Key Derivation**: PBKDF2 with 100K iterations
✅ **Encryption**: AES-256 symmetric encryption
✅ **Secure Storage**: No plaintext on disk
✅ **Password Confirmation**: Prevent typos
✅ **Encrypted Packages**: Complete file encryption

## 📋 System Requirements

- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: 3.9 or higher (for source)
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Disk Space**: 100MB for installation

### Python Dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

**Core Dependencies**:

- `cryptography>=41.0.0` - AES-256 encryption
- `pillow>=10.0.0` - Icon processing (optional)

**Development Dependencies**:

- `pytest>=7.4.0` - Testing
- `black>=23.7.0` - Code formatting
- `ruff>=0.0.285` - Linting
- `mypy>=1.5.0` - Type checking

## 🚀 Quick Start

### Running from Source

```bash
# Navigate to the tool directory
cd python/folder_packer_pro

# Install dependencies
pip install -r requirements.txt

# Run the application
python folder_packer_pro.py
```

### Building Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Run build script
python build_exe.py

# Executable will be in dist/FolderPackerPro.exe
```

## 📖 Usage Guide

### Packing a Project

1. **Select Source Folder**

   - Click "Browse" in the Source Folder section
   - Choose the project folder to pack
   - Or type the path directly

2. **Choose Output Location**

   - Click "Browse" in Output Package File section
   - Select where to save the .fpp package
   - File extension will be added automatically

3. **Configure Compression**

   - **None**: Fastest, largest size
   - **Fast**: Quick compression
   - **Balanced**: Recommended for most cases
   - **Best**: Slowest, smallest size

4. **Enable Encryption (Optional)**

   - Check "Enable AES-256 Encryption"
   - Enter a strong password
   - Confirm password
   - Remember it - there's no recovery!

5. **Advanced Options**

   - **Include .git folder**: For repositories
   - **Create manifest file**: Generate file catalog
   - **Verify package**: Check integrity after creation

6. **Create Package**
   - Click "📦 Create Package"
   - Monitor progress in Progress section
   - Review results in Log tab

### Unpacking a Package

1. **Select Package File**

   - Click "Browse" in Package File section
   - Choose the .fpp package to extract
   - Verify file exists

2. **Choose Destination**

   - Click "Browse" in Destination Folder section
   - Select where to extract files
   - Folder will be created if needed

3. **Decryption (If Encrypted)**

   - Check "Package is encrypted"
   - Enter the decryption password
   - Must match the original password

4. **Inspect Package (Optional)**

   - Click "🔍 Inspect Package"
   - View package metadata
   - Check encryption status
   - See file count

5. **Extract Package**
   - Click "📂 Extract Package"
   - Monitor progress
   - Verify extraction successful

### Preview Files

1. **Navigate to Preview Tab**

   - View files that will be packed
   - See file sizes, types, and dates
   - Browse folder structure

2. **Select File**

   - Click on any file in the tree
   - View content in preview pane
   - See basic syntax highlighting

3. **File Information**
   - **Size**: Human-readable file size
   - **Type**: Detected file category
   - **Modified**: Last modification date

## 🔐 Security Guide

### Encryption Best Practices

**Strong Passwords**:

- Use 12+ characters
- Mix uppercase, lowercase, numbers, symbols
- Avoid dictionary words
- Don't reuse passwords

**Password Management**:

- Store passwords in a password manager
- Never write passwords in plain text
- Don't email passwords
- Consider password generation tools

**Encryption Use Cases**:

- Source code with trade secrets
- Projects with API keys or credentials
- Sensitive business logic
- Proprietary algorithms
- Client projects under NDA

### Compression Guide

**Choosing Compression Level**:

**None (Store)**:

- Use when: Speed is critical
- Files are already compressed (images, videos)
- Package will be encrypted (encryption takes time)

**Fast**:

- Use when: Quick packaging needed
- Moderate size reduction acceptable
- Good for testing

**Balanced (Recommended)**:

- Use when: Best overall performance
- Good compression ratio
- Reasonable speed
- Most common use case

**Best**:

- Use when: Minimum size is critical
- Network transfer costs are high
- Storage space is limited
- Time is not a constraint

## 📊 Package Format

### .fpp File Structure

```
package.fpp (encrypted or compressed JSON)
├── metadata
│   ├── created_at
│   ├── source
│   ├── total_files
│   ├── compression
│   └── encrypted
└── files
    ├── file1.py (base64 encoded)
    ├── file2.js (base64 encoded)
    └── ... (more files)
```

### Manifest Format (Optional)

```json
{
  "package_file": "path/to/package.fpp",
  "created_at": "2024-01-15T10:30:00",
  "files": ["src/main.py", "src/utils.py", "README.md"],
  "total_files": 3,
  "package_size": 1024000
}
```

## 🎨 Features in Detail

### File Type Detection

**Code Files**:

- Python, JavaScript, TypeScript, Java, C++, C, Go, Rust, Ruby, PHP, Swift, Kotlin, R, MATLAB

**Markup Files**:

- HTML, XML, CSS, SCSS, SASS, Vue, JSX, TSX

**Configuration Files**:

- JSON, YAML, TOML, INI, CFG

**Documents**:

- PDF, DOC, DOCX, TXT, MD, RST

**Media Files**:

- Images: JPG, PNG, GIF, BMP, SVG
- Audio: MP3, WAV, FLAC, OGG, M4A
- Video: MP4, AVI, MKV, MOV, WMV

### Smart Exclusions

**Automatically Excluded**:

- `__pycache__` - Python bytecode cache
- `.git` (unless explicitly included)
- `.svn`, `.hg` - Version control
- `node_modules` - Node.js dependencies
- `.venv`, `venv`, `env` - Virtual environments
- `.pytest_cache`, `.mypy_cache` - Testing/linting caches
- `build`, `dist`, `*.egg-info` - Build artifacts
- `.DS_Store`, `Thumbs.db` - OS files
- `*.pyc`, `*.pyo`, `*.pyd` - Compiled Python
- `.coverage`, `htmlcov` - Coverage reports

**Customizable**:

- Manage exclusions via **Tools → Manage Exclusions**
- Add patterns for your specific needs
- Remove patterns you want to include
- Reset to defaults anytime

### Syntax Highlighting

**Basic Detection**:

- Keywords: `def`, `class`, `if`, `for`, etc.
- Strings: Text in quotes
- Comments: Lines starting with `#`
- Numbers: Numeric literals
- Functions: Detected function names

**Supported Languages**:

- Python (primary support)
- Basic support for other code files
- Markup language detection
- Configuration file detection

## 💡 Tips & Best Practices

### Packing Tips

**Before Packing**:

- Clean build artifacts first
- Remove unnecessary files
- Check folder size
- Verify git status (if including .git)
- Test exclusion patterns

**During Packing**:

- Use "Balanced" compression for most cases
- Enable encryption for sensitive code
- Create manifests for documentation
- Monitor progress for large projects
- Check logs for any warnings

**After Packing**:

- Verify package was created
- Check package size is reasonable
- Test unpacking in temp directory
- Store password securely if encrypted
- Keep manifest file with package

### Unpacking Tips

**Before Unpacking**:

- Inspect package first
- Verify destination is empty or safe
- Have password ready if encrypted
- Check available disk space

**During Unpacking**:

- Monitor progress
- Watch for errors in log
- Don't cancel unless necessary

**After Unpacking**:

- Verify all files extracted
- Check file integrity
- Test project functionality
- Delete package if no longer needed

### Security Tips

**Passwords**:

- Never share passwords via email
- Use password managers
- Don't write passwords down
- Create unique passwords per package
- Test password before distributing package

**Sensitive Data**:

- Always encrypt packages with credentials
- Review files before packing
- Remove API keys and secrets if possible
- Use environment variables instead
- Consider separate package for secrets

## 🆚 Comparison with v1.1

| Feature             | v1.1             | v2.0 Pro                           |
| ------------------- | ---------------- | ---------------------------------- |
| User Interface      | Basic tabs       | Professional tabbed UI             |
| Encryption          | None             | AES-256 encryption                 |
| Compression         | None             | 4 levels (none/fast/balanced/best) |
| Git Support         | Manual inclusion | Dedicated option                   |
| File Preview        | No               | Tree view + content preview        |
| Syntax Highlighting | No               | Basic code highlighting            |
| Themes              | Basic            | Dark + Light professional          |
| Package Format      | Basic            | JSON-based with metadata           |
| Manifests           | No               | Optional detailed manifests        |
| Progress Tracking   | Basic            | Detailed with file counts          |
| Exclusions          | Fixed            | Customizable patterns              |
| Package Inspection  | No               | Full inspection before unpack      |
| Error Handling      | Basic            | Comprehensive with recovery        |

## 🐛 Troubleshooting

**Packing Issues**:

_"Password Mismatch"_

- Ensure Password and Confirm fields match exactly
- Check for extra spaces
- Verify Caps Lock is off

_"Package creation failed"_

- Check destination is writable
- Verify sufficient disk space
- Review error in Log tab
- Try disabling encryption as test

**Unpacking Issues**:

_"Decryption failed"_

- Verify password is correct
- Check package isn't corrupted
- Ensure package was actually encrypted
- Try package inspection first

_"Package format error"_

- Package may be corrupted
- Try re-downloading if from network
- Check file size matches expected
- Inspect package to verify format

**Performance Issues**:

_"Packing is very slow"_

- Use "Fast" compression instead of "Best"
- Check for large binary files
- Disable encryption for testing
- Close other applications

_"Preview is slow"_

- Large projects take time to scan
- Limit preview to relevant files
- Skip preview for very large projects
- Use filters to reduce file count

## 📝 License

This software is provided as-is for use in the Tools repository. All rights reserved.

## 🤝 Contributing

This is an enhanced professional edition built for the Tools repository. Contributions and improvements are welcome through the standard repository workflow.

## 📞 Support

For issues, questions, or feature requests:

1. Check this README and usage guide
2. Review the Log tab for error messages
3. Inspect packages before unpacking
4. Consult the repository documentation

## 🎯 Roadmap

Future enhancements under consideration:

- Multi-part packages for large projects
- Incremental packing (only changes)
- Package signing and verification
- Cloud storage integration
- Command-line interface
- Diff viewer for packages
- Automated backup scheduling
- Package compression statistics
- Password strength meter
- Two-factor authentication

## ✨ Credits

**Folder Packer Pro v2.0** - Enhanced Professional Edition
Built as a competitor to the original Folder Packer v1.1
Designed for secure, professional project packaging

---

**Version**: 2.0.0
**Release Date**: 2024
**Status**: Production Ready
**Platform**: Cross-platform (Windows, Linux, macOS)

---

_Making project packaging secure, efficient, and professional._ 📦🔐✨
