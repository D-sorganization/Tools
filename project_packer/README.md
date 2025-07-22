# Folder Packer / Unpacker Tool

A modern GUI tool for packing multiple programming files from a folder structure into a single text file, preserving the folder hierarchy. It can then unpack this file to recreate the original folder structure.

## Features

- **Pack folders** into a single text file with preserved structure
- **Unpack** text files back to original folder structure
- **Smart exclusions** for common archive/backup folders
- **Modern GUI** with clean, intuitive interface
- **File type filtering** for programming files
- **Batch operations** with progress tracking

## Supported File Types

`.py`, `.html`, `.css`, `.js`, `.m`, `.txt`, `.json`, `.xml`, `.cpp`, `.java`, `.c`, `.h`, `.hpp`, `.php`, `.rb`, `.go`, `.rs`, `.swift`, `.kt`, `.r`, `.sql`, `.sh`, `.bat`, `.yml`, `.yaml`, `.md`, `.rst`, `.tex`, `.vue`, `.jsx`, `.tsx`, `.ts`

## Usage

### Running from Source
```bash
python folder_packer_gui.py
```

### Building Executable

1. **Quick Build**: Double-click `build.bat`

2. **Manual Build**:
   ```bash
   python build_exe.py
   ```

3. **Clean Build Files**:
   ```bash
   python build_exe.py clean
   ```

### Requirements
- Python 3.7+
- tkinter (usually included with Python)
- Pillow (for icon support)
- PyInstaller (for building executable)

Install dependencies:
```bash
pip install -r requirements.txt
```

## How It Works

### Packing
1. Select a source folder
2. Choose exclusion patterns (optional)
3. Select output location for the packed file
4. The tool creates a single text file with all your code

### Unpacking
1. Select a packed text file
2. Choose destination folder
3. The tool recreates the original folder structure

### File Format
The packed file uses clear delimiters:
```
%%%%%% START FILE: path/to/file.py %%%%%%
[file content]
%%%%%% END FILE: path/to/file.py %%%%%%
```

## Interface Tabs

- **Pack / Unpack**: Main operations
- **Folder Exclusions**: Configure which folders to skip
- **Instructions**: Help and usage guide

## Automatic Exclusions

The tool automatically excludes common archive/backup patterns:
- `*_archive`, `*_backup`, `*_old`
- `.git`, `__pycache__`, `node_modules`
- `build`, `dist`, `venv`, `.env`

## Building the Executable

The build process:
1. Converts the JPG icon to ICO format
2. Uses PyInstaller to create a single executable
3. Includes all dependencies
4. Creates a windowed application (no console)

### Build Output
- **Executable**: `dist/FolderPacker.exe`
- **Build files**: `build/` (can be deleted)
- **Spec file**: `FolderPacker.spec` (PyInstaller configuration)

## Troubleshooting

### Build Issues
- Ensure Python and pip are in your PATH
- Run `pip install pyinstaller pillow` manually if the build script fails
- Check that all source files are present

### Runtime Issues
- The executable is self-contained but may trigger antivirus warnings
- If the GUI doesn't appear, try running from command line to see error messages

## Version History

- **v1.1**: Added modern GUI, exclusion management, and executable building
- **v1.0**: Initial command-line version

## License

Open source - feel free to modify and distribute.
