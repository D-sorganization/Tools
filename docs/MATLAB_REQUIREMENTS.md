# MATLAB Requirements & Tool Availability

This repository contains several tools written in MATLAB. To run these tools, a local MATLAB installation is required.

## Requirements

- **MATLAB Version:** R2020a or later is recommended.
- **Toolboxes:** Specific tools may require additional toolboxes (e.g., Signal Processing, Statistics).

## Launching MATLAB Tools

When you launch a MATLAB tool via `UnifiedToolsLauncher.py`, the system attempts to:

1.  Run the tool using the `matlab` command-line interface.
2.  If `matlab` is not in your PATH, it will attempt to open the `.m` file in your default editor/viewer.

## Configuration

Ensure `matlab` is in your system PATH.

### Windows
Add the MATLAB `bin` directory (e.g., `C:\Program Files\MATLAB\R2023b\bin`) to your system PATH environment variable.

### Linux / macOS
Ensure `matlab` command works in your terminal. You may need to create a symlink:
```bash
sudo ln -s /usr/local/MATLAB/R2023b/bin/matlab /usr/local/bin/matlab
```

## Troubleshooting

If MATLAB tools fail to launch:
1.  Verify `matlab` command runs in your terminal.
2.  Check the `unified_launcher.log` for error messages.
3.  Open the `.m` file manually in MATLAB.
