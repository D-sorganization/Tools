# Launcher Guide

This repository has one supported entry point for its graphical tools. This
document records that entry point, how individual tools are reached, and what to
do when the launcher fails.

## Canonical entry point

**`UnifiedToolsLauncher.py`** — PyQt6 desktop launcher, at the repository root.

```bash
python UnifiedToolsLauncher.py
```

- **Requirement**: Python 3.11 or newer, with PyQt6 installed
  (`pip install -e ".[gui]"` or `".[dev]"`).
- **Role**: presents every graphical application grouped by domain, validates
  each tool path before launching, and captures the child process's output and
  errors into an activity log.
- **Diagnostics**: pass `--verbose` for detailed logging. A session log is
  written to `unified_launcher.log` in the working directory.

The launcher discovers tools through the plugin system described in
[Plugin system](PLUGIN_SYSTEM.md), so a correctly registered tool appears
without changes to the launcher itself.

## Individual tool launchers

Some tools also carry their own entry point for direct use, for example
`src/web_applications/urdf_viewer/main.py`. These exist to support development
and embedding; the launcher remains the supported path for normal use.

A subset of tools is installed as console scripts by
`pip install -e .`. Those are listed in the
[repository README](../../README.md#command-line-entry-points) and defined under
`[project.scripts]` in `pyproject.toml`.

## Unsupported entry points

Names that appear in older documentation, commit messages, or issue history do
not resolve to files in this repository and are not supported:

- `tools_launcher.py`
- `launch_tools_main.py`

Neither exists. There is no Tkinter fallback launcher. If PyQt6 cannot be
installed in the target environment, use the command-line entry points or the
browser-based utilities under `src/web_applications/` instead.

## When the launcher fails

See [Troubleshooting](../help/troubleshooting.md#the-launcher-will-not-start)
for the ordered checks.
