# P&ID Generator

Generate P&ID drawings from YAML specifications (DXF + SVG output)

## Current Features

- Purpose: Programmatic P&ID (Piping & Instrumentation Diagram) generation
- Category: Engineering Drafting
- Engine: `programmatic_pid` shared library (in `src/shared/python/programmatic_pid/`)
- Surface support: PyQt6=implemented, CLI=implemented, Web=planned
- Test visibility: 1 name-matched test file under tests/

## Quick Start

### PyQt6 GUI

```bash
python src/pid_generator/launch_pyqt6.py
```

### CLI

```bash
generate-pid spec.yml -o output.dxf
```

## Implementation State

- PyQt6 launcher: Implemented (file-picker GUI)
- CLI launcher: Implemented (via `programmatic_pid.cli`)
- Web surface: Not yet implemented
- README last reviewed: 2026-03-11

## Dependencies

- `ezdxf[draw] >= 1.4.3` (install with `pip install -e '.[pid]'`)
- `PyYAML >= 6.0` (shared dependency)

## Implementation Gaps

- Web surface not yet implemented.
