# Changelog

All notable changes to the Tools repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive assessment framework (A-O) with 15 quality categories
- Executive summary (Highlight) assessment template
- `src/shared/python/deprecation.py`: shared `@deprecated` decorator for marking
  functions scheduled for removal; emits `DeprecationWarning` with optional
  `reason` and `removal_version` parameters

### Changed

- `toolstrip_widget.py`: decomposed `_build_action_buttons_group` (101 LOC) into `_build_model_selector` (28 LOC) and `_build_sim_controls` (60 LOC) helpers; extracted `_build_playback_group` (33 LOC) from `_build_row1`, reducing the orchestrating function to 8 LOC (partially closes #2011 P1 oversized functions)
- README: Fixed title from "Golf Biomechanics" to "Tools Monorepo"
- README: Clarified primary launcher entry point
- `pyproject.toml`: bumped package version from `0.3.0` to `1.0.0` to align
  with the semantic-versioning baseline established in CHANGELOG [1.0.0]
- `src/shared/python/programmatic_pid/__init__.py`: bumped `__version__` from
  `0.3.0` to `1.0.0` to match package-level version

### Fixed

- pytest.ini: Resolved 17 test collection errors
- UnifiedToolsLauncher.py: Removed shell=True security vulnerability
- Launcher.py: Removed shell=True security vulnerability

### Security

- Replaced shell=True subprocess calls with explicit command lists
- Mitigated CWE-78 command injection risk

## [1.0.0] - 2025-12-25

### Added

- Initial release of Tools Monorepo
- UnifiedToolsLauncher.py - PyQt6-based centralized launcher
- Data Processor Integrated for CSV/Parquet analysis
- Solar System Model interactive simulation
- Folder Packer Pro for project archiving
- Audio Processor Pro (MATLAB)
- Video Processor Web Platform (Next.js)
- Calculator web application (Flask)
- Unit Converter (browser-based)

### Infrastructure

- pytest configuration for multi-project testing
- Ruff and Black for code formatting
- MyPy for type checking
