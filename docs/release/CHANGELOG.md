# Changelog

All notable changes to the Tools repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive assessment framework (A-O) with 15 quality categories
- Executive summary (Highlight) assessment template
- Top-level READMEs for all major project areas (data_processing, scientific_modeling, web_applications, media_processing)

### Changed

- README: Fixed title from "Golf Biomechanics" to "Tools Monorepo"
- README: Clarified primary launcher entry point
- Documentation: Reorganized CI/CD reports into docs/ci-cd/archive/
- Documentation: Consolidated redundant status reports

### Fixed

- pytest.ini: Resolved 17 test collection errors
- UnifiedToolsLauncher.py: Removed shell=True security vulnerability
- Launcher.py: Removed shell=True security vulnerability
- docs/index.md: Fixed broken links to project READMEs

### Security

- Replaced shell=True subprocess calls with explicit command lists
- Mitigated CWE-78 command injection risk

## [1.1.0] - 2026-01-21

### Added

- CI/CD pipeline improvements with comprehensive quality gates
- Workflow automation scripts for GitHub Actions
- Priority issue resolution system

### Fixed

- YAML indentation issues in CI/CD scripts
- Workflow automation reliability improvements

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
- Pre-commit hooks for quality enforcement

## [0.1.0] - 2025-08-09

### Added

- Initial project structure
- Cursor guardrails and safety scripts
- Pre-commit hooks setup
- Git LFS configuration
- CI pipeline stubs
- Test framework stubs
