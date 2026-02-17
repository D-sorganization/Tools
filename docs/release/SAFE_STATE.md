# Safe State Record

This document tracks known stable states of the repository that have passed all quality checks.

## Current Safe State

- **Date**: 2026-01-21
- **Version**: 1.1.0
- **Branch**: main
- **CI Status**: Passing
- **Notes**:
  - All CI/CD quality gates passing
  - Documentation reorganized and consolidated
  - Top-level READMEs added for all major project areas

## Quality Verification

```bash
# Verify safe state
ruff check .              # Linting passes
black --check .           # Formatting passes
mypy --config-file pyproject.toml .  # Type checking passes
pytest                    # Tests pass
```

## Previous Safe States

### 2025-12-25 (v1.0.0)

- **Notes**: Initial stable release
- **CI Status**: Passing
- **Key Features**: UnifiedToolsLauncher, all core tools operational

### 2025-08-09 (v0.1.0)

- **Notes**: Initial project setup
- **CI Status**: Passing (stubs only)
- **Key Features**: Added rule banning hard resets/clean on tracked dirs

## Recovery Instructions

If the repository is in an unstable state, you can recover to a safe state:

```bash
# Find the latest safe tag
git tag -l "v*" --sort=-v:refname | head -5

# Reset to a safe state (use with caution)
git checkout v1.1.0  # or appropriate version
```

## Guidelines

1. **Before marking safe**: Ensure all CI checks pass
2. **Documentation**: Update this file when establishing a new safe state
3. **Tagging**: Consider creating a git tag for significant safe states
4. **Communication**: Notify team when safe state changes
