# Fleet-Wide Updates - January 31, 2026

## Summary

Major infrastructure and tooling updates across all 9 repositories in the fleet.

## Changes Applied to All Repos

### 1. Workflow Schedule Standardization

- All automated workflows now run on `0 8 */3 * *` (midnight PST, every 3 days)
- Reduces CI costs and eliminates daytime automated PR spam

### 2. Claude Skills Added

Two new Claude skills in `.claude/skills/`:

- **`/lint`**: Runs ruff, black, mypy and finds/fixes placeholder statements
- **`/tests`**: Runs full test suite and iterates to fix failures

### 3. CI/CD Verification

- All repos have `ci-standard.yml` with mypy enabled

## Repository-Specific Changes

### UpstreamDrift

- Added LOD generation for URDF meshes
- Added click-to-highlight and side-by-side comparison
- Renamed package from `golf-suite` to `upstream-drift`
- Fixed ruff linting issues

### Tools

- Updated `pyproject.toml` to export all shared packages as `ud-tools`
- Updated branding/logo to golfer/UD arc design

### Gasification_Model

- Added Data Processor module with headless API and PyQt6 UI

### Games, MLProjects

- Fixed ruff linting issues

## Package Structure

```bash
pip install ud-tools[urdf]     # URDF generation
pip install ud-tools[signal]   # Signal processing
pip install ud-tools[all]      # Everything
```
