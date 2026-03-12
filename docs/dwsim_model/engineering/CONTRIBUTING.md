# Contributing to DWSIM_Model

This guide explains how to set up a development environment, run the test suite,
and contribute to the DWSIM gasification train model.

## Repository Overview

`DWSIM_Model` is a Python package (`dwsim-model`) that models a combined gasification
train using DWSIM Automation3 (Windows) or pure-Python fallbacks (all platforms).

The package lives under `src/dwsim_model/` and is pip-installable:

```bash
pip install -e "."
```

## Prerequisites

### All Platforms (pure-Python suite)

- Python 3.11 or newer
- `pip` or a virtual environment manager (`venv`, `uv`, etc.)

### Windows Only (DWSIM runtime)

- [DWSIM](https://dwsim.org) 8.x installed to the default path
  (`C:\Users\<user>\AppData\Local\DWSIM8\`)
- `pythonnet >= 3.0.0` (installed automatically via requirements)

DWSIM integration tests are skipped automatically on non-Windows platforms or when
the DWSIM runtime is not found. The pure-Python unit tests run on all platforms.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/D-sorganization/DWSIM_Model.git
cd DWSIM_Model
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install the package and dev dependencies

```bash
pip install -r requirements.txt
pip install -e "."
```

Optional extras:

- `pip install mypy types-PyYAML` — for type checking
- `pip install ruff black` — for linting and formatting

### 4. Verify the installation

```bash
python -c "import dwsim_model; print(dwsim_model.__version__)"
python -m dwsim_model --help
```

## Running the Tests

### Pure-Python unit tests (all platforms)

These tests run without a DWSIM installation and cover configuration loading,
topology construction, metrics calculation, and the public API contracts:

```bash
pytest tests/ -v -m "not integration and not dwsim_runtime"
```

### Full test suite (Windows with DWSIM installed)

```bash
pytest tests/ -v
```

Integration tests that need the DWSIM runtime are automatically skipped when
`pythonnet` cannot find `DWSIM.Automation3.dll`.

### Test markers

| Marker          | Description                  |
| --------------- | ---------------------------- |
| `unit`          | Fast, no external deps       |
| `integration`   | Requires DWSIM runtime       |
| `dwsim_runtime` | Needs live DWSIM COM objects |

## Code Standards

This project follows the same quality bar as the sibling repositories:

- **Linting**: `ruff check .` (zero errors required)
- **Formatting**: `black .`
- **Type checking**: `mypy src/`
- **Testing**: `pytest` with >80% coverage on new code
- **Style**: see `AGENTS.md` for full conventions

### Pre-commit (optional but recommended)

```bash
pip install pre-commit
pre-commit install
```

## Relationship to Other Repositories

### Tools repository (D-sorganization/Tools)

`DWSIM_Model` is designed to be usable as a standalone pip package. The Tools repo
may install it as an optional dependency via:

```bash
pip install -e "path/to/DWSIM_Model"
```

No `sys.path` manipulation is needed. There are no cross-repo imports at module level.

### Gasification_Model repository (D-sorganization/Gasification_Model)

The Gasification_Model IPS integrates `DWSIM_Model` as a git submodule under
`src/dwsim/`. After cloning with `--recurse-submodules`, install it with:

```bash
pip install -e "src/dwsim"
```

The dependency direction is strictly one-way:

```
Gasification_Model (IPS) --> dwsim-model
Tools (optional)         --> dwsim-model
```

`dwsim_model` never imports from either IPS or Tools.

## Making a Pull Request

1. Branch from `main`: `git checkout -b feat/your-feature`
2. Write tests first (TDD) — all new code needs test coverage
3. Run the linting and test checks locally before pushing
4. Open a PR against `main`
5. CI must be green before merging

## Reporting Issues

Open a GitHub issue with:

- Python version and OS
- DWSIM version (if relevant)
- Minimal reproduction steps
- Expected vs. actual behaviour
