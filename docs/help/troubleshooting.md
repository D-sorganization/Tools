# Troubleshooting

Common failures when installing, launching, or testing the tools in this
repository, with the checks that resolve them.

## Contents

- [Python version errors](#python-version-errors)
- [The launcher will not start](#the-launcher-will-not-start)
- [MATLAB tools do nothing](#matlab-tools-do-nothing)
- [Tests fail to collect](#tests-fail-to-collect)
- [Rust extensions are not used](#rust-extensions-are-not-used)

## Python version errors

**Symptom**

```text
ImportError: cannot import name 'StrEnum' from 'enum'
ImportError: cannot import name 'UTC' from 'datetime'
```

**Cause**

The interpreter is older than Python 3.11. Both symbols were added in 3.11 and
the installable package requires it.

**Resolution**

Install Python 3.11 or 3.12 and recreate the virtual environment.

```bash
# Ubuntu and Debian
sudo apt update && sudo apt install python3.12

# macOS with Homebrew
brew install python@3.12
```

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Some legacy helper modules still carry compatibility shims for older
interpreters. Those shims do not make the package itself importable below 3.11.

## The launcher will not start

**Symptom**

`python UnifiedToolsLauncher.py` exits immediately or raises on import.

**Checks, in order**

1. Confirm the interpreter: `python --version` must report 3.11 or newer.
2. Confirm the environment is active and the package is installed:
   `python -m pip install -e ".[dev]"`.
3. Confirm the GUI dependency is present: `python -c "import PyQt6"`. If it
   fails, install it with `pip install "PyQt6>=6.6.0"`.
4. Re-run with diagnostics: `python UnifiedToolsLauncher.py --verbose`, and
   check `unified_launcher.log` in the working directory.

On Windows, a `WinError 127` from a PyQt6 import usually means the installed
Microsoft Visual C++ runtime is older than the wheel requires. Install the
current Visual C++ Redistributable, or pin an earlier PyQt6 release.

## MATLAB tools do nothing

**Symptom**

MATLAB-backed tools fail silently, or the launcher opens the `.m` file in a text
editor instead of running it.

**Cause**

MATLAB is not on the system `PATH`. The launcher falls back to opening the file
when it cannot find the executable.

**Requirements**

- MATLAB R2020a or later.
- Signal Processing Toolbox for the audio tools.
- Statistics and Machine Learning Toolbox for the planning and modeling tools.
- Image Processing Toolbox for the visualization tools.

**Resolution**

Add MATLAB to `PATH` and verify it responds:

```bash
# Linux and macOS
export PATH="/usr/local/MATLAB/R2023a/bin:$PATH"
```

```powershell
# Windows PowerShell, current session
$env:PATH += ";C:\Program Files\MATLAB\R2023a\bin"
```

```bash
matlab -batch "disp('MATLAB is working')"
```

Set the variable permanently through the system environment settings rather than
per session. Python-only tools and the browser-based utilities do not depend on
MATLAB and remain available without it.

## Tests fail to collect

**Symptom**

`pytest` reports collection or import errors before running any test.

**Checks**

1. Run from the repository root, not from a subdirectory.
2. Activate the virtual environment used for the editable install.
3. Confirm test dependencies are present: `python -m pip install -e ".[test]"`.
4. Confirm the interpreter version is 3.11 or 3.12.

A second Python installation on `PATH` is the most common cause: the tests then
import a different interpreter's packages than the one the editable install
targeted.

## Rust extensions are not used

**Symptom**

A warning that the pure-Python fallback path is active, and slower than expected
numerical performance.

**Cause**

The optional Rust extensions are not published as pre-built wheels, so they are
absent until built locally.

**Resolution**

```bash
pip install maturin
cd rust_core/tools-core && maturin develop --features python
cd rust_core/ai_backend  && maturin develop --features python
```

See [Rust distribution](../development/rust_distribution.md) for crate contents
and measured performance differences.

## Still stuck

Search the [issue tracker](https://github.com/D-sorganization/Tools/issues)
before opening a new issue. Include the output of `python --version`,
`python -m pip show ud-tools`, and the relevant portion of
`unified_launcher.log`.
