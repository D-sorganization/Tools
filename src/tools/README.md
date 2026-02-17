# Utility Tools

A collection of development utilities, code quality tools, and analysis scripts for the Golf Biomechanics Simulator & Game Engine repository.

## Overview

This directory contains general-purpose utility scripts and tools for code quality checking, scientific auditing, and MATLAB analysis. These tools support the development workflow and CI/CD pipeline.

## Directory Structure

```
tools/
├── README.md                    # This file
├── code_quality_check.py        # Python code quality checker
├── scientific_auditor.py        # Scientific code auditor
├── matlab_code_analyzer_gui/    # MATLAB analysis GUI tool
└── matlab_utilities/            # MATLAB quality utilities package
```

## Tools

### Code Quality Check (`code_quality_check.py`)

Automated Python code quality verification tool that checks for:

- **Banned Patterns**: TODO, FIXME, placeholders, NotImplementedError
- **Pass Statement Analysis**: Detects empty placeholder pass statements
- **Magic Numbers**: Flags hardcoded physics constants (pi, gravity, etc.)
- **AST Analysis**: Missing docstrings and type hints

**Usage:**

```bash
# Check all Python files in current directory
python tools/code_quality_check.py

# Check specific files
python tools/code_quality_check.py file1.py file2.py
```

**Pre-commit Integration:**

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: code-quality-check
        name: Code Quality Check
        entry: python tools/code_quality_check.py
        language: system
        types: [python]
```

### Scientific Auditor (`scientific_auditor.py`)

AST-based auditor for scientific Python code that detects potential issues:

- **Division by Zero Risk**: Flags variable division without zero checks
- **Unit Ambiguity**: Warns about trig functions with numeric constants (radians vs degrees)

**Usage:**

```bash
# Audit current directory
python tools/scientific_auditor.py

# Audit specific directory
python tools/scientific_auditor.py /path/to/code

# Output is JSON format
python tools/scientific_auditor.py | jq .
```

**Output Format:**

```json
[
  {
    "line": 42,
    "type": "Singularity Risk",
    "msg": "Division by variable detected. Check denominator."
  }
]
```

### MATLAB Code Analyzer GUI (`matlab_code_analyzer_gui/`)

Interactive graphical interface for MATLAB code analysis using the built-in Code Analyzer (MLint).

**Features:**

- Interactive file/folder selection
- Configurable analysis options
- Multiple output formats (CSV, Excel, JSON, Markdown)
- Progress tracking and results summary

See [matlab_code_analyzer_gui/README.md](matlab_code_analyzer_gui/README.md) for details.

### MATLAB Utilities (`matlab_utilities/`)

Comprehensive quality toolkit for MATLAB projects including:

- **Code Analysis**: mlint/checkcode integration
- **Static Analysis**: Python-based (no MATLAB license required)
- **Testing Framework**: Automated test execution
- **CI/CD Integration**: Pre-commit hooks and GitHub Actions support

See [matlab_utilities/README.md](matlab_utilities/README.md) for details.

## Integration with CI/CD

These tools are designed to work with the repository's CI/CD pipeline:

1. **Pre-commit Hooks**: Quality checks run before each commit
2. **GitHub Actions**: Automated quality checks on PR/push
3. **Local Development**: Run checks manually during development

### Quick Setup

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
bash scripts/setup_precommit.sh

# Or manually
pre-commit install
```

## Requirements

### Python Tools

- Python 3.8+ (3.11+ recommended)
- No additional dependencies for basic quality checks

### MATLAB Tools

- MATLAB R2019b+ (for full functionality)
- Code Analyzer (included with MATLAB)
- Python alternative available for CI/CD without MATLAB

## Excluded Directories

The quality check tools automatically exclude:

- `archive/`, `legacy/`, `experimental/`
- `.git/`, `__pycache__/`, `.mypy_cache/`
- `matlab/`, `output/`, `replicants/`
- `.ipynb_checkpoints/`, `.Trash/`

## Contributing

When adding new tools:

1. Follow existing code patterns and documentation style
2. Add self-documentation (help text, docstrings)
3. Exclude the tool from its own quality checks
4. Update this README with tool information

## License

Part of the Tools repository. See main repository license for details.
