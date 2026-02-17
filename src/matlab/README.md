# MATLAB Core

Core MATLAB scientific computing module for the Golf Biomechanics Simulator & Game Engine repository.

## Overview

This directory contains the core MATLAB scripts and infrastructure for running scientific simulations and reproducibility workflows. It serves as the central hub for MATLAB-based computations in the project.

## Directory Structure

```
matlab/
├── README.md           # This file
├── run_all.m           # Master workflow script
└── tests/              # Unit tests
    └── test_example.m  # Example test suite
```

## Core Components

### run_all.m

The `run_all.m` function executes a complete end-to-end workflow to recreate key results:

- **Reproducibility Setup**: Sets a standard seed (42) for consistent results
- **Output Management**: Creates timestamped output directories
- **Metadata Tracking**: Saves run metadata including MATLAB version and timestamps

**Usage:**

```matlab
run_all();
```

**Output:**

- Creates `output/YYYY-MM-DD/baseline/` directory
- Generates `metadata.json` with run information

### Tests

Unit tests follow MATLAB's `functiontests` framework:

```matlab
% Run example tests
tests = test_example;
run(tests);
```

## Requirements

- **MATLAB**: R2020b or later (for `arguments` validation blocks)
- **Toolboxes**: None required for core functionality

## Integration

This module integrates with other repository components:

- **Scientific Modeling** (`scientific_modeling/`): Advanced modeling resources
- **MATLAB Utilities** (`tools/matlab_utilities/`): Quality checking and testing tools

## Development Guidelines

1. **Reproducibility**: Always use the `REPRODUCIBILITY_SEED` constant for random number generation
2. **Documentation**: Include help text and arguments blocks in all functions
3. **Testing**: Add tests to the `tests/` directory for new functionality
4. **Output Management**: Use the established output directory structure

## Running Quality Checks

Use the MATLAB utilities package for code quality:

```matlab
addpath('../tools/matlab_utilities/quality');
results = run_quality_checks('.');
```

Or use Python for static analysis without MATLAB:

```bash
python ../tools/matlab_utilities/scripts/matlab_quality_check.py --project-root .
```

## License

Part of the Tools repository. See main repository license for details.
