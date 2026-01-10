# Data Processor

A comprehensive, high-performance data processing application designed for chemical plant data analysis, signal processing, and visualization.

## Overview

Data Processor is a full-featured application that provides both GUI and CLI interfaces for processing time-series data from industrial sensors and scientific instruments. It features vectorized filtering operations, batch processing, and extensive export capabilities.

## Features

### Core Functionality
- **Multi-format Support**: CSV, Excel, JSON, Parquet file import/export
- **Signal Discovery**: Quickly inspect files to discover available signals
- **Batch Processing**: Process multiple files in automated pipelines
- **High-performance Filtering**: Vectorized operations using NumPy/SciPy

### Advanced Filtering Suite
- **Butterworth Filters**: Low-pass, high-pass, band-pass, band-stop
- **Moving Average**: Configurable window size
- **Median Filter**: Kernel-based noise reduction
- **Savitzky-Golay**: Smoothing with polynomial fitting
- **Gaussian Filter**: Sigma-controlled smoothing
- **FFT Filters**: Frequency-domain filtering with customizable windows
- **Hampel Filter**: Outlier detection and removal
- **Z-Score Filter**: Statistical outlier removal

### Visualization
- Interactive plotting with Matplotlib
- Multi-signal overlay support
- Zoom and pan controls
- Export plots at configurable DPI (default: 300)

### GUI Application
- Modern interface with CustomTkinter
- Signal browser and selector
- Real-time filter preview
- Format converter (CSV, Excel, Parquet)
- Parquet file analyzer

## Directory Structure

```
data_processor/
├── README.md                           # This file
├── ruff.toml                           # Linter configuration
├── archive/                            # Legacy versions
├── data/                               # Sample data files
├── python/
│   ├── benchmarks/
│   │   └── performance_benchmark.py    # Performance testing
│   ├── data_processor/
│   │   ├── __init__.py
│   │   ├── cli.py                      # Command-line interface
│   │   ├── constants.py                # Application constants
│   │   ├── file_utils.py               # File handling utilities
│   │   ├── gui_refactored.py           # Main GUI application
│   │   ├── high_performance_loader.py  # Optimized data loading
│   │   ├── launch_integrated.py        # GUI launcher
│   │   ├── logging_config.py           # Logging setup
│   │   ├── security_utils.py           # Security functions
│   │   ├── vectorized_filter_engine.py # High-performance filters
│   │   ├── core/                       # Core processing modules
│   │   ├── data/                       # Data assets
│   │   └── models/                     # Data models
│   └── tests/
│       ├── test_signal_processor.py
│       ├── test_processing_config.py
│       └── test_file_utils.py
└── tools/                              # Additional utilities
```

## Installation

### Dependencies

```bash
pip install customtkinter pandas numpy scipy matplotlib openpyxl \
    Pillow simpledbf pyarrow tables feather-format typer rich
```

### Quick Start

```bash
cd data_processing/data_processor/python

# Launch GUI
python -m data_processor.launch_integrated

# Or use CLI
python -m data_processor.cli --help
```

## Usage

### GUI Application

```bash
python -m data_processor.launch_integrated
```

Features:
- Original CSV processing functionality
- Format converter with support for multiple file formats
- Parquet file analyzer
- All existing plotting and analysis features

### Command Line Interface

The CLI focuses on two core automated workflows:

#### 1. Inspect Files

```bash
python -m data_processor.cli inspect ./data/example.csv
```

#### 2. Run Processing Pipeline

Using CLI flags:
```bash
python -m data_processor.cli run \
    --files ./data/example.csv \
    --signals time,pressure,temperature \
    --filter moving_average \
    --output ./output/processed.csv
```

Using JSON configuration:
```bash
python -m data_processor.cli run --config pipeline.json
```

Example `pipeline.json`:
```json
{
  "files": ["./data/example.csv"],
  "combine": true,
  "selected_signals": ["time", "pressure", "temperature"],
  "filter": {
    "filter_type": "Moving Average",
    "ma_window": 5
  },
  "output": {
    "path": "./output/processed.csv",
    "format": "csv"
  }
}
```

## Filter Configuration

### Butterworth Filter
```python
{
    "filter_type": "Butterworth",
    "cutoff": 0.1,      # Normalized cutoff frequency
    "order": 4,         # Filter order
    "btype": "lowpass"  # lowpass, highpass, bandpass, bandstop
}
```

### Moving Average
```python
{
    "filter_type": "Moving Average",
    "ma_window": 5      # Window size
}
```

### FFT Filter
```python
{
    "filter_type": "FFT",
    "freq_low": 0.1,           # Low frequency cutoff (Hz)
    "freq_high": 100.0,        # High frequency cutoff (Hz)
    "transition_bw": 0.1,      # Transition bandwidth
    "window_shape": "hann",    # Window function
    "zero_phase": true         # Zero-phase filtering
}
```

## Constants Reference

Key processing constants (from `constants.py`):

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_FILE_SIZE_MB` | 500 | Maximum file size in MB |
| `CHUNK_SIZE` | 10000 | Processing chunk size |
| `DEFAULT_SAMPLE_RATE` | 1000 | Default sample rate (Hz) |
| `MAX_PLOT_POINTS` | 10000 | Maximum points to plot |
| `DEFAULT_DPI` | 300 | Export image resolution |

## Running Tests

```bash
cd data_processing/data_processor/python

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=data_processor

# Run specific test
pytest tests/test_signal_processor.py -v
```

## Performance

The vectorized filter engine is optimized for:
- NumPy/SciPy vectorized operations
- Batch processing of multiple signals
- Memory-efficient streaming operations
- Parallel processing via ThreadPoolExecutor

## Integration

This tool integrates with:
- **Scientific Modeling** (`scientific_modeling/`) - Advanced analysis
- **MATLAB** (`matlab/`) - Cross-platform compatibility
- **Tools** (`tools/`) - Quality checking and auditing

## License

Part of the Tools repository. See main repository license for details.
