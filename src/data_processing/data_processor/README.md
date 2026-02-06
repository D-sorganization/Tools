# Data Processor

A high-performance data processing application for filtering, smoothing, and transforming time-series data. Features a GUI built with CustomTkinter, vectorized NumPy/SciPy operations, and support for CSV/Excel import/export.

## Purpose

The Data Processor provides comprehensive signal processing capabilities for:

- Loading and combining multiple data files
- Applying digital filters (low-pass, high-pass, band-pass)
- Removing outliers and noise from sensor data
- Performing integration and differentiation
- Exporting processed data in multiple formats

## Key Features

- **Multiple Filter Types**: Moving Average, Butterworth, Median, Gaussian, Hampel, Z-Score, Savitzky-Golay, FFT filters
- **High-Performance Engine**: Vectorized operations using NumPy and SciPy
- **Parallel Processing**: Multi-threaded batch processing for large datasets
- **Signal Detection**: Automatic identification of numeric signals in data files
- **Custom Formulas**: Create derived signals using mathematical expressions
- **Multiple Export Formats**: CSV, Excel, Parquet, HDF5, Feather
- **Statistics Calculation**: Mean, std, min, max, median for all signals
- **Unit Conversion**: Built-in support for common engineering units

## Installation

### Prerequisites

- Python 3.10 or higher
- CustomTkinter
- Pandas
- NumPy
- SciPy

### Install Dependencies

```bash
pip install customtkinter pandas numpy scipy openpyxl pyarrow tables feather-format
```

### From Repository

```bash
cd Tools/src/data_processing/data_processor
python -m data_processor.gui_refactored
```

## Usage Instructions

### Launching the Application

```bash
# Using module
python -m data_processor.gui_refactored

# Or launcher script
python -m data_processor.launch_integrated

# CLI mode
python -m data_processor.cli --help
```

### Basic Workflow

1. **File Selection Tab**:
   - Click "Select Files" (Ctrl+O)
   - Choose one or more CSV files
   - Click "Load Data" (Ctrl+L)
   - Click "Detect Signals" to identify columns

2. **Signal Processing Tab**:
   - Select filter type from dropdown
   - Configure filter parameters
   - Click "Apply Filter"

3. **Advanced Operations Tab**:
   - Integrate or differentiate signals
   - Apply custom formulas

4. **Export Tab**:
   - Select export format
   - Click "Export Data" (Ctrl+S)
   - View statistics

## Input Parameters

### Filter Types and Parameters

#### Moving Average
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Window Size | Averaging window | 10 | 3-1000 |

#### Butterworth (Low-pass/High-pass)
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Order | Filter order | 3 | 1-10 |
| Cutoff Frequency | Normalized frequency | 0.1 | 0.01-0.99 |

#### Median Filter
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Kernel Size | Filter kernel (odd) | 5 | 3-101 |

#### Gaussian Filter
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Sigma | Standard deviation | 1.0 | 0.1-100 |

#### Hampel Filter
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Window Size | Detection window | 5 | 3-100 |
| Threshold | MAD multiplier | 3.0 | 1.0-10.0 |

#### Z-Score Filter
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Threshold | Outlier threshold | 3.0 | 1.0-10.0 |
| Method | Standard/Modified | Standard | Dropdown |

#### Savitzky-Golay
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Window Length | Must be odd | 11 | 3-101 |
| Polynomial Order | Fitting order | 2 | 1-6 |

#### FFT Filters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Frequency Low | Low cutoff | 0.01 | 0.0-0.5 |
| Frequency High | High cutoff | 0.1 | 0.0-0.5 |
| Window Shape | Window function | Gaussian | Dropdown |
| Transition BW | Rolloff width | 0.01 | 0.001-0.1 |
| Zero Phase | Phase preservation | True | Checkbox |

## Output Format

### Processed Data

Data maintains original structure with filtered columns:
```
Time,Signal1,Signal2,Signal3,...
0.0,1.234,5.678,9.012,...
0.1,1.235,5.679,9.013,...
```

### Statistics Output

```
=== Signal Statistics ===

Temperature:
  Mean: 823.45
  Std: 12.34
  Min: 798.12
  Max: 856.78
  Median: 822.50

Pressure:
  Mean: 1.02
  Std: 0.05
  Min: 0.89
  Max: 1.15
  Median: 1.01
```

### Export Formats

| Format | Extension | Best For |
|--------|-----------|----------|
| CSV | .csv | Universal compatibility |
| Excel | .xlsx | Spreadsheet analysis |
| Parquet | .parquet | Large datasets, fast I/O |
| HDF5 | .h5 | Hierarchical data, metadata |
| Feather | .feather | Python/R interchange |

## Example Usage

### Basic Filtering (GUI)

```bash
# Launch application
python -m data_processor.gui_refactored

# 1. Select CSV files
# 2. Load data
# 3. Apply Moving Average filter (window=20)
# 4. Export as CSV
```

### Command Line Interface

```bash
# Inspect file contents
python -m data_processor.cli inspect ./data/example.csv

# Run processing pipeline
python -m data_processor.cli run \
    --files ./data/example.csv \
    --signals time,pressure,temperature \
    --filter moving_average \
    --output ./output/processed.csv
```

### JSON Configuration Pipeline

Create `pipeline.json`:
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

Run with:
```bash
python -m data_processor.cli run --config pipeline.json
```

### Programmatic Usage

```python
import pandas as pd
from data_processor.vectorized_filter_engine import VectorizedFilterEngine

# Load data
df = pd.read_csv('sensor_data.csv')

# Create filter engine
engine = VectorizedFilterEngine(n_jobs=4)

# Apply filter
params = {'ma_window': 20}
filtered_df = engine.apply_filter_batch(
    df,
    'Moving Average',
    params,
    signal_names=['Temperature', 'Pressure']
)

# Export
filtered_df.to_csv('processed_data.csv', index=False)
```

### Batch Processing Multiple Files

```python
from data_processor.core.data_loader import DataLoader

loader = DataLoader(use_high_performance=True)

# Load multiple files
files = ['data1.csv', 'data2.csv', 'data3.csv']
dataframes = loader.load_multiple_files(files)
combined = loader.combine_dataframes(dataframes)

# Process combined data
engine = VectorizedFilterEngine()
processed = engine.apply_filter_batch(
    combined,
    'Butterworth Low-pass',
    {'bw_order': 3, 'bw_cutoff': 0.1}
)
```

### Custom Formula Application

```python
from data_processor.core.signal_processor import SignalProcessor

processor = SignalProcessor()

# Create derived signal
df, success = processor.apply_custom_formula(
    df,
    'Power',
    'Voltage * Current'
)

# Integration
from data_processor.models.processing_config import IntegrationConfig

int_config = IntegrationConfig(
    signals=['Velocity'],
    method='cumulative'
)
df = processor.integrate_signals(df, int_config)
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Select files |
| Ctrl+L | Load data |
| Ctrl+S | Export data |
| Ctrl+Q | Quit application |

## Troubleshooting

### File Loading Errors

**Issue**: "Failed to load data" error

**Solutions**:
- Verify file is valid CSV format
- Check file encoding (UTF-8 recommended)
- Ensure no locked files (close Excel)

### Filter Application Fails

**Issue**: "Signal too short for filtering"

**Solutions**:
- Reduce filter window/kernel size
- Ensure minimum data points (typically 10+)
- Check for excessive NaN values

### Memory Issues with Large Files

**Solutions**:
- Process files one at a time
- Use Parquet format for efficiency
- Increase system virtual memory
- Reduce parallel workers (n_jobs=1)

### Butterworth Filter Instability

**Issue**: Oscillations or artifacts

**Solutions**:
- Reduce filter order
- Adjust cutoff frequency
- Ensure adequate data length (order x 10 minimum)

### FFT Filter Ringing

**Solutions**:
- Increase transition bandwidth
- Use Gaussian or Tukey window
- Enable zero-phase filtering

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

## Related Tools

- **Multi-Parameter Analysis**: For analyzing processed data sensitivities
- **Optimizer GUI**: For optimizing filter parameters
- **Financial Calculator**: For processing financial time series

## Technical Notes

### Vectorized Operations

The filter engine uses NumPy/SciPy vectorization for performance:

```python
# Moving Average using scipy.ndimage
from scipy.ndimage import uniform_filter1d
filtered = uniform_filter1d(signal, size=window, mode='nearest')

# Butterworth using scipy.signal
from scipy.signal import butter, filtfilt
b, a = butter(N=order, Wn=cutoff, btype='low', fs=sample_rate)
filtered = filtfilt(b, a, signal)
```

### Parallel Processing

Thread pool executor for batch operations:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=n_jobs) as executor:
    futures = {executor.submit(filter_func, signal): name
               for name, signal in signals.items()}
```

### NaN Handling

Filters preserve NaN positions:
```python
result[original_signal.isna()] = np.nan
```

### Hampel Algorithm

Robust outlier detection using Median Absolute Deviation:
```python
MAD = median(|x_i - median(x)|)
threshold = k * 1.4826 * MAD  # k typically 3.0
```

### Constants Reference

Key processing constants (from `constants.py`):

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_FILE_SIZE_MB` | 500 | Maximum file size in MB |
| `CHUNK_SIZE` | 10000 | Processing chunk size |
| `DEFAULT_SAMPLE_RATE` | 1000 | Default sample rate (Hz) |
| `MAX_PLOT_POINTS` | 10000 | Maximum points to plot |
| `DEFAULT_DPI` | 300 | Export image resolution |

## Directory Structure

```
data_processor/
├── README.md                           # This file
├── ruff.toml                           # Linter configuration
├── data/                               # Sample data files
├── python/
│   ├── data_processor/
│   │   ├── __init__.py
│   │   ├── cli.py                      # Command-line interface
│   │   ├── constants.py                # Application constants
│   │   ├── gui_refactored.py           # Main GUI application
│   │   ├── logging_config.py           # Logging setup
│   │   ├── security_utils.py           # Security functions
│   │   ├── vectorized_filter_engine.py # High-performance filters
│   │   ├── core/                       # Core processing modules
│   │   └── models/                     # Data models
│   └── tests/                          # Unit tests
└── web/                                # Web application (Tauri)
```

## Version History

- **1.0.0**: Initial release with basic filters
- **1.1.0**: Added FFT filters and batch processing
- **1.2.0**: Vectorized engine for 10x performance
- **1.3.0**: CustomTkinter GUI refactoring
- **1.4.0**: Added Hampel and Z-Score filters
- **1.5.0**: Multi-format export support
- **1.6.0**: CLI interface with JSON configuration
