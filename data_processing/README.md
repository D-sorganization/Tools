# Data Processing

A comprehensive suite for analyzing and converting time-series data from industrial sensors and scientific instruments.

## Overview

This directory contains tools for data analysis, signal processing, and visualization. The suite supports multiple file formats and provides both GUI and CLI interfaces for processing workflows.

## Components

### [Data Processor](data_processor/README.md)

A high-performance data processing application featuring:

- **Multi-format Support**: CSV, Excel, JSON, Parquet file import/export
- **Advanced Filtering**: Butterworth, FFT, Moving Average, Savitzky-Golay, and more
- **Batch Processing**: Process multiple files in automated pipelines
- **Visualization**: Interactive plotting with Matplotlib
- **GUI Application**: Modern interface with CustomTkinter
- **CLI Interface**: Command-line tool for automated workflows

## Quick Start

```bash
cd data_processing/data_processor/python

# Launch GUI
python -m data_processor.launch_integrated

# Or use CLI
python -m data_processor.cli --help
```

## Dependencies

```bash
pip install customtkinter pandas numpy scipy matplotlib openpyxl \
    Pillow simpledbf pyarrow tables feather-format typer rich
```

## Integration

This suite integrates with:
- **Scientific Modeling** (`scientific_modeling/`) - Advanced analysis
- **MATLAB** (`matlab/`) - Cross-platform compatibility
- **Tools** (`tools/`) - Quality checking and auditing

## License

Part of the Tools repository. See main repository license for details.
