# CSV to Parquet Converter

A PyQt6-based GUI application for converting multiple CSV files to Parquet format with advanced features.

## Features

- **Bulk Conversion**: Convert multiple CSV files to Parquet format
- **Column Selection**: Advanced column selection with search and filtering
- **Save/Load Lists**: Save and load column selection lists for reuse
- **Large Dataset Support**: Efficient handling of files with 2000+ columns
- **Memory Optimization**: Chunked processing for large datasets
- **Progress Tracking**: Real-time progress updates and error handling
- **Flexible Output**: Combine into single file or keep as multiple files

## Files

- `csv_to_parquet_converter_enhanced.py` - Main application (most recent version)
- `csv_to_parquet_converter_stable.py` - Stable version
- `csv_to_parquet_converter.py` - Original version
- `test_*.py` - Test files for functionality verification
- `integration_example.py` - Example integration code
- `install_parquet_deps.py` - Dependency installation script
- `run_converter.bat` - Windows batch launcher

## Usage

1. Install dependencies: `python install_parquet_deps.py`
2. Run the converter: `python csv_to_parquet_converter_enhanced.py`
3. Or use the batch file: `run_converter.bat`

## Requirements

See `requirements.txt` for dependencies. 