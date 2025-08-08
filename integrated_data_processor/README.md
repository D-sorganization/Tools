# Integrated Data Processor

A comprehensive, self-contained data processing application that combines multiple tools into one unified interface.

## Features

- **CSV Processing**: Load, process, and analyze CSV files with advanced filtering and signal processing
- **Format Converter**: Convert between 15+ file formats (CSV, Excel, Parquet, JSON, HDF5, etc.)
- **Parquet Analyzer**: Analyze Parquet file metadata and structure
- **Folder Tool**: Comprehensive folder processing (combine, flatten, deduplicate, analyze)
- **DAT File Import**: Import and process DAT files
- **Plotting & Analysis**: Interactive plotting with matplotlib integration
- **Batch Processing**: Process multiple files with progress tracking

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the application:
```bash
python launch.py
```

## File Structure

```
integrated_data_processor/
├── Data_Processor_Integrated.py    # Main integrated application
├── Data_Processor_r0.py            # Base CSV processor class
├── file_utils.py                   # File utility functions
├── threads.py                      # Threading utilities
├── folder_tool_tab.py              # Folder tool UI components
├── folder_tool/                    # Folder processing functionality
│   └── Folder_Cleanup_Tool_Rev0.py
├── launch.py                       # Launch script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Usage

The application provides a tabbed interface with the following sections:

1. **Processing**: CSV file loading and processing
2. **Plotting & Analysis**: Data visualization and analysis
3. **Plots List**: Plot management and export
4. **Format Converter**: Multi-format file conversion
5. **DAT File Import**: DAT file processing
6. **Folder Tool**: Folder processing operations
7. **Help**: Documentation and assistance

## Self-Contained

This version is completely self-contained with all necessary modules included. No external file references are required.
