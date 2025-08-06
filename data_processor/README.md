# CSV to Parquet Bulk Converter

A robust PyQt6-based GUI application for converting multiple CSV files to Parquet format with support for large file sets (10,000+ files) and memory-efficient processing.

## Features

### Core Functionality
- **Bulk Conversion**: Convert thousands of CSV files to Parquet format in a single operation
- **Large File Support**: Handle files with millions of rows using chunked processing
- **Flexible Output**: Choose between single combined file or multiple individual files
- **Memory Efficient**: Uses PyArrow for optimized memory usage and processing speed
- **Progress Tracking**: Real-time progress updates and detailed logging
- **Error Handling**: Robust error handling with detailed error reporting

### GUI Features
- **Modern Interface**: Clean, intuitive PyQt6-based user interface
- **File Selection**: Easy folder browsing with automatic CSV file detection
- **Configuration Options**: Adjustable chunk sizes and output preferences
- **Real-time Logging**: Live conversion log with save functionality
- **Cancel Support**: Ability to cancel long-running conversions
- **Responsive Design**: Resizable interface with proper layout management

### Integration Ready
- **Modular Design**: Can be easily integrated as a subtab in other PyQt6 applications
- **Standalone Operation**: Works as a complete standalone application
- **Clean API**: Well-documented classes and methods for easy integration

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install PyQt6>=6.4.0 pandas>=1.5.0 pyarrow>=10.0.0 numpy>=1.21.0
```

## Usage

### Standalone Application

Run the converter as a standalone application:

```bash
python csv_to_parquet_converter.py
```

### Integration Example

See the integration example to understand how to use it as a subtab:

```bash
python integration_example.py
```

## How to Use

### 1. Select Input Files
- Click "Browse Folder" to select a folder containing CSV files
- The application will automatically scan for all `.csv` files in the selected folder and subfolders
- The file count will be displayed once scanning is complete

### 2. Configure Output Options
- **Combine Files**: Check this option to combine all CSV files into a single Parquet file
- **Individual Files**: Uncheck to create separate Parquet files for each CSV
- **Chunk Size**: Adjust the chunk size for processing large files (default: 10,000 rows)
- **Output Path**: Select the output file (for combined mode) or folder (for individual mode)

### 3. Start Conversion
- Click "Start Conversion" to begin the process
- Monitor progress in the progress bar and status updates
- View detailed logs in the conversion log section
- Cancel the operation at any time using the "Cancel" button

### 4. Review Results
- Check the conversion log for any errors or warnings
- Save the log file for future reference if needed
- The converted Parquet files will be available at the specified output location

## Integration Guide

### Basic Integration

To integrate the converter as a subtab in your PyQt6 application:

```python
from csv_to_parquet_converter import CSVToParquetConverter
from PyQt6.QtWidgets import QTabWidget

# Create your main application
class MyApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Add the converter as a tab
        converter_widget = CSVToParquetConverter()
        self.tab_widget.addTab(converter_widget, "CSV to Parquet")
```

### Advanced Integration

For more advanced integration with custom styling and functionality:

```python
class CustomCSVConverter(CSVToParquetConverter):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Add custom functionality
        self.setup_custom_features()
    
    def setup_custom_features(self):
        # Add your custom features here
        pass
    
    def _conversion_finished(self, success, message):
        # Override to add custom completion handling
        super()._conversion_finished(success, message)
        
        if success:
            # Trigger custom actions on success
            self.parent().handle_conversion_success()
```

## Performance Considerations

### Large File Sets (10,000+ files)
- The application uses efficient file scanning with `pathlib.rglob()`
- Progress updates are batched to prevent UI freezing
- Memory usage is optimized through chunked processing

### Large Individual Files
- Adjust chunk size based on available memory
- Default chunk size (10,000 rows) works well for most scenarios
- For very large files, consider reducing chunk size to 5,000-8,000 rows

### Memory Optimization
- Uses PyArrow for efficient data handling
- Processes files in chunks to minimize memory usage
- Automatically cleans up resources after processing

## File Format Support

### Input Formats
- **CSV Files**: Standard comma-separated values
- **Encoding**: UTF-8 (default), other encodings supported by pandas
- **Headers**: First row assumed to be column headers
- **Delimiters**: Comma (configurable in pandas if needed)

### Output Formats
- **Parquet Files**: Columnar storage format optimized for analytics
- **Compression**: Automatic compression for smaller file sizes
- **Schema Preservation**: Maintains data types and column structure

## Troubleshooting

### Common Issues

**No CSV files found:**
- Ensure the selected folder contains `.csv` files
- Check file extensions (case-sensitive)
- Verify folder permissions

**Memory errors:**
- Reduce chunk size in the settings
- Close other applications to free memory
- Process files in smaller batches

**Conversion errors:**
- Check CSV file format and encoding
- Ensure all files have consistent column structure
- Review the conversion log for specific error details

### Log Files
- Application logs are saved to `csv_to_parquet.log`
- Conversion logs can be saved manually from the GUI
- Check logs for detailed error information

## Development

### Project Structure
```
csv_to_parquet_converter.py    # Main application
integration_example.py         # Integration example
requirements.txt              # Dependencies
README.md                     # This file
```

### Key Classes

- **`CSVToParquetConverter`**: Main GUI widget
- **`ConversionWorker`**: Background processing thread
- **`MainWindow`**: Standalone application window

### Extending Functionality

To add new features:

1. **New File Formats**: Extend the `ConversionWorker` class
2. **Additional Options**: Add UI elements to `CSVToParquetConverter`
3. **Custom Processing**: Override conversion methods in `ConversionWorker`

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the conversion logs
3. Ensure all dependencies are properly installed
4. Verify file formats and permissions
