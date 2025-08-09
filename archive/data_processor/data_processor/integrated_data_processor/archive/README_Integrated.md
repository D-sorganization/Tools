# Integrated Data Processor

This is an integrated version of the Advanced CSV Time Series Processor & Analyzer that includes the compiler converter functionality as an additional tab.

## Features

### Original Functionality
- CSV file processing and analysis
- Advanced filtering (Moving Average, Butterworth, Median, Savitzky-Golay, Hampel, Z-Score)
- Integration and differentiation of signals
- Custom variable calculations
- Comprehensive plotting and visualization
- DAT file import capabilities
- Configuration management and settings persistence

### New Integrated Features
- **Format Converter Tab**: Convert between multiple file formats
- **Parquet File Analyzer**: Analyze parquet file metadata without loading the entire file
- **Multi-format Support**: CSV, TSV, Parquet, Excel, JSON, HDF5, Pickle, NumPy, MATLAB, Feather, Arrow, SQLite
- **Batch Processing**: Process multiple files at once
- **File Splitting**: Split large files into smaller chunks
- **Column Selection**: Choose specific columns for conversion

## File Formats Supported

### Input/Output Formats
- **CSV** (.csv) - Comma-separated values
- **TSV** (.tsv, .txt) - Tab-separated values
- **Parquet** (.parquet, .pq) - Columnar storage format
- **Excel** (.xlsx, .xls) - Microsoft Excel files
- **JSON** (.json) - JavaScript Object Notation
- **HDF5** (.h5, .hdf5) - Hierarchical Data Format
- **Pickle** (.pkl, .pickle) - Python serialization format
- **NumPy** (.npy) - NumPy array format
- **MATLAB** (.mat) - MATLAB matrix files
- **Feather** (.feather) - Fast columnar format
- **Arrow** (.arrow) - Apache Arrow format
- **SQLite** (.db, .sqlite) - SQLite database files

## Installation

1. Ensure you have Python 3.8+ installed
2. Install required dependencies:
   ```bash
   pip install customtkinter pandas numpy scipy matplotlib openpyxl Pillow simpledbf pyarrow tables feather-format
   ```

## Usage

### Launching the Application

#### Windows
- Double-click `run_integrated.bat` or `run_integrated.ps1`
- Or run: `python launch_integrated.py`

#### Linux/Mac
- Run: `python3 launch_integrated.py`

### Using the Format Converter

1. **Select Input Files**: Use "Browse Files" or "Browse Folder" to select input files
2. **Choose Output Format**: Select the desired output format from the dropdown
3. **Set Output Path**: Choose where to save the converted files
4. **Configure Options**:
   - **Combine Files**: Merge all input files into a single output file
   - **Use All Columns**: Include all columns in the output
   - **Batch Processing**: Process files in batches for memory efficiency
   - **Split Large Files**: Split files that exceed size limits
5. **Convert**: Click "Convert Files" to start the conversion process

### Using the Parquet Analyzer

1. Click "Analyze Parquet" button in the Format Converter tab
2. Select a parquet file to analyze
3. View detailed metadata including:
   - File size and structure
   - Row and column counts
   - Schema information
   - Row group details
   - Column statistics

## File Structure

```
data_processor/
├── Data_Processor_Integrated.py    # Main integrated application
├── Data_Processor_r0.py            # Original data processor
├── launch_integrated.py            # Launch script
├── run_integrated.bat              # Windows batch file
├── run_integrated.ps1              # PowerShell script
├── README_Integrated.md            # This file
└── requirements.txt                # Dependencies
```

## Dependencies

- **customtkinter**: Modern GUI framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **matplotlib**: Plotting and visualization
- **openpyxl**: Excel file support
- **Pillow**: Image processing
- **simpledbf**: DBF file support
- **pyarrow**: Parquet and Arrow format support
- **tables**: HDF5 support
- **feather-format**: Feather format support

## Notes

- The integrated version extends the original data processor without modifying its core functionality
- All original features remain available in their respective tabs
- The Format Converter tab provides additional file format conversion capabilities
- The Parquet Analyzer is accessible from the Format Converter tab
- Settings and configurations are preserved between sessions

## Troubleshooting

### Import Errors
If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### PyArrow Issues
For parquet file support, ensure PyArrow is installed:
```bash
pip install pyarrow
```

### Memory Issues
For large files, consider:
- Using batch processing
- Enabling file splitting
- Selecting only necessary columns
- Using more efficient formats like Parquet

## Contributing

This integrated version maintains compatibility with the original data processor while adding new functionality. When making changes:

1. Preserve the original functionality
2. Test both original and new features
3. Update documentation as needed
4. Ensure backward compatibility
