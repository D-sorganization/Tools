# Large Dataset Optimizations

## Overview
The data processor has been optimized to handle large numbers of CSV files (10,000+ files) without freezing the UI. This document explains the optimizations and how to use them.

## Key Optimizations

### 1. Smart File List Display
- **For ≤100 files**: Shows detailed list with individual remove buttons
- **For >100 files**: Shows summary display with file count and sample files
- **Benefits**: Prevents UI freezing when loading thousands of files

### 2. Batch Processing for Signal Loading
- Processes files in batches (50 files per batch for large datasets)
- Updates UI progress between batches
- Prevents memory issues and UI freezing

### 3. CSV to Parquet Conversion
- **New Feature**: Direct conversion of multiple CSV files to a single Parquet file
- **Optimized for large datasets**: Processes files in batches of 100
- **Memory efficient**: Clears batch data after processing
- **Progress tracking**: Real-time progress bar and status updates

## How to Use

### 1. Install Parquet Dependencies
```bash
python install_parquet_deps.py
```

### 2. Load Large Numbers of Files
1. Click "Select Input CSV Files"
2. Select your 10,000+ CSV files
3. The UI will automatically switch to summary mode for large file counts
4. Use "Show All Files" button to view the complete list if needed

### 3. Convert to Parquet
1. Select your CSV files
2. Click the green "Convert to Parquet (Large Datasets)" button
3. Choose output location for the Parquet file
4. Monitor progress in the conversion window
5. The resulting Parquet file will contain all your data with a `source_file` column

## Benefits of Parquet Format

### Storage Efficiency
- **Compression**: Typically 2-4x smaller than CSV
- **Columnar storage**: Better for analytical queries
- **Schema preservation**: Maintains data types

### Performance
- **Faster reading**: 10-100x faster than CSV for large datasets
- **Selective column reading**: Can read only needed columns
- **Better memory usage**: More efficient memory layout

### Compatibility
- **Wide support**: Works with pandas, Dask, Spark, and many other tools
- **Cloud storage**: Optimized for cloud storage systems
- **Big data tools**: Native support in most big data platforms

## Technical Details

### Memory Management
- Files are processed in batches to prevent memory overflow
- Batch dataframes are cleared after each batch
- Final concatenation is done efficiently

### Error Handling
- Individual file errors don't stop the entire process
- Progress is maintained even if some files fail
- Detailed error logging for troubleshooting

### UI Responsiveness
- Progress updates every batch
- UI remains responsive during processing
- Cancel option available (can be added if needed)

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size in the code (currently 100)
   - Close other applications to free memory
   - Process files in smaller groups

2. **Slow Performance**
   - Ensure files are on fast storage (SSD preferred)
   - Close other applications
   - Consider using a machine with more RAM

3. **Missing pyarrow**
   - Run `python install_parquet_deps.py`
   - Or manually install: `pip install pyarrow`

### Performance Tips

1. **File Organization**
   - Keep files in the same directory for faster access
   - Use consistent file naming patterns

2. **System Resources**
   - Use SSD storage for better I/O performance
   - Ensure sufficient RAM (8GB+ recommended for 10k files)
   - Close unnecessary applications

3. **File Size Considerations**
   - Very large individual files (>1GB) may need special handling
   - Consider splitting extremely large files first

## Future Enhancements

Potential improvements that could be added:
- Parallel processing for even faster conversion
- Compression options for Parquet files
- Progress saving/resuming for interrupted conversions
- Automatic file validation before conversion
- Integration with cloud storage services 