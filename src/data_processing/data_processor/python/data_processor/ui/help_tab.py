"""Help Tab for Data Processor."""

from __future__ import annotations

import customtkinter as ctk


class HelpTabMixin:
    """Mixin containing help tab UI."""

    def create_help_tab(self, tab: ctk.CTkFrame):
        """Create the help tab with comprehensive documentation.

        For all integrated features.
        """
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(tab)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(
            header_frame,
            text="🚀 Advanced Data Processor - Complete Help Guide",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left", padx=10, pady=10)

        # Main content with scrollable help
        help_frame = ctk.CTkScrollableFrame(tab)
        help_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        help_frame.grid_columnconfigure(0, weight=1)

        # Comprehensive help content with enhanced formatting
        help_content = """
# 🚀 Advanced Data Processor - Complete Feature Guide

## 📋 Application Overview
This integrated application combines multiple powerful tools for data processing,
analysis, and visualization:

### 🎯 Core Components
1. **📊 CSV Processor** -
Advanced time series data processing with mathematical operations
2. **🔄 Format Converter** -
Multi-format file conversion with batch processing and Parquet analysis
3. **📁 Folder Tool** -
Comprehensive folder processing and organization with 5 operation modes
4. **📄 DAT File Import** - DAT file processing with DBF tag files for structured data
5. **📈 Plotting & Analysis** -
Interactive visualization with smart auto-zoom and trendlines
6. **📋 Plots List** - Save and manage plot configurations for batch processing
7. **❓ Help** - This comprehensive documentation

### 🏗️ Architecture
- **Framework**: CustomTkinter (modern Python GUI framework)
- **Data Processing**: Pandas, NumPy, SciPy for advanced mathematical operations
- **File Formats**: Support for 15+ file formats (CSV, Parquet, Excel, JSON, HDF5, etc.)
- **Visualization**: Matplotlib with interactive features
- **Threading**: Background processing for non-blocking operations

---

## 📊 CSV Processor Tab - Advanced Time Series Processing

### 🎯 Purpose & Capabilities
Transform raw CSV time series data into processed, analyzed,
    and visualized datasets with
    professional-grade mathematical operations.

### 📁 Setup Sub-tab - File Management & Configuration

#### 🔧 File Selection & Processing
- **📂 Input Files**: Multi-file selection with drag-and-drop support
- **📁 Output Directory**: Configurable output location with automatic creation
- **⚙️ Configuration Management**: Save/load complete processing settings
- **📤 Export Format**:
15+ output formats (CSV, Excel, MAT, Parquet, HDF5, Feather, etc.)
- **📊 Sorting Options**: Time-based and value-based sorting configurations

#### 🚀 Usage Workflow
1. **Select Files**: Click "Select Files" or drag CSV files into the interface
2. **Set Output**: Choose destination folder for processed files
3. **Configure Processing**: Set up filtering, integration, differentiation options
4. **Save Configuration**: Store settings for future use (recommended)
5. **Select Signals**: Choose which data columns to process
6. **Process & Export**: Execute processing with real-time progress tracking

### 🔬 Processing Sub-tab - Advanced Signal Processing

#### 🔧 Signal Filtering (6 Professional Filters)
- **📈 Moving Average**: Smooth data with configurable window size (3-1000 points)
- **🌊 Butterworth Filter**: Low-pass, high-pass, band-pass filtering with order control
- **🎯 Median Filter**: Remove outliers with configurable kernel size
- **📊 Savitzky-Golay**:
Polynomial smoothing for noisy data with window/polynomial control
- **🛡️ Hampel Filter**: Robust outlier detection and removal with statistical thresholds
- **📏 Z-Score Filter**: Statistical outlier removal with configurable sigma values

#### ⏱️ Time Resampling & Interpolation
- **🔄 Resample Data**: Convert to different time intervals (1s, 1min, 1h, 1d, custom)
- **📐 Interpolation Methods**: Linear, cubic, nearest neighbor, polynomial
- **📊 Aggregation Functions**: Mean, sum, min, max, median, std, custom functions
- **🎯 Time Alignment**: Automatic time column detection and alignment

#### 📈 Signal Integration (Mathematical Operations)
- **📊 Trapezoidal Integration**: Calculate cumulative values with error estimation
- **🌊 Flow Calculations**: Convert rate data to total volumes with unit conversion
- **🔧 Custom Integration**: User-defined integration methods and formulas
- **📏 Unit Conversion**: Automatic unit detection and conversion

#### 📉 Signal Differentiation (Advanced Calculus)
- **📐 Spline Differentiation**: Smooth derivative calculation with configurable order
- **🔢 Finite Difference**: Direct numerical differentiation
  (forward, backward, central)
- **📊 Multiple Orders**: 1st through 5th order derivatives with error analysis
- **🎯 Smoothing Options**: Pre-filtering for noisy derivative calculations

### 🧮 Custom Variables Sub-tab - Formula Builder

#### 🔧 Mathematical Formula Creation
- **📝 Formula Builder**: Visual formula creation with syntax highlighting
- **🔗 Signal Reference**: Use [SignalName] syntax to reference existing data columns
- **📊 Mathematical Functions**: sin, cos, exp, log, sqrt, abs, pow, etc.
- **🔀 Conditional Logic**: if/else statements for complex conditional calculations
- **📈 Statistical Functions**: mean, std, min, max, percentile, etc.

#### 💡 Example Formulas
```
[Flow] * 3600                    # Convert flow rate to hourly volume
sqrt([Pressure]^2 + [Temp]^2)    # Calculate magnitude from components
if([Value] > 100, [Value] * 2, [Value])  # Conditional processing
[Signal1] + [Signal2] * 0.5      # Weighted combination
log10([Concentration] + 1)        # Log transformation with offset
```

#### 🎯 Advanced Features
- **🔍 Formula Validation**: Real-time syntax checking and error detection
- **📊 Result Preview**: Preview calculated values before processing

---

## 🔄 Format Converter Tab - Universal File Conversion

### 🎯 Purpose & Capabilities
Convert datasets between dozens of different formats with high performance
and optional batch processing.

### 🔧 Supported Formats
- **Data Formats**: CSV, Parquet, Excel (XLSX), JSON, HDF5, Feather, Arrow
- **Scientific Formats**: MATLAB (.mat), Pickle, NumPy (.npy, .npz)
- **Database Formats**: SQLite

### 🚀 Usage Workflow
1. **Select Input**: Use "Browse Files" or "Browse Folder" to select data
2. **Choose Output**: Select target format and destination directory
3. **Configure Options**: Set combination, column selection, and splitting options
4. **Convert**: Start high-performance conversion process
5. **Analyze**: Use "Analyze Parquet" to inspect metadata of generated parquet files

### 📊 Column Selection
- Selective column extraction to reduce file size and improve performance
- Search and filter capabilities for datasets with hundreds of columns
- Save column selection profiles for repeated tasks

### ⚡ Performance Optimization
- **Batch Processing**: Parallel conversion for large file sets
- **Smart Chunking**: Memory-efficient processing of extremely large files
- **Format-Specific Optimization**: High-speed I/O for Parquet and HDF5

---

## 📁 Folder Tool Tab - Industrial Folder Management

### 🎯 Purpose & Capabilities
A professional toolkit for managing massive file collections, organizing data,
and performing industrial-grade directory operations.

### 🛠️ Main Operation Modes
1. **Combine & Copy**: Aggregate files from multiple sources into one location
2. **Flatten & Tidy**: Extract all files from complex nested structures to a flat root
3. **Copy & Prune**: Preserve structure while automatically removing empty folders
4. **Deduplicate**: Reclaim space by removing renamed duplicates (In-Place)
5. **Analyze & Report**: Generate detailed statistical reports of folder contents

### 🔍 Advanced Filtering
- **Extension Filtering**: Process only specific file types (e.g., .csv, .dat)
- **Size Filtering**: Exclude files based on minimum and maximum size thresholds
- **Date Filtering**: Process files within specific time ranges

### 📁 Smart Organization
- **Auto-Type**: Group files into "Images", "Documents", "Sensors", etc.
- **Auto-Date**: Organize files into YYYY/MM/DD structure based on creation date
- **ZIP Integration**: Automatically archive results for easy distribution

### 🛡️ Safety Features
- **Preview Mode**: See exactly what would happen before any files are touched
- **Backup Option**: Create automatic backups before destructive operations
- **Conflict Resolution**: Smart renaming for files with identical names

---

## 📄 DAT File Import - Engineering Data Support

### 🎯 Purpose & Capabilities
Specialized support for industrial DAT files using DBF tag files for metadata parsing.

### 🚀 Usage Workflow
1. **Select DAT**: Choose the raw binary data file
2. **Select DBF**: Choose the corresponding tag definition file
3. **Map Sensors**: Use the auto-mapping feature to link binary data to sensors
4. **Import**: Convert specialized industrial data into standard formats

---

## 📊 Plotting & Visualization - Interactive Analysis

### 🎯 Overview
High-performance interactive plotting system designed for large engineering datasets.

### 🔧 Key Interactive Features
- **🔍 Smart Zoom**: Rectangle zoom and auto-rescaling
- **📏 Trendlines**: Polynomial, linear, and moving average trendlines
- **🎯 Value Inspection**: Real-time cursor coordinates and data peaking
- **📉 Multi-Axis Support**: Compare signals with different units on separate axes
- **📊 Statistical Overlay**: Mean, min, max, and std lines

### 📋 Plots List Management
- Save specific plot configurations including zoom levels and trendlines
- Batch generate images for report documentation
- Share plot configurations between team members

---

## 🛠️ Performance & Troubleshooting

### 💾 Memory Management
- The application uses memory mapping and chunking for files larger than system RAM
- For extremely large files, use the "Batch processing" or "Split" options

### 🛡️ Error Handling
- Check the "Conversion Log" or console output for detailed error messages
- Ensure all necessary libraries (PyArrow, FastParquet) are installed for specific formats

---

## 💾 Advanced Configuration & API

### 📜 Configuration Files (.json)
All processing settings are stored in standard JSON format and can be manually edited
if necessary.

### 🔄 External Integration
Exported formats like Parquet and HDF5 are optimized for use in:
- Python (Pandas/Dask)
- MATLAB
- R
- Excel (PowerQuery)
- Tableau / PowerBI / Spotfire

### 🛠️ Support & Updates
- **💾 Memory Efficient**: Smart memory management
- **🔄 Batch Operations**: Process hundreds of files simultaneously
- **📊 Large File Support**: Handle files of any size
- **🛡️ Robust Error Handling**: Comprehensive error recovery
- **📈 Real-time Progress**: Live progress tracking

For technical support or feature requests, please refer to
    the application documentation or contact the development team.

---

## 🎉 Getting Started - Quick Start Guide

### 🚀 First Steps
1. **📁 Load Data**: Start with the CSV Processor tab to load your data
2. **🔧 Configure Settings**: Set up your processing preferences
3. **📊 Process Data**: Apply filters and mathematical operations
4. **📈 Visualize Results**: Use the Plotting tab to explore your data
5. **💾 Save Work**: Save configurations and results for future use

### 🎯 Common Workflows
- **📊 Data Analysis**: Load → Process → Visualize → Export
- **🔄 Format Conversion**: Select → Convert → Analyze → Save
- **📁 File Organization**: Select → Organize → Validate → Backup
- **📈 Report Generation**: Process → Plot → Configure → Export

### 💡 Pro Tips
- **💾 Always Backup**: Create backups before major operations
- **👁️ Use Preview**: Preview operations before execution
- **📊 Save Configurations**: Save frequently used settings
- **🔍 Validate Results**: Always check processing results
- **📈 Start Small**: Test with small datasets first

Welcome to professional data processing! 🚀
"""

        # Create text widget for help content with enhanced styling
        help_text = ctk.CTkTextbox(help_frame, wrap="word", font=ctk.CTkFont(size=11))
        help_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Insert help content
        help_text.insert("1.0", help_content)

        # Make text read-only
        help_text.configure(state="disabled")
