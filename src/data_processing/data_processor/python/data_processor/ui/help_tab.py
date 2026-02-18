"""Help Tab for Data Processor."""

from __future__ import annotations

import customtkinter as ctk

_HELP_TEXT = """\
# 🚀 Advanced Data Processor - Complete Feature Guide

## 📋 Application Overview
This integrated application combines multiple powerful tools for data \
processing, analysis, and visualization:

### 🎯 Core Components
1. **📊 CSV Processor** - \
Advanced time series data processing with mathematical operations
2. **🔄 Format Converter** - \
Multi-format file conversion with batch processing and Parquet analysis
3. **📁 Folder Tool** - \
Comprehensive folder processing and organization with 5 operation modes
4. **📄 DAT File Import** - \
DAT file processing with DBF tag files for structured data
5. **📈 Plotting & Analysis** - \
Interactive visualization with smart auto-zoom and trendlines
6. **📋 Plots List** - \
Save and manage plot configurations for batch processing
7. **❓ Help** - This comprehensive documentation

### 🏗️ Architecture
- **Framework**: CustomTkinter (modern Python GUI framework)
- **Data Processing**: Pandas, NumPy, SciPy
- **File Formats**: Support for 15+ file formats
- **Visualization**: Matplotlib with interactive features
- **Threading**: Background processing for non-blocking operations

---

## 📊 CSV Processor Tab - Advanced Time Series Processing

### 🎯 Purpose & Capabilities
Transform raw CSV time series data into processed, analyzed, \
and visualized datasets with professional-grade mathematical operations.

### 📁 Setup Sub-tab - File Management & Configuration

#### 🔧 File Selection & Processing
- **📂 Input Files**: Multi-file selection with drag-and-drop
- **📁 Output Directory**: Configurable output location
- **⚙️ Configuration Management**: Save/load processing settings
- **📤 Export Format**: 15+ output formats
- **📊 Sorting Options**: Time-based and value-based sorting

#### 🚀 Usage Workflow
1. **Select Files**: Click "Select Files" or drag CSV files
2. **Set Output**: Choose destination folder
3. **Configure Processing**: Set up filtering, integration options
4. **Save Configuration**: Store settings for future use
5. **Select Signals**: Choose data columns to process
6. **Process & Export**: Execute with real-time progress

### 🔬 Processing Sub-tab - Advanced Signal Processing

#### 🔧 Signal Filtering (6 Professional Filters)
- **📈 Moving Average**: Configurable window size (3-1000 pts)
- **🌊 Butterworth Filter**: Low/high/band-pass with order control
- **🎯 Median Filter**: Configurable kernel size
- **📊 Savitzky-Golay**: Polynomial smoothing
- **🛡️ Hampel Filter**: Robust outlier detection
- **📏 Z-Score Filter**: Statistical outlier removal

#### ⏱️ Time Resampling & Interpolation
- **🔄 Resample Data**: Convert to different time intervals
- **📐 Interpolation**: Linear, cubic, nearest, polynomial
- **📊 Aggregation**: Mean, sum, min, max, median, std
- **🎯 Time Alignment**: Auto time column detection

#### 📈 Signal Integration
- **📊 Trapezoidal Integration**: With error estimation
- **🌊 Flow Calculations**: Rate → total volume conversion
- **🔧 Custom Integration**: User-defined methods
- **📏 Unit Conversion**: Automatic detection

#### 📉 Signal Differentiation
- **📐 Spline Differentiation**: Smooth derivatives
- **🔢 Finite Difference**: Forward, backward, central
- **📊 Multiple Orders**: 1st through 5th order
- **🎯 Smoothing Options**: Pre-filtering for noisy data

### 🧮 Custom Variables Sub-tab - Formula Builder

#### 🔧 Mathematical Formula Creation
- **📝 Formula Builder**: Visual with syntax highlighting
- **🔗 Signal Reference**: Use [SignalName] syntax
- **📊 Math Functions**: sin, cos, exp, log, sqrt, etc.
- **🔀 Conditional Logic**: if/else statements
- **📈 Statistical Functions**: mean, std, percentile, etc.

#### 💡 Example Formulas
```
[Flow] * 3600                  # Convert to hourly volume
sqrt([Pressure]^2 + [Temp]^2)  # Magnitude calculation
if([Value] > 100, [Value] * 2, [Value])  # Conditional
[Signal1] + [Signal2] * 0.5    # Weighted combination
log10([Concentration] + 1)     # Log transformation
```

---

## 🔄 Format Converter Tab - Universal File Conversion

### 🔧 Supported Formats
- **Data**: CSV, Parquet, Excel, JSON, HDF5, Feather, Arrow
- **Scientific**: MATLAB (.mat), Pickle, NumPy (.npy, .npz)
- **Database**: SQLite

### 🚀 Usage Workflow
1. **Select Input**: Browse files or folder
2. **Choose Output**: Select target format and destination
3. **Configure Options**: Set combination, column selection
4. **Convert**: Start high-performance conversion
5. **Analyze**: Inspect metadata of generated parquet files

---

## 📁 Folder Tool Tab - Industrial Folder Management

### 🛠️ Main Operation Modes
1. **Combine & Copy**: Aggregate files into one location
2. **Flatten & Tidy**: Extract from nested structures
3. **Copy & Prune**: Preserve structure, remove empty folders
4. **Deduplicate**: Remove renamed duplicates (in-place)
5. **Analyze & Report**: Generate statistical reports

### 🔍 Advanced Filtering
- Extension, size, and date filtering
- Smart organization (auto-type, auto-date)
- ZIP integration for archiving

---

## 📊 Plotting & Visualization

### 🔧 Key Features
- **🔍 Smart Zoom**: Rectangle zoom and auto-rescaling
- **📏 Trendlines**: Polynomial, linear, moving average
- **🎯 Value Inspection**: Real-time cursor coordinates
- **📉 Multi-Axis Support**: Separate axes for different units
- **📊 Statistical Overlay**: Mean, min, max, std lines

---

## 🎉 Getting Started - Quick Start Guide

### 🚀 First Steps
1. **📁 Load Data**: Start with CSV Processor tab
2. **🔧 Configure Settings**: Set processing preferences
3. **📊 Process Data**: Apply filters and math operations
4. **📈 Visualize Results**: Use Plotting tab
5. **💾 Save Work**: Save configs and results

### 💡 Pro Tips
- **💾 Always Backup**: Create backups before major ops
- **👁️ Use Preview**: Preview before execution
- **📊 Save Configurations**: Save frequently used settings
- **🔍 Validate Results**: Always check outputs
- **📈 Start Small**: Test with small datasets first

Welcome to professional data processing! 🚀
"""


class HelpTabMixin:
    """Mixin containing help tab UI."""

    def create_help_tab(self, tab: ctk.CTkFrame) -> None:
        """Create the help tab with comprehensive documentation."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(tab)
        header.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(
            header,
            text="🚀 Advanced Data Processor - Complete Help Guide",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left", padx=10, pady=10)

        help_frame = ctk.CTkScrollableFrame(tab)
        help_frame.grid(
            row=1,
            column=0,
            padx=10,
            pady=(0, 10),
            sticky="nsew",
        )
        help_frame.grid_columnconfigure(0, weight=1)

        help_text = ctk.CTkTextbox(
            help_frame,
            wrap="word",
            font=ctk.CTkFont(size=11),
        )
        help_text.pack(fill="both", expand=True, padx=10, pady=10)
        help_text.insert("1.0", _HELP_TEXT)
        help_text.configure(state="disabled")
