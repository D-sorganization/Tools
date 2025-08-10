"""
Constants for Data Processor application.

This module contains all configuration constants used throughout the application
to avoid magic numbers and provide clear documentation of values and their sources.
"""

from typing import Final

# =============================================================================
# UI CONSTANTS
# =============================================================================

# Window dimensions
DEFAULT_WINDOW_WIDTH: Final[int] = 1200  # [px] Default application window width
DEFAULT_WINDOW_HEIGHT: Final[int] = 800  # [px] Default application window height

# UI spacing and layout
DEFAULT_PADDING: Final[int] = 10  # [px] Default padding for UI elements
DEFAULT_BUTTON_HEIGHT: Final[int] = 40  # [px] Default button height
DEFAULT_TEXT_HEIGHT: Final[int] = 120  # [px] Default text area height
DEFAULT_SEARCH_WIDTH: Final[int] = 28  # [px] Default search entry width

# Grid weights
GRID_WEIGHT_MAIN: Final[int] = 1  # Main grid weight for responsive layout

# =============================================================================
# SIGNAL PROCESSING CONSTANTS
# =============================================================================

# Minimum data requirements
MIN_SIGNAL_DATA_POINTS: Final[int] = 2  # Minimum data points required for processing
MIN_PERIODS_DEFAULT: Final[int] = 1  # Default minimum periods for rolling operations

# Moving average defaults
DEFAULT_MA_WINDOW: Final[int] = 10  # Default moving average window size

# Butterworth filter defaults
DEFAULT_BW_ORDER: Final[int] = 3  # Default Butterworth filter order
DEFAULT_BW_CUTOFF: Final[float] = 0.1  # Default Butterworth filter cutoff frequency
DEFAULT_BW_NYQUIST: Final[float] = 1.0  # Default Nyquist frequency for Butterworth

# Median filter defaults
DEFAULT_MEDIAN_KERNEL: Final[int] = 5  # Default median filter kernel size
MIN_KERNEL_SIZE: Final[int] = 3  # Minimum kernel size for median filter

# Savitzky-Golay filter defaults
DEFAULT_SAVGOL_WINDOW: Final[int] = 11  # Default Savitzky-Golay window size
DEFAULT_SAVGOL_POLYORDER: Final[int] = 2  # Default Savitzky-Golay polynomial order

# Derivative processing
MAX_DERIVATIVE_ORDER: Final[int] = 5  # Maximum derivative order supported

# =============================================================================
# DATA PROCESSING CONSTANTS
# =============================================================================

# Time column detection keywords
TIME_COLUMN_KEYWORDS: Final[tuple[str, ...]] = (
    "time", "timestamp", "date", "datetime", "local_time", "utc_time"
)

# Signal display limits
LARGE_SIGNAL_THRESHOLD: Final[int] = 200  # Threshold for using efficient signal display
SIGNAL_BATCH_SIZE: Final[int] = 200  # Number of signals to display in each batch

# Bulk processing limits
BULK_SAMPLE_SIZE: Final[int] = 3  # Number of files to sample in bulk mode
LARGE_FILE_THRESHOLD: Final[int] = 100  # Threshold for large file handling

# =============================================================================
# PLOTTING CONSTANTS
# =============================================================================

# Plot update scheduling
PLOT_UPDATE_DELAY_MS: Final[int] = 200  # [ms] Delay for plot update coalescing

# Zoom factors
ZOOM_OUT_FACTOR: Final[float] = 1.25  # Zoom out by 25%
ZOOM_IN_FACTOR: Final[float] = 0.75  # Zoom in by 25%

# Plot styling
DEFAULT_LINE_WIDTH: Final[float] = 1.0  # Default line width for plots
DEFAULT_GRID_ALPHA: Final[float] = 0.6  # Default grid transparency
DEFAULT_GRID_LINESTYLE: Final[str] = "--"  # Default grid line style

# =============================================================================
# FILE PROCESSING CONSTANTS
# =============================================================================

# File extensions
SUPPORTED_CSV_EXTENSIONS: Final[tuple[str, ...]] = (".csv", ".txt")
SUPPORTED_EXCEL_EXTENSIONS: Final[tuple[str, ...]] = (".xlsx", ".xls")
SUPPORTED_MATLAB_EXTENSIONS: Final[tuple[str, ...]] = (".mat",)
SUPPORTED_PARQUET_EXTENSIONS: Final[tuple[str, ...]] = (".parquet",)
SUPPORTED_HDF5_EXTENSIONS: Final[tuple[str, ...]] = (".h5", ".hdf5")
SUPPORTED_FEATHER_EXTENSIONS: Final[tuple[str, ...]] = (".feather",)
SUPPORTED_PICKLE_EXTENSIONS: Final[tuple[str, ...]] = (".pkl", ".pickle")

# =============================================================================
# ERROR HANDLING CONSTANTS
# =============================================================================

# Error messages
ERROR_MSG_NO_FILES: Final[str] = "Please select files first before loading signals."
ERROR_MSG_EMPTY_FILE: Final[str] = "The selected file contains no signals."
ERROR_MSG_NO_PLOTS: Final[str] = "No plots to export."

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Default plot settings
DEFAULT_PLOT_TITLE: Final[str] = ""
DEFAULT_PLOT_XLABEL: Final[str] = ""
DEFAULT_PLOT_YLABEL: Final[str] = "Value"
DEFAULT_LEGEND_POSITION: Final[str] = "best"

# Time formatting
DEFAULT_TIME_FORMAT: Final[str] = "%H:%M"  # Default time format for plots

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Numeric validation
MIN_WINDOW_SIZE: Final[int] = 1  # Minimum window size for filters
MAX_WINDOW_SIZE: Final[int] = 1000  # Maximum window size for filters
MIN_CUTOFF_FREQUENCY: Final[float] = 0.001  # Minimum cutoff frequency
MAX_CUTOFF_FREQUENCY: Final[float] = 0.999  # Maximum cutoff frequency

# =============================================================================
# SOURCES AND REFERENCES
# =============================================================================

"""
Sources for constants:

1. UI Constants:
   - Window dimensions: Based on common desktop screen resolutions (1920x1080)
   - Padding values: Standard UI design guidelines for comfortable spacing
   - Button heights: Accessibility guidelines for touch targets

2. Signal Processing Constants:
   - Butterworth filter defaults: Standard signal processing literature
   - Savitzky-Golay defaults: Original paper by Savitzky and Golay (1964)
   - Median filter kernel: Common practice for noise reduction
   - Moving average window: Balance between smoothing and responsiveness

3. Data Processing Constants:
   - Time column keywords: Common naming conventions in data files
   - Signal display thresholds: Performance optimization for large datasets
   - Batch sizes: Memory management for large signal lists

4. Plotting Constants:
   - Zoom factors: Standard zoom increments for data visualization
   - Update delays: UI responsiveness optimization
   - Styling defaults: Matplotlib best practices

5. File Processing Constants:
   - Supported extensions: Industry standard file formats for data analysis

6. Validation Constants:
   - Numeric ranges: Mathematical constraints for signal processing algorithms
   - Frequency limits: Nyquist theorem constraints
"""
