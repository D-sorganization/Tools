"""
Constants for the Data Processor application.
"""

# File processing constants
MAX_FILE_SIZE_MB = 500
CHUNK_SIZE = 10000
DEFAULT_ENCODING = "utf-8"

# UI constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FONT_SIZE = 12

# Processing constants
DEFAULT_SAMPLE_RATE = 1000
MAX_PLOT_POINTS = 10000

# Export constants
DEFAULT_DPI = 300
SUPPORTED_FORMATS = [".csv", ".xlsx", ".json", ".parquet"]

# Logging constants
LOG_LEVEL = "INFO"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
