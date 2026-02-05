"""Core data processor modules.

This package provides the shared processing logic for all GUI implementations.
All modules here are UI-agnostic and can be used by TKinter, PyQt6, React, or CLI.

Modules:
    signal_processing: Integration, differentiation, resampling, formulas, trendlines
    config_manager: Application settings persistence
    signal_list_manager: Signal selection persistence
    plot_config_manager: Plot configuration persistence
    dataset_naming: File naming utilities
    dat_importer: DAT/DBF file import
    data_loader: CSV/data file loading
    signal_processor: Filter orchestration
"""

from data_processor.core.config_manager import ConfigManager
from data_processor.core.dat_importer import (
    DBF_AVAILABLE,
    export_dat_to_csv,
    get_dat_columns,
    get_dat_file_info,
    import_dat_with_tags,
    preview_dat_file,
    read_dat_file,
    read_dbf_tags,
)
from data_processor.core.dataset_naming import (
    generate_dataset_name,
    generate_unique_name,
    sanitize_dataset_name,
    validate_dataset_name,
)
from data_processor.core.plot_config_manager import PlotConfigManager
from data_processor.core.signal_list_manager import SignalListManager
from data_processor.core.signal_processing import (
    DifferentiationMethod,
    IntegrationMethod,
    TrendlineType,
    apply_custom_variable,
    calculate_trendline,
    differentiate_signals,
    integrate_signals,
    resample_data,
    trim_time_range,
)

__all__ = [
    # Signal processing
    "integrate_signals",
    "differentiate_signals",
    "resample_data",
    "apply_custom_variable",
    "calculate_trendline",
    "trim_time_range",
    "IntegrationMethod",
    "DifferentiationMethod",
    "TrendlineType",
    # Configuration
    "ConfigManager",
    "SignalListManager",
    "PlotConfigManager",
    # Dataset naming
    "generate_dataset_name",
    "validate_dataset_name",
    "sanitize_dataset_name",
    "generate_unique_name",
    # DAT import
    "read_dat_file",
    "read_dbf_tags",
    "get_dat_columns",
    "import_dat_with_tags",
    "export_dat_to_csv",
    "preview_dat_file",
    "get_dat_file_info",
    "DBF_AVAILABLE",
]
