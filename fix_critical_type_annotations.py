#!/usr/bin/env python3
"""
Targeted fix script for the most critical type annotation issues.
This script focuses on fixing the most impactful errors first.
"""

import re
from pathlib import Path


def fix_critical_function_signatures(content: str) -> str:
    """Fix the most critical function signatures."""
    
    # Fix functions that are missing basic type annotations
    fixes = [
        # Helper function
        (r'def _poly_derivative\(series, window, poly_order, deriv_order, delta_x\):', 
         r'def _poly_derivative(series: pd.Series, window: int, poly_order: int, deriv_order: int, delta_x: float) -> pd.Series:'),
        
        # Basic UI functions
        (r'def create_setup_and_process_tab\(self, parent_tab\):', 
         r'def create_setup_and_process_tab(self, parent_tab: Any) -> None:'),
        (r'def create_plotting_tab\(self, parent_tab\):', 
         r'def create_plotting_tab(self, parent_tab: Any) -> None:'),
        (r'def create_plots_list_tab\(self, parent_tab\):', 
         r'def create_plots_list_tab(self, parent_tab: Any) -> None:'),
        (r'def create_dat_import_tab\(self, parent_tab\):', 
         r'def create_dat_import_tab(self, parent_tab: Any) -> None:'),
        (r'def create_help_tab\(self, parent_tab\):', 
         r'def create_help_tab(self, parent_tab: Any) -> None:'),
        (r'def create_status_bar\(self\):', 
         r'def create_status_bar(self) -> None:'),
        
        # Load/save functions
        (r'def _load_layout_config\(self\):', 
         r'def _load_layout_config(self) -> Dict[str, Any]:'),
        (r'def _load_plots_from_file\(self\):', 
         r'def _load_plots_from_file(self) -> None:'),
        (r'def _save_layout_config\(self\):', 
         r'def _save_layout_config(self) -> None:'),
        
        # Update functions
        (r'def update_file_list\(self\):', 
         r'def update_file_list(self) -> None:'),
        (r'def update_signal_list\(self\):', 
         r'def update_signal_list(self) -> None:'),
        (r'def update_plot\(self\):', 
         r'def update_plot(self) -> None:'),
        
        # Process functions
        (r'def process_files\(self\):', 
         r'def process_files(self) -> None:'),
        (r'def _process_single_file\(self, ([^)]+)\):', 
         r'def _process_single_file(self, \1) -> Optional[pd.DataFrame]:'),
        
        # Event handlers
        (r'def select_files\(self\):', 
         r'def select_files(self) -> None:'),
        (r'def select_output_folder\(self\):', 
         r'def select_output_folder(self) -> None:'),
        (r'def select_all\(self\):', 
         r'def select_all(self) -> None:'),
        (r'def deselect_all\(self\):', 
         r'def deselect_all(self) -> None:'),
        
        # Get functions that return values
        (r'def get_data_for_plotting\(self\):', 
         r'def get_data_for_plotting(self) -> Optional[pd.DataFrame]:'),
        (r'def _get_resample_rule\(self\):', 
         r'def _get_resample_rule(self) -> Optional[str]:'),
        
        # Apply functions
        (r'def _apply_custom_variables\(self, ([^)]+)\):', 
         r'def _apply_custom_variables(self, \1) -> pd.DataFrame:'),
        (r'def _apply_integration\(self, ([^)]+)\):', 
         r'def _apply_integration(self, \1) -> pd.DataFrame:'),
        (r'def _apply_differentiation\(self, ([^)]+)\):', 
         r'def _apply_differentiation(self, \1) -> pd.DataFrame:'),
        (r'def _apply_sorting\(self, ([^)]+)\):', 
         r'def _apply_sorting(self, \1) -> pd.DataFrame:'),
        (r'def _apply_plot_filter\(self, ([^)]+)\):', 
         r'def _apply_plot_filter(self, \1) -> pd.Series:'),
        
        # Export functions
        (r'def _export_processed_files\(self\):', 
         r'def _export_processed_files(self) -> None:'),
        (r'def _export_csv_separate\(self, ([^)]+)\):', 
         r'def _export_csv_separate(self, \1) -> None:'),
        (r'def _export_csv_compiled\(self, ([^)]+)\):', 
         r'def _export_csv_compiled(self, \1) -> None:'),
        (r'def _export_excel_multisheet\(self, ([^)]+)\):', 
         r'def _export_excel_multisheet(self, \1) -> None:'),
        (r'def _export_excel_separate\(self, ([^)]+)\):', 
         r'def _export_excel_separate(self, \1) -> None:'),
        (r'def _export_mat_separate\(self, ([^)]+)\):', 
         r'def _export_mat_separate(self, \1) -> None:'),
        (r'def _export_mat_compiled\(self, ([^)]+)\):', 
         r'def _export_mat_compiled(self, \1) -> None:'),
        
        # Check functions
        (r'def _check_file_overwrite\(self, ([^)]+)\):', 
         r'def _check_file_overwrite(self, \1) -> bool:'),
        
        # UI creation functions
        (r'def _create_splitter\(self, ([^)]+)\):', 
         r'def _create_splitter(self, \1) -> Any:'),
        (r'def _create_ma_param_frame\(self, ([^)]+)\):', 
         r'def _create_ma_param_frame(self, \1) -> Tuple[Any, Any, Any]:'),
        (r'def _create_bw_param_frame\(self, ([^)]+)\):', 
         r'def _create_bw_param_frame(self, \1) -> Tuple[Any, Any, Any]:'),
        (r'def _create_median_param_frame\(self, ([^)]+)\):', 
         r'def _create_median_param_frame(self, \1) -> Tuple[Any, Any]:'),
        (r'def _create_hampel_param_frame\(self, ([^)]+)\):', 
         r'def _create_hampel_param_frame(self, \1) -> Tuple[Any, Any, Any]:'),
        (r'def _create_zscore_param_frame\(self, ([^)]+)\):', 
         r'def _create_zscore_param_frame(self, \1) -> Tuple[Any, Any, Any]:'),
        (r'def _create_savgol_param_frame\(self, ([^)]+)\):', 
         r'def _create_savgol_param_frame(self, \1) -> Tuple[Any, Any, Any]:'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    return content


def fix_matplotlib_issues(content: str) -> str:
    """Fix matplotlib-related issues."""
    
    # Fix matplotlib.cm attribute access
    cm_fixes = [
        (r'plt\.cm\.tab10', r'getattr(plt.cm, "tab10", plt.cm.viridis)'),
        (r'plt\.cm\.viridis', r'getattr(plt.cm, "viridis", plt.cm.viridis)'),
        (r'plt\.cm\.plasma', r'getattr(plt.cm, "plasma", plt.cm.viridis)'),
        (r'plt\.cm\.cool', r'getattr(plt.cm, "cool", plt.cm.viridis)'),
        (r'plt\.cm\.autumn', r'getattr(plt.cm, "autumn", plt.cm.viridis)'),
        (r'plt\.cm\.rainbow', r'getattr(plt.cm, "rainbow", plt.cm.viridis)'),
        (r'plt\.cm\.Set1', r'getattr(plt.cm, "Set1", plt.cm.viridis)'),
    ]
    
    for pattern, replacement in cm_fixes:
        content = re.sub(pattern, replacement, content)
    
    return content


def fix_variable_annotations(content: str) -> str:
    """Fix critical variable annotations."""
    
    # Fix the plots_signal_vars annotation
    content = re.sub(
        r'plots_signal_vars = {}',
        r'plots_signal_vars: Dict[str, Any] = {}',
        content
    )
    
    return content


def fix_slice_issues(content: str) -> str:
    """Fix slice index issues."""
    
    # Fix specific slice operations that are causing issues
    content = re.sub(
        r'(\w+)\[([^:]+):([^:]+)\]',
        lambda m: f'{m.group(1)}[int({m.group(2)}):int({m.group(3)})]' 
        if ':' in m.group(0) and not m.group(2).isdigit() and not m.group(3).isdigit()
        else m.group(0),
        content
    )
    
    return content


def main():
    """Apply targeted fixes to the most critical issues."""
    
    file_path = Path("data_processing/data_processor/archive/Data_Processor_r0.py")
    
    if not file_path.exists():
        print(f"File {file_path} not found!")
        return
    
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying critical fixes...")
    
    # Apply fixes in order of importance
    content = fix_critical_function_signatures(content)
    content = fix_matplotlib_issues(content)
    content = fix_variable_annotations(content)
    content = fix_slice_issues(content)
    
    print(f"Writing fixed content to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Critical type annotation fixes applied!")
    print("Run diagnostics to check progress.")


if __name__ == "__main__":
    main()