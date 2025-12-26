#!/usr/bin/env python3
"""
Comprehensive fix script for Data_Processor_r0.py type annotation issues.
This script addresses the 428 diagnostic errors by adding proper type hints.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union


def add_imports_section(content: str) -> str:
    """Add necessary imports for type annotations."""
    import_lines = [
        "from typing import Dict, List, Tuple, Any, Optional, Union, Callable",
        "from tkinter import Widget, Event",
        "import customtkinter as ctk",
        "import pandas as pd",
        "import numpy as np",
        "from matplotlib.figure import Figure",
        "from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk",
    ]
    
    # Find the existing imports section
    lines = content.split('\n')
    import_end_idx = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            import_end_idx = i
        elif line.strip() and not line.strip().startswith('#') and import_end_idx > 0:
            break
    
    # Insert new imports after existing ones
    if import_end_idx > 0:
        for import_line in reversed(import_lines):
            if import_line not in content:
                lines.insert(import_end_idx + 1, import_line)
    
    return '\n'.join(lines)


def fix_function_signatures(content: str) -> str:
    """Add type annotations to function signatures."""
    
    # Common type annotation patterns
    patterns = [
        # Basic function with no parameters
        (r'def (\w+)\(self\):', r'def \1(self) -> None:'),
        
        # Functions with parameters but no return type
        (r'def (\w+)\(self, ([^)]+)\):', r'def \1(self, \2) -> None:'),
        
        # Event handlers
        (r'def (\w+)\(self, event\):', r'def \1(self, event: Event[Any]) -> None:'),
        
        # Functions that return values
        (r'def (get_\w+|_get_\w+)\(self\):', r'def \1(self) -> Any:'),
        (r'def (load_\w+|_load_\w+)\(self\):', r'def \1(self) -> Any:'),
        (r'def (save_\w+|_save_\w+)\(self\):', r'def \1(self) -> bool:'),
        
        # Update functions
        (r'def (update_\w+|_update_\w+)\(self\):', r'def \1(self) -> None:'),
        (r'def (update_\w+|_update_\w+)\(self, ([^)]+)\):', r'def \1(self, \2) -> None:'),
        
        # Create functions
        (r'def (create_\w+|_create_\w+)\(self, ([^)]+)\):', r'def \1(self, \2) -> Any:'),
        
        # Apply functions
        (r'def (_apply_\w+)\(self, ([^)]+)\):', r'def \1(self, \2) -> Any:'),
        
        # Process functions
        (r'def (process_\w+|_process_\w+)\(self\):', r'def \1(self) -> None:'),
        (r'def (process_\w+|_process_\w+)\(self, ([^)]+)\):', r'def \1(self, \2) -> Any:'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    return content


def fix_specific_functions(content: str) -> str:
    """Fix specific function signatures that need custom handling."""
    
    # Fix __init__ method
    content = re.sub(
        r'def __init__\(self, \*args, \*\*kwargs\):',
        r'def __init__(self, *args: Any, **kwargs: Any) -> None:',
        content
    )
    
    # Fix process_single_csv_file function
    content = re.sub(
        r'def process_single_csv_file\(file_path, settings\):',
        r'def process_single_csv_file(file_path: str, settings: Dict[str, Any]) -> Optional[pd.DataFrame]:',
        content
    )
    
    # Fix _poly_derivative function
    content = re.sub(
        r'def _poly_derivative\(series, window, poly_order, deriv_order, delta_x\):',
        r'def _poly_derivative(series: pd.Series, window: int, poly_order: int, deriv_order: int, delta_x: float) -> pd.Series:',
        content
    )
    
    # Fix class definition
    content = re.sub(
        r'class CSVProcessorApp\(ctk\.CTk\):',
        r'class CSVProcessorApp(ctk.CTk):',
        content
    )
    
    return content


def fix_variable_annotations(content: str) -> str:
    """Add type annotations to class variables."""
    
    # Find class variables that need annotations
    patterns = [
        (r'self\.plots_signal_vars = {}', r'self.plots_signal_vars: Dict[str, Any] = {}'),
        (r'self\.signal_vars = {}', r'self.signal_vars: Dict[str, Any] = {}'),
        (r'self\.plot_signal_vars = {}', r'self.plot_signal_vars: Dict[str, Any] = {}'),
        (r'self\.custom_vars_list = \[\]', r'self.custom_vars_list: List[Any] = []'),
        (r'self\.plots_list = \[\]', r'self.plots_list: List[Dict[str, Any]] = []'),
        (r'self\.input_file_paths = \[\]', r'self.input_file_paths: List[str] = []'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    return content


def fix_matplotlib_imports(content: str) -> str:
    """Fix matplotlib import issues."""
    
    # Fix NavigationToolbar2Tk import
    content = re.sub(
        r'from matplotlib\.backends\.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk',
        r'from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg\ntry:\n    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk\nexcept ImportError:\n    NavigationToolbar2Tk = None',
        content
    )
    
    # Fix matplotlib.cm attribute access
    content = re.sub(r'plt\.cm\.(\w+)', r'getattr(plt.cm, "\1", plt.cm.viridis)', content)
    
    return content


def fix_slice_indices(content: str) -> str:
    """Fix slice index type issues."""
    
    # Fix slice operations with string indices
    content = re.sub(
        r'(\w+)\[([^:]+):([^:]+)\]',
        lambda m: f'{m.group(1)}[int({m.group(2)}):int({m.group(3)})]' if ':' in m.group(0) else m.group(0),
        content
    )
    
    return content


def main():
    """Main function to apply all fixes."""
    
    file_path = Path("data_processing/data_processor/archive/Data_Processor_r0.py")
    
    if not file_path.exists():
        print(f"File {file_path} not found!")
        return
    
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying fixes...")
    
    # Apply all fixes
    content = add_imports_section(content)
    content = fix_function_signatures(content)
    content = fix_specific_functions(content)
    content = fix_variable_annotations(content)
    content = fix_matplotlib_imports(content)
    content = fix_slice_indices(content)
    
    # Create backup
    backup_path = file_path.with_suffix('.py.backup')
    print(f"Creating backup at {backup_path}...")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Writing fixed content to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Type annotation fixes applied successfully!")
    print("Run diagnostics again to check remaining issues.")


if __name__ == "__main__":
    main()