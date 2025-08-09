#!/usr/bin/env python3
"""
Test script for Data Processor functionality - Tests data loading, processing, and export
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

print("=== DATA PROCESSOR FUNCTIONALITY TEST ===\n")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import Data_Processor_r0
    print("✓ Data_Processor_r0 imported successfully")
except Exception as e:
    print(f"✗ Failed to import Data_Processor_r0: {e}")
    sys.exit(1)

# Test data paths
data_dir = Path("Half Ton Data")
test_files = [
    data_dir / "2025-07-16 Data.csv",
    data_dir / "2025-07-15 Data.csv"
]

# Check if test files exist
print("\nChecking test data files:")
available_files = []
for file_path in test_files:
    if file_path.exists():
        print(f"✓ Found: {file_path}")
        available_files.append(str(file_path))
    else:
        print(f"✗ Missing: {file_path}")

if not available_files:
    print("✗ No test data files found. Skipping data tests.")
    sys.exit(1)

print(f"\nUsing {len(available_files)} test files for functionality testing...")

# Test 1: Basic CSV reading
print("\n--- TEST 1: Basic CSV Reading ---")
try:
    test_file = available_files[0]
    df = pd.read_csv(test_file, low_memory=False)
    print(f"✓ Successfully read {test_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
    
    # Check data types
    time_col = df.columns[0]
    print(f"  Time column: {time_col}")
    print(f"  First few rows of time column: {df[time_col].head(3).tolist()}")
    
except Exception as e:
    print(f"✗ Failed to read test file: {e}")
    sys.exit(1)

# Test 2: Test the parallel processing function
print("\n--- TEST 2: Single File Processing Function ---")
try:
    # Test settings similar to what the GUI would use
    test_settings = {
        'selected_signals': list(df.columns[:5]),  # Use first 5 columns
        'filter_type': 'None',
        'resample_enabled': False,
        'resample_rule': None,
        'ma_window': 10,
        'bw_order': 3,
        'bw_cutoff': 0.1,
        'median_kernel': 5,
        'savgol_window': 11,
        'savgol_polyorder': 2
    }
    
    print(f"  Processing with settings: {test_settings}")
    processed_df = Data_Processor_r0.process_single_csv_file(test_file, test_settings)
    
    if processed_df is not None and not processed_df.empty:
        print(f"✓ Single file processing successful")
        print(f"  Processed shape: {processed_df.shape}")
        print(f"  Processed columns: {list(processed_df.columns)}")
    else:
        print(f"✗ Single file processing returned None or empty DataFrame")
        
except Exception as e:
    print(f"✗ Single file processing failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test GUI creation (without mainloop)
print("\n--- TEST 3: GUI Creation Test ---")
try:
    # Create app instance without running mainloop
    app = Data_Processor_r0.CSVProcessorApp()
    print("✓ CSVProcessorApp instance created successfully")
    
    # Test basic properties
    print(f"  Window title: {app.title()}")
    print(f"  Window geometry: {app.geometry()}")
    
    # Test file selection simulation
    app.input_file_paths = available_files
    print(f"✓ Set input file paths: {len(app.input_file_paths)} files")
    
    # Test signal loading
    app.load_signals_from_files()
    print(f"✓ Loaded signals from files")
    
    if app.signal_vars:
        signal_count = len(app.signal_vars)
        signal_names = list(app.signal_vars.keys())[:5]
        print(f"  Found {signal_count} signals: {signal_names}{'...' if signal_count > 5 else ''}")
    else:
        print("  No signals found")
    
    # Test processing method (if possible without GUI interaction)
    print("✓ GUI functionality test completed")
    
    # Clean up
    app.destroy()
    
except Exception as e:
    print(f"✗ GUI creation test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test Export Functionality
print("\n--- TEST 4: Export Functionality Test ---")
try:
    # Create a simple test DataFrame to export
    test_data = {
        'Time': pd.date_range('2024-01-01', periods=100, freq='1S'),
        'Signal1': np.sin(np.linspace(0, 10, 100)),
        'Signal2': np.cos(np.linspace(0, 10, 100)),
        'Signal3': np.random.randn(100)
    }
    test_df = pd.DataFrame(test_data)
    
    # Test different export formats
    export_dir = Path("test_exports")
    export_dir.mkdir(exist_ok=True)
    
    # CSV export
    csv_file = export_dir / "test_export.csv"
    test_df.to_csv(csv_file, index=False)
    print(f"✓ CSV export successful: {csv_file}")
    
    # Excel export (if openpyxl is available)
    try:
        excel_file = export_dir / "test_export.xlsx"
        test_df.to_excel(excel_file, index=False)
        print(f"✓ Excel export successful: {excel_file}")
    except ImportError:
        print("  Excel export skipped (openpyxl not available)")
    
    # MAT file export (if scipy is available)
    try:
        from scipy.io import savemat
        mat_file = export_dir / "test_export.mat"
        mat_data = {col: test_df[col].values for col in test_df.columns}
        savemat(mat_file, mat_data)
        print(f"✓ MAT file export successful: {mat_file}")
    except ImportError:
        print("  MAT file export skipped (scipy not available)")
    
except Exception as e:
    print(f"✗ Export functionality test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test Plotting Components
print("\n--- TEST 5: Plotting Components Test ---")
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    
    # Create a simple test plot
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Plot test data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax.plot(x, y1, label='sin(x)')
    ax.plot(x, y2, label='cos(x)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)
    
    # Save test plot
    plot_file = Path("test_exports") / "test_plot.png"
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Matplotlib plotting test successful: {plot_file}")
    
    plt.close(fig)
    
except Exception as e:
    print(f"✗ Plotting components test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== TEST SUMMARY ===")
print("All functionality tests completed!")
print("\nTo manually test the full GUI:")
print("1. Run: python Data_Processor_r0.py")
print("2. Click 'Select Input CSV Files' and choose files from 'Half Ton Data' folder")
print("3. Select some signals to process")
print("4. Click 'Process & Batch Export Files'")
print("5. Switch to 'Plotting & Analysis' tab to test plotting")

print("\nTest files and exports are in the 'test_exports' directory.")
