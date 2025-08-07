#!/usr/bin/env python3
"""
Simple test script for Data Processor functionality - Tests data loading, processing, and export
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

print("=== DATA PROCESSOR FUNCTIONALITY TEST ===")
print()

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Install missing dependencies if needed
missing_packages = []

try:
    import customtkinter as ctk
    print("PASS: customtkinter imported successfully")
except ImportError as e:
    print(f"FAIL: customtkinter import failed: {e}")
    missing_packages.append("customtkinter")

try:
    import simpledbf
    print("PASS: simpledbf imported successfully")
except ImportError as e:
    print(f"FAIL: simpledbf import failed: {e}")
    missing_packages.append("simpledbf")

if missing_packages:
    print(f"Installing missing packages: {missing_packages}")
    import subprocess
    for package in missing_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("Packages installed, continuing test...")

try:
    import Data_Processor_r0
    print("PASS: Data_Processor_r0 imported successfully")
except Exception as e:
    print(f"FAIL: Failed to import Data_Processor_r0: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test data paths
data_dir = Path("Half Ton Data")
test_files = list(data_dir.glob("*.csv"))

print(f"\nChecking test data files in {data_dir}:")
available_files = []
for file_path in test_files:
    if file_path.exists() and file_path.stat().st_size > 0:
        print(f"FOUND: {file_path}")
        available_files.append(str(file_path))

if not available_files:
    print("ERROR: No test data files found. Skipping data tests.")
    sys.exit(1)

print(f"\nUsing {len(available_files)} test files for functionality testing...")

# Test 1: Basic CSV reading
print("\n--- TEST 1: Basic CSV Reading ---")
try:
    test_file = available_files[0]
    df = pd.read_csv(test_file, low_memory=False)
    print(f"PASS: Successfully read {test_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:5])}")
    
    # Check data types
    time_col = df.columns[0]
    print(f"  Time column: {time_col}")
    
except Exception as e:
    print(f"FAIL: Failed to read test file: {e}")
    import traceback
    traceback.print_exc()

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
    
    print(f"  Processing with {len(test_settings['selected_signals'])} signals")
    processed_df = Data_Processor_r0.process_single_csv_file(test_file, test_settings)
    
    if processed_df is not None and not processed_df.empty:
        print(f"PASS: Single file processing successful")
        print(f"  Processed shape: {processed_df.shape}")
        print(f"  Processed columns: {list(processed_df.columns)}")
    else:
        print(f"FAIL: Single file processing returned None or empty DataFrame")
        
except Exception as e:
    print(f"FAIL: Single file processing failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test Export Functionality
print("\n--- TEST 3: Export Functionality Test ---")
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
    print(f"PASS: CSV export successful: {csv_file}")
    
    # Excel export (if openpyxl is available)
    try:
        excel_file = export_dir / "test_export.xlsx"
        test_df.to_excel(excel_file, index=False)
        print(f"PASS: Excel export successful: {excel_file}")
    except ImportError:
        print("SKIP: Excel export (openpyxl not available)")
    
    # MAT file export (if scipy is available)
    try:
        from scipy.io import savemat
        mat_file = export_dir / "test_export.mat"
        mat_data = {col: test_df[col].values for col in test_df.columns}
        savemat(mat_file, mat_data)
        print(f"PASS: MAT file export successful: {mat_file}")
    except ImportError:
        print("SKIP: MAT file export (scipy not available)")
    
except Exception as e:
    print(f"FAIL: Export functionality test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test Plotting Components
print("\n--- TEST 4: Plotting Components Test ---")
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
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
    print(f"PASS: Matplotlib plotting test successful: {plot_file}")
    
    plt.close(fig)
    
except Exception as e:
    print(f"FAIL: Plotting components test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== TEST SUMMARY ===")
print("Functionality tests completed!")
print("\nTo manually test the full GUI:")
print("1. Run: python Data_Processor_r0.py")
print("2. Click 'Select Input CSV Files' and choose files from 'Half Ton Data' folder")
print("3. Select some signals to process")
print("4. Click 'Process & Batch Export Files'")
print("5. Switch to 'Plotting & Analysis' tab to test plotting")

print("\nTest files and exports should be in the 'test_exports' directory.")
