#!/usr/bin/env python3
"""
Comprehensive plotting diagnostic script for Data Processor
This will test each step of the plotting process to identify where it fails.
"""

import os
import sys
import tkinter as tk

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

print("=" * 60)
print("PLOTTING DIAGNOSTIC SCRIPT")
print("=" * 60)

# Test 1: Basic imports and setup
print("\n1. Testing imports...")
try:
    import Data_Processor_r0

    print("✓ Data_Processor_r0 imported successfully")
except Exception as e:
    print(f"✗ Data_Processor_r0 import failed: {e}")
    sys.exit(1)

# Test 2: Create app instance
print("\n2. Creating Data Processor instance...")
try:
    app = Data_Processor_r0.CSVProcessorApp()
    app.withdraw()  # Hide the main window for testing
    print("✓ CSVProcessorApp instance created")
except Exception as e:
    print(f"✗ DataProcessor creation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Check for sample data files
print("\n3. Checking for sample data files...")
data_dir = "Half Ton Data"
sample_files = []
if os.path.exists(data_dir):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    sample_files = [
        os.path.join(data_dir, f) for f in csv_files[:2]
    ]  # Take first 2 files
    print(f"✓ Found {len(csv_files)} CSV files in {data_dir}")
    for f in csv_files[:3]:
        print(f"  - {f}")
else:
    print(f"✗ {data_dir} directory not found")

if not sample_files:
    print("✗ No sample data files found")
    sys.exit(1)

# Test 4: Load sample file and check data structure
print("\n4. Testing data loading...")
sample_file = sample_files[0]
filename = os.path.basename(sample_file)
print(f"Testing with file: {filename}")

try:
    # Simulate file selection
    app.input_file_paths = [sample_file]

    # Test get_data_for_plotting directly
    print("Testing get_data_for_plotting...")
    df = app.get_data_for_plotting(filename)

    if df is not None and not df.empty:
        print(f"✓ Data loaded successfully. Shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
        print(f"✓ Data types: {df.dtypes.to_dict()}")

        # Check for time columns
        time_cols = [
            col
            for col in df.columns
            if any(word in col.lower() for word in ["time", "date", "timestamp"])
        ]
        print(f"✓ Time columns found: {time_cols}")

    else:
        print("✗ Failed to load data or data is empty")
        sys.exit(1)

except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Check plotting components initialization
print("\n5. Testing plotting components...")
try:
    # Check if plotting components exist
    has_plot_canvas = hasattr(app, "plot_canvas")
    has_plot_ax = hasattr(app, "plot_ax")
    has_plot_fig = hasattr(app, "plot_fig")

    print(f"plot_canvas exists: {has_plot_canvas}")
    print(f"plot_ax exists: {has_plot_ax}")
    print(f"plot_fig exists: {has_plot_fig}")

    if has_plot_canvas and has_plot_ax:
        print("✓ Plotting components initialized")
    else:
        print("✗ Missing plotting components")

        # Try to initialize them manually for testing
        print("Attempting to initialize plotting components manually...")
        app.plot_fig = Figure(figsize=(12, 8))
        app.plot_ax = app.plot_fig.add_subplot(111)
        print("✓ Manual plotting components created")

except Exception as e:
    print(f"✗ Plotting components check failed: {e}")
    import traceback

    traceback.print_exc()

# Test 6: Test signal variable creation
print("\n6. Testing signal variable creation...")
try:
    # Simulate on_plot_file_select behavior
    print(f"Simulating file selection for: {filename}")

    # Initialize plot_signal_vars if not exists
    if not hasattr(app, "plot_signal_vars"):
        app.plot_signal_vars = {}

    # Create signal variables manually
    app.plot_signal_vars = {}
    signal_count = 0
    for signal in df.columns:
        var = tk.BooleanVar(value=False)
        app.plot_signal_vars[signal] = {"var": var, "widget": None}
        signal_count += 1

    print(f"✓ Created {signal_count} signal variables")
    print(f"✓ Signal names: {list(app.plot_signal_vars.keys())[:5]}...")

    # Auto-select first few non-time signals for testing
    selected_count = 0
    for signal in df.columns:
        if selected_count >= 3:  # Select first 3 signals
            break
        if not any(word in signal.lower() for word in ["time", "date", "timestamp"]):
            app.plot_signal_vars[signal]["var"].set(True)
            selected_count += 1
            print(f"✓ Auto-selected signal: {signal}")

    # Check which signals are selected
    selected_signals = [
        s for s, data in app.plot_signal_vars.items() if data["var"].get()
    ]
    print(f"✓ Total selected signals: {len(selected_signals)}")
    print(f"✓ Selected signals: {selected_signals}")

except Exception as e:
    print(f"✗ Signal variable creation failed: {e}")
    import traceback

    traceback.print_exc()

# Test 7: Test update_plot function
print("\n7. Testing update_plot function...")
try:
    # Set up required variables
    if not hasattr(app, "plot_file_menu"):
        app.plot_file_menu = ctk.CTkOptionMenu(app, values=[filename])
        app.plot_file_menu.set(filename)

    if not hasattr(app, "plot_xaxis_menu"):
        app.plot_xaxis_menu = ctk.CTkOptionMenu(app, values=list(df.columns))
        app.plot_xaxis_menu.set(df.columns[0])  # Set to first column (usually time)

    # Set up other required variables with defaults
    if not hasattr(app, "show_both_signals_var"):
        app.show_both_signals_var = tk.BooleanVar(value=False)

    if not hasattr(app, "plot_filter_type"):
        app.plot_filter_type = tk.StringVar(value="None")

    if not hasattr(app, "plot_type_var"):
        app.plot_type_var = tk.StringVar(value="Line")

    if not hasattr(app, "line_width_var"):
        app.line_width_var = tk.StringVar(value="1.0")

    if not hasattr(app, "color_scheme_var"):
        app.color_scheme_var = tk.StringVar(value="Auto (Matplotlib)")

    if not hasattr(app, "custom_colors"):
        app.custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if not hasattr(app, "custom_legend_entries"):
        app.custom_legend_entries = {}

    if not hasattr(app, "trendline_type_var"):
        app.trendline_type_var = tk.StringVar(value="None")

    if not hasattr(app, "trendline_signal_var"):
        app.trendline_signal_var = tk.StringVar(value="Select signal...")

    if not hasattr(app, "plot_title_entry"):
        app.plot_title_entry = ctk.CTkEntry(app)

    if not hasattr(app, "plot_xlabel_entry"):
        app.plot_xlabel_entry = ctk.CTkEntry(app)

    if not hasattr(app, "plot_ylabel_entry"):
        app.plot_ylabel_entry = ctk.CTkEntry(app)

    if not hasattr(app, "legend_position_var"):
        app.legend_position_var = tk.StringVar(value="best")

    if not hasattr(app, "status_label"):
        app.status_label = ctk.CTkLabel(app, text="Ready")

    print("✓ All required variables set up")

    # Now test the actual update_plot function
    print("Calling update_plot...")
    app.update_plot()

    print("✓ update_plot completed without errors")

    # Check if plot was actually created
    if hasattr(app, "plot_ax") and len(app.plot_ax.lines) > 0:
        print(f"✓ Plot created successfully with {len(app.plot_ax.lines)} lines")
    else:
        print(
            "⚠ No plot lines created (this might be expected if no signals were selected)"
        )

except Exception as e:
    print(f"✗ update_plot failed: {e}")
    import traceback

    traceback.print_exc()

# Test 8: Test manual plotting with known data
print("\n8. Testing manual plotting with sample data...")
try:
    # Create a simple test plot manually
    test_fig = Figure(figsize=(10, 6))
    test_ax = test_fig.add_subplot(111)

    # Get time column and a numeric column
    time_col = df.columns[0]
    numeric_cols = [
        col for col in df.columns[1:3] if df[col].dtype in ["float64", "int64"]
    ]

    if numeric_cols:
        test_signal = numeric_cols[0]
        print(f"Creating test plot with time='{time_col}' and signal='{test_signal}'")

        # Plot the data
        plot_data = df[[time_col, test_signal]].dropna()
        test_ax.plot(plot_data[time_col], plot_data[test_signal], label=test_signal)
        test_ax.set_xlabel(time_col)
        test_ax.set_ylabel(test_signal)
        test_ax.set_title(f"Test Plot: {test_signal} vs {time_col}")
        test_ax.legend()
        test_ax.grid(True)

        print("✓ Manual test plot created successfully")
        print(f"✓ Plot has {len(plot_data)} data points")

    else:
        print("✗ No numeric columns found for plotting")

except Exception as e:
    print(f"✗ Manual plotting test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 9: Check for missing methods or attributes
print("\n9. Checking for missing critical methods/attributes...")
critical_methods = [
    "get_data_for_plotting",
    "update_plot",
    "on_plot_file_select",
    "_ensure_data_loaded",
    "_apply_plot_filter",
]

missing_methods = []
for method in critical_methods:
    if hasattr(app, method):
        print(f"✓ {method} exists")
    else:
        print(f"✗ {method} missing")
        missing_methods.append(method)

critical_attributes = ["processed_files", "loaded_data_cache", "input_file_paths"]

missing_attributes = []
for attr in critical_attributes:
    if hasattr(app, attr):
        print(f"✓ {attr} exists: {type(getattr(app, attr))}")
    else:
        print(f"✗ {attr} missing")
        missing_attributes.append(attr)

# Summary
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)

if missing_methods:
    print(f"✗ Missing methods: {missing_methods}")
else:
    print("✓ All critical methods present")

if missing_attributes:
    print(f"✗ Missing attributes: {missing_attributes}")
else:
    print("✓ All critical attributes present")

print(f"\nData file tested: {filename}")
print(f"Data shape: {df.shape}")
print(f"Data columns: {len(df.columns)}")
print(
    f"Signal variables created: {len(app.plot_signal_vars) if hasattr(app, 'plot_signal_vars') else 0}"
)

selected_signals = []
if hasattr(app, "plot_signal_vars"):
    selected_signals = [
        s for s, data in app.plot_signal_vars.items() if data["var"].get()
    ]
print(f"Selected signals: {len(selected_signals)}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

if len(selected_signals) == 0:
    print("1. CRITICAL: No signals are being selected for plotting")
    print("   - Check signal checkbox creation and event binding")
    print("   - Verify that plot_signal_vars are properly initialized")

if not hasattr(app, "plot_canvas") or not hasattr(app, "plot_ax"):
    print("2. CRITICAL: Plot canvas or axes not initialized")
    print("   - Check create_plotting_tab method")
    print("   - Verify matplotlib integration")

print("3. Next steps:")
print("   - Run the actual GUI and check console output")
print("   - Test file selection in plotting tab")
print("   - Manually select signals and check if update_plot is called")

print("\nTest completed. Check the debug output above for specific issues.")

# Clean up
try:
    app.destroy()
except:
    pass
