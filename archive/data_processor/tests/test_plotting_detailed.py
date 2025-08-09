#!/usr/bin/env python3
"""
Test the plotting functionality step by step
Run this while the main app is running to test plotting components
"""

import os
import sys
import tkinter as tk

import numpy as np
import pandas as pd

print("=" * 60)
print("PLOTTING FUNCTIONALITY TEST")
print("=" * 60)

# Test 1: Check if we can load sample data
print("\n1. Testing data loading...")
data_dir = "Half Ton Data"
if os.path.exists(data_dir):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if csv_files:
        sample_file = os.path.join(data_dir, csv_files[0])
        filename = csv_files[0]
        print(f"✓ Found sample file: {filename}")

        try:
            df = pd.read_csv(sample_file, low_memory=False)
            print(f"✓ Data loaded successfully: {df.shape}")
            print(f"✓ Columns: {list(df.columns[:5])}")

            # Check for time columns
            time_cols = [
                col
                for col in df.columns
                if any(word in col.lower() for word in ["time", "date", "timestamp"])
            ]
            print(f"✓ Time columns: {time_cols}")

            # Check for numeric columns
            numeric_cols = [
                col for col in df.columns if df[col].dtype in ["float64", "int64"]
            ]
            print(
                f"✓ Numeric columns: {len(numeric_cols)} (showing first 5: {numeric_cols[:5]})"
            )

        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            sys.exit(1)
    else:
        print("✗ No CSV files found")
        sys.exit(1)
else:
    print(f"✗ Directory {data_dir} not found")
    sys.exit(1)

# Test 2: Test signal selection logic
print("\n2. Testing signal selection logic...")
try:
    # Simulate plot_signal_vars creation
    plot_signal_vars = {}
    for signal in df.columns:
        var = tk.BooleanVar(value=False)
        plot_signal_vars[signal] = {"var": var, "widget": None}

    print(f"✓ Created {len(plot_signal_vars)} signal variables")

    # Test auto-selection logic
    common_signals = [
        "co_pct",
        "co2_pct",
        "h2_pct",
        "ch4_pct",
        "o2_pct",
        "n2_pct",
        "temp",
        "temperature",
        "pressure",
        "flow",
        "level",
    ]
    auto_selected = 0
    for signal in df.columns:
        if auto_selected >= 4:
            break
        signal_lower = signal.lower()
        if any(common in signal_lower for common in common_signals):
            plot_signal_vars[signal]["var"].set(True)
            auto_selected += 1
            print(f"✓ Auto-selected: {signal}")

    # Force selection if no auto-selection worked
    if auto_selected == 0:
        print("No auto-selection worked, using fallback...")
        for signal in df.columns:
            if not any(
                word in signal.lower() for word in ["time", "date", "timestamp"]
            ):
                if df[signal].dtype in ["float64", "int64"]:
                    plot_signal_vars[signal]["var"].set(True)
                    print(f"✓ Fallback selected numeric: {signal}")
                    break

    # Check final selection
    selected_signals = [s for s, data in plot_signal_vars.items() if data["var"].get()]
    print(f"✓ Final selected signals: {selected_signals}")

    if not selected_signals:
        print("✗ NO SIGNALS SELECTED - This is a critical issue!")
    else:
        print(f"✓ Signal selection working: {len(selected_signals)} signals selected")

except Exception as e:
    print(f"✗ Signal selection test failed: {e}")

# Test 3: Test matplotlib plotting
print("\n3. Testing matplotlib plotting...")
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    print("✓ Matplotlib imports successful")

    # Create test plot
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    if selected_signals and len(selected_signals) > 0:
        time_col = df.columns[0]  # Assume first column is time
        test_signal = selected_signals[0]

        # Create test plot
        plot_data = df[[time_col, test_signal]].dropna()
        ax.plot(plot_data[time_col], plot_data[test_signal], label=test_signal)
        ax.set_xlabel(time_col)
        ax.set_ylabel(test_signal)
        ax.set_title(f"Test Plot: {test_signal}")
        ax.legend()
        ax.grid(True)

        print(f"✓ Test plot created with {len(plot_data)} data points")
        print(f"✓ X-axis: {time_col}, Y-axis: {test_signal}")
    else:
        ax.text(0.5, 0.5, "No signals selected for plotting", ha="center", va="center")
        print("⚠ No signals to plot - created empty plot")

except Exception as e:
    print(f"✗ Matplotlib test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Check if Data Processor is running
print("\n4. Checking if Data Processor application is accessible...")
try:
    # Try to import and check if instance exists
    import Data_Processor_r0

    print("✓ Data_Processor_r0 module accessible")

    # Check if matplotlib backend is working
    import matplotlib

    print(f"✓ Matplotlib backend: {matplotlib.get_backend()}")

except Exception as e:
    print(f"✗ Data Processor access failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)

print(f"Data loading: ✓ Working ({df.shape})")
print(f"Signal selection: {'✓ Working' if selected_signals else '✗ FAILED'}")
print(
    f"Selected signals: {len(selected_signals) if 'selected_signals' in locals() else 0}"
)
print(f"Matplotlib: ✓ Available")

print("\nRECOMMENDATIONS:")
if not selected_signals:
    print("1. CRITICAL: Signal selection is not working")
    print("   - Check if plot_signal_vars is being created properly")
    print("   - Verify checkbox creation and event binding")
    print("   - Add debugging to on_plot_file_select method")
else:
    print("1. Signal selection appears to be working")

print("2. Check if plotting tab is calling update_plot() correctly")
print("3. Verify that plot_canvas and plot_ax are properly initialized")
print("4. Use the debug output from the main application")

print("\nNext steps:")
print("1. Go to plotting tab in the main application")
print("2. Select a file and observe console output")
print("3. Check if signals appear as checkboxes")
print("4. Try clicking the '🔄 Update Plot' button")
print("5. Look for detailed debug output in console")

print("\nTest completed.")
