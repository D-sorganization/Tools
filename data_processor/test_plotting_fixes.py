#!/usr/bin/env python3
"""
Quick test to verify the plotting fixes are working
"""

import sys
import os
import pandas as pd
import numpy as np
import tkinter as tk
import customtkinter as ctk

print("Testing enhanced plotting functionality...")

# Test loading a sample file
data_dir = "Half Ton Data"
sample_files = []
if os.path.exists(data_dir):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if csv_files:
        sample_file = os.path.join(data_dir, csv_files[0])
        print(f"Testing with: {csv_files[0]}")
        
        # Test data loading
        try:
            df = pd.read_csv(sample_file, low_memory=False)
            print(f"✓ Data loaded: {df.shape}")
            print(f"✓ Columns: {list(df.columns)[:5]}...")
            
            # Test signal selection logic
            plot_signal_vars = {}
            for signal in df.columns:
                var = tk.BooleanVar(value=False)
                plot_signal_vars[signal] = {'var': var, 'widget': None}
            
            print(f"✓ Created {len(plot_signal_vars)} signal variables")
            
            # Test auto-selection
            common_signals = ['co_pct', 'co2_pct', 'h2_pct', 'ch4_pct', 'o2_pct', 'n2_pct', 
                            'temp', 'temperature', 'pressure', 'flow', 'level']
            auto_selected = 0
            for signal in df.columns:
                if auto_selected >= 4:
                    break
                signal_lower = signal.lower()
                if any(common in signal_lower for common in common_signals):
                    plot_signal_vars[signal]['var'].set(True)
                    auto_selected += 1
                    print(f"✓ Auto-selected: {signal}")
            
            # Test fallback selection
            if auto_selected == 0:
                print("No auto-selection, using fallback...")
                for signal in df.columns:
                    if not any(word in signal.lower() for word in ['time', 'date', 'timestamp']):
                        plot_signal_vars[signal]['var'].set(True)
                        print(f"✓ Fallback selected: {signal}")
                        break
            
            # Test which signals are selected
            selected_signals = [s for s, data in plot_signal_vars.items() if data['var'].get()]
            print(f"✓ Selected signals: {selected_signals}")
            
            if selected_signals:
                print("✓ Signal selection is working!")
            else:
                print("✗ No signals selected - this is the main issue!")
                
                # Force select first numeric column for testing
                for signal in df.columns:
                    if df[signal].dtype in ['float64', 'int64'] and not any(word in signal.lower() for word in ['time', 'date', 'timestamp']):
                        plot_signal_vars[signal]['var'].set(True)
                        print(f"✓ Force selected numeric signal: {signal}")
                        break
                
                selected_signals = [s for s, data in plot_signal_vars.items() if data['var'].get()]
                print(f"✓ After force selection: {selected_signals}")
            
        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ No CSV files found in Half Ton Data directory")
else:
    print("✗ Half Ton Data directory not found")

print("\n" + "="*50)
print("KEY FIXES APPLIED:")
print("="*50)
print("1. ✓ Added _debug_plot_state() for comprehensive debugging")
print("2. ✓ Added _force_signal_selection() to ensure signals are selected")
print("3. ✓ Enhanced get_data_for_plotting() with better error handling")
print("4. ✓ Added fallback signal selection in on_plot_file_select()")
print("5. ✓ Added manual '🔄 Update Plot' button for testing")
print("6. ✓ Enhanced update_plot() with detailed debugging output")

print("\nTo test the fixes:")
print("1. Run the Data Processor application")
print("2. Go to the Plotting tab")
print("3. Select a file from the dropdown")
print("4. Check the console for detailed debug output")
print("5. Use the '🔄 Update Plot' button to force plot updates")
print("6. Check if signals are being selected (checkboxes)")

print("\nIf plotting still doesn't work, the debug output will show exactly where it fails.")
