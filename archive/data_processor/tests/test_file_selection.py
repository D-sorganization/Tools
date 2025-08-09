#!/usr/bin/env python3
"""
Test script to verify file selection and signal loading
"""

import sys
import os
import pandas as pd
import tkinter as tk
import customtkinter as ctk

print("Testing file selection and signal loading...")

try:
    # Test the Data Processor import
    import Data_Processor_r0
    print("✓ Data_Processor_r0 imported successfully")
    
    # Create app instance
    app = Data_Processor_r0.CSVProcessorApp()
    print("✓ App instance created")
    
    # Check if components exist
    has_file_list_frame = hasattr(app, 'file_list_frame')
    has_signal_list_frame = hasattr(app, 'signal_list_frame')
    has_input_file_paths = hasattr(app, 'input_file_paths')
    
    print(f"file_list_frame exists: {has_file_list_frame}")
    print(f"signal_list_frame exists: {has_signal_list_frame}")
    print(f"input_file_paths exists: {has_input_file_paths}")
    
    if has_input_file_paths:
        print(f"Current input_file_paths: {getattr(app, 'input_file_paths', [])}")
    
    # Test with sample data
    data_dir = "Half Ton Data"
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            # Simulate file selection
            sample_files = [os.path.join(data_dir, f) for f in csv_files[:2]]  # Take first 2 files
            print(f"Simulating file selection with: {[os.path.basename(f) for f in sample_files]}")
            
            # Manually set the files (simulating file dialog selection)
            app.input_file_paths = sample_files
            print(f"✓ Set input_file_paths: {len(app.input_file_paths)} files")
            
            # Test update_file_list
            print("Testing update_file_list()...")
            try:
                app.update_file_list()
                print("✓ update_file_list() completed successfully")
                
                # Check if widgets were created
                file_widgets = app.file_list_frame.winfo_children()
                print(f"✓ Created {len(file_widgets)} file list widgets")
                
            except Exception as e:
                print(f"✗ update_file_list() failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Test load_signals_from_files
            print("Testing load_signals_from_files()...")
            try:
                app.load_signals_from_files()
                print("✓ load_signals_from_files() completed successfully")
                
                # Check if signal variables were created
                if hasattr(app, 'signal_vars'):
                    print(f"✓ Created {len(app.signal_vars)} signal variables")
                    signal_names = list(app.signal_vars.keys())[:5]  # Show first 5
                    print(f"✓ Signal names (first 5): {signal_names}")
                    
                    # Check if widgets were created
                    signal_widgets = app.signal_list_frame.winfo_children()
                    print(f"✓ Created {len(signal_widgets)} signal list widgets")
                else:
                    print("✗ signal_vars not created")
                
            except Exception as e:
                print(f"✗ load_signals_from_files() failed: {e}")
                import traceback
                traceback.print_exc()
                
            # Test plotting file menu update
            if hasattr(app, 'plot_file_menu'):
                current_values = getattr(app.plot_file_menu, '_values', [])
                print(f"Plot file menu values: {current_values}")
            else:
                print("plot_file_menu not found")
        else:
            print("✗ No CSV files found in Half Ton Data directory")
    else:
        print("✗ Half Ton Data directory not found")
    
    print("\nTest completed. Starting GUI for manual verification...")
    print("1. Go to Processing tab")
    print("2. Click 'Select Input CSV Files'")
    print("3. Check if files appear in the list")
    print("4. Check if signals appear in the signal list")
    
    # Run the app
    app.mainloop()
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
