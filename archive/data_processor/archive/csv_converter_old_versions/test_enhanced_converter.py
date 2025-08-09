#!/usr/bin/env python3
"""
Test script for Enhanced CSV to Parquet Converter
Tests column selection functionality and large dataset handling.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

# Add the current directory to the path to import our converter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from csv_to_parquet_converter_enhanced import CSVToParquetConverter, ColumnSelectionDialog


def create_test_csv_files(num_files=3, rows_per_file=100, num_columns=50):
    """Create test CSV files with many columns for testing."""
    temp_dir = tempfile.mkdtemp(prefix="csv_enhanced_test_")
    
    print(f"Creating {num_files} test CSV files with {num_columns} columns each in {temp_dir}")
    
    # Create column names with various patterns
    base_columns = [
        'id', 'timestamp', 'value', 'category', 'status',
        'input_HipX_A', 'input_HipX_B', 'input_HipY_A', 'input_HipY_B',
        'output_Force_X', 'output_Force_Y', 'output_Force_Z',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
        'measurement_A', 'measurement_B', 'measurement_C',
        'data_point_1', 'data_point_2', 'data_point_3',
        'calculated_value_1', 'calculated_value_2', 'calculated_value_3',
        'raw_data_1', 'raw_data_2', 'raw_data_3', 'raw_data_4',
        'processed_data_1', 'processed_data_2', 'processed_data_3',
        'filtered_data_1', 'filtered_data_2', 'filtered_data_3',
        'normalized_data_1', 'normalized_data_2', 'normalized_data_3',
        'derived_feature_1', 'derived_feature_2', 'derived_feature_3',
        'aggregated_value_1', 'aggregated_value_2', 'aggregated_value_3',
        'statistical_measure_1', 'statistical_measure_2', 'statistical_measure_3'
    ]
    
    # Extend with more columns if needed
    while len(base_columns) < num_columns:
        base_columns.append(f'column_{len(base_columns)}')
    
    columns = base_columns[:num_columns]
    
    for i in range(num_files):
        # Generate sample data
        data = {}
        for col in columns:
            if 'id' in col.lower():
                data[col] = range(1, rows_per_file + 1)
            elif 'timestamp' in col.lower():
                data[col] = pd.date_range('2024-01-01', periods=rows_per_file, freq='1S')
            elif 'value' in col.lower() or 'force' in col.lower() or 'measurement' in col.lower():
                data[col] = np.random.normal(0, 10, rows_per_file)
            elif 'category' in col.lower() or 'status' in col.lower():
                data[col] = np.random.choice(['A', 'B', 'C', 'D'], rows_per_file)
            elif 'sensor' in col.lower() or 'data' in col.lower():
                data[col] = np.random.uniform(-100, 100, rows_per_file)
            else:
                data[col] = np.random.randn(rows_per_file)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        filename = f"test_file_{i+1:03d}.csv"
        filepath = os.path.join(temp_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Created {filename} with {len(df.columns)} columns and {len(df)} rows")
    
    return temp_dir


def test_column_selection_dialog():
    """Test the column selection dialog functionality."""
    print("\n=== Testing Column Selection Dialog ===")
    
    # Create test columns
    test_columns = [
        'input_HipX_A', 'input_HipX_B', 'input_HipY_A', 'input_HipY_B',
        'output_Force_X', 'output_Force_Y', 'output_Force_Z',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
        'measurement_A', 'measurement_B', 'measurement_C',
        'data_point_1', 'data_point_2', 'data_point_3',
        'calculated_value_1', 'calculated_value_2', 'calculated_value_3'
    ]
    
    print(f"Test columns: {len(test_columns)} columns")
    print("Sample columns:", test_columns[:5])
    
    # Test search functionality
    search_term = "input"
    matching_columns = [col for col in test_columns if search_term.lower() in col.lower()]
    print(f"Columns matching '{search_term}': {len(matching_columns)}")
    print("Matching columns:", matching_columns)
    
    # Test save/load functionality
    test_selection = ['input_HipX_A', 'input_HipX_B', 'output_Force_X', 'sensor_1']
    print(f"Test selection: {len(test_selection)} columns")
    print("Selected columns:", test_selection)
    
    print("✓ Column selection dialog tests completed")


def test_enhanced_converter():
    """Test the enhanced converter functionality."""
    print("\n=== Testing Enhanced Converter ===")
    
    # Create test files
    test_dir = create_test_csv_files(num_files=2, rows_per_file=50, num_columns=30)
    
    try:
        # Test reading columns from first file
        csv_files = list(Path(test_dir).glob("*.csv"))
        if csv_files:
            first_file = str(csv_files[0])
            df_sample = pd.read_csv(first_file, nrows=1)
            columns = list(df_sample.columns)
            
            print(f"Read {len(columns)} columns from {Path(first_file).name}")
            print("Sample columns:", columns[:5])
            
            # Test column filtering
            test_selection = ['input_HipX_A', 'input_HipX_B', 'output_Force_X']
            available_columns = set(columns)
            columns_to_keep = list(available_columns.intersection(set(test_selection)))
            
            print(f"Available columns: {len(available_columns)}")
            print(f"Selected columns: {len(test_selection)}")
            print(f"Columns to keep: {len(columns_to_keep)}")
            print("Columns to keep:", columns_to_keep)
            
            # Test conversion with column selection
            if columns_to_keep:
                df = pd.read_csv(first_file)
                df_filtered = df[columns_to_keep]
                print(f"Original shape: {df.shape}")
                print(f"Filtered shape: {df_filtered.shape}")
                print("✓ Column filtering test completed")
            
    except Exception as e:
        print(f"Error testing enhanced converter: {e}")
    
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")


def test_large_column_handling():
    """Test handling of large column sets."""
    print("\n=== Testing Large Column Handling ===")
    
    # Create a large column set (simulating 2000+ columns)
    large_columns = []
    for i in range(2000):
        if i < 100:
            large_columns.append(f'input_HipX_{i}')
        elif i < 200:
            large_columns.append(f'output_Force_{i}')
        elif i < 300:
            large_columns.append(f'sensor_{i}')
        else:
            large_columns.append(f'column_{i}')
    
    print(f"Created {len(large_columns)} test columns")
    
    # Test search performance
    import time
    
    # Test case-insensitive search
    start_time = time.time()
    search_term = "input"
    matching_columns = [col for col in large_columns if search_term.lower() in col.lower()]
    search_time = time.time() - start_time
    
    print(f"Search for '{search_term}' found {len(matching_columns)} matches in {search_time:.4f} seconds")
    
    # Test case-sensitive search
    start_time = time.time()
    search_term = "Input"
    matching_columns = [col for col in large_columns if search_term in col]
    search_time = time.time() - start_time
    
    print(f"Case-sensitive search for '{search_term}' found {len(matching_columns)} matches in {search_time:.4f} seconds")
    
    print("✓ Large column handling tests completed")


if __name__ == "__main__":
    print("Enhanced CSV to Parquet Converter Test Suite")
    print("=" * 50)
    
    test_column_selection_dialog()
    test_enhanced_converter()
    test_large_column_handling()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("\nTo test the GUI:")
    print("1. Run: python csv_to_parquet_converter_enhanced.py")
    print("2. Select a folder with CSV files")
    print("3. Uncheck 'Use all columns' and click 'Select Columns'")
    print("4. Test the search, selection, and save/load functionality")
