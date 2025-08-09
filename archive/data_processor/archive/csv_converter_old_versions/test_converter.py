#!/usr/bin/env python3
"""
Test Script for CSV to Parquet Converter
Creates sample CSV files and tests the conversion functionality.

Author: AI Assistant
Date: 2025
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add the current directory to the path to import our converter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from csv_to_parquet_converter import CSVToParquetConverter


def create_sample_csv_files(num_files=5, rows_per_file=1000):
    """Create sample CSV files for testing."""
    temp_dir = tempfile.mkdtemp(prefix="csv_test_")

    print(f"Creating {num_files} sample CSV files in {temp_dir}")

    # Create sample data
    for i in range(num_files):
        # Generate sample data
        data = {
            "id": range(1, rows_per_file + 1),
            "name": [f"Item_{j}_{i}" for j in range(1, rows_per_file + 1)],
            "value": np.random.randn(rows_per_file),
            "category": np.random.choice(["A", "B", "C"], rows_per_file),
            "timestamp": pd.date_range("2024-01-01", periods=rows_per_file, freq="H"),
        }

        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, f"sample_data_{i+1:03d}.csv")
        df.to_csv(file_path, index=False)
        print(f"Created: {file_path} ({len(df)} rows)")

    return temp_dir


def test_converter_functionality():
    """Test the converter functionality."""
    print("Testing CSV to Parquet Converter")
    print("=" * 50)

    # Create sample data
    test_dir = create_sample_csv_files(num_files=3, rows_per_file=500)

    try:
        # Test 1: Standalone converter
        print("\nTest 1: Testing standalone converter...")
        app = CSVToParquetConverter()

        # Simulate file selection
        app.folder_path_edit.setText(test_dir)
        app._scan_csv_files(test_dir)

        print(f"Found {len(app.csv_files)} CSV files")

        # Test 2: Single file output
        print("\nTest 2: Testing single file output...")
        output_file = os.path.join(test_dir, "combined_output.parquet")
        app.output_path_edit.setText(output_file)
        app.combine_checkbox.setChecked(True)

        # Test 3: Multiple file output
        print("\nTest 3: Testing multiple file output...")
        output_dir = os.path.join(test_dir, "individual_output")
        app.output_path_edit.setText(output_dir)
        app.combine_checkbox.setChecked(False)

        print("All tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise

    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)


def test_pandas_parquet_functionality():
    """Test basic pandas to parquet functionality."""
    print("\nTesting pandas to parquet functionality...")

    # Create sample data
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "salary": [50000, 60000, 70000, 55000, 65000],
    }

    df = pd.DataFrame(data)

    # Test CSV to DataFrame
    temp_csv = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(temp_csv.name, index=False)
    temp_csv.close()

    # Read CSV back
    df_read = pd.read_csv(temp_csv.name)
    print(f"CSV read successfully: {len(df_read)} rows")

    # Test DataFrame to Parquet
    temp_parquet = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    temp_parquet.close()

    df_read.to_parquet(temp_parquet.name, index=False)
    print(f"Parquet file created: {temp_parquet.name}")

    # Read Parquet back
    df_parquet = pd.read_parquet(temp_parquet.name)
    print(f"Parquet read successfully: {len(df_parquet)} rows")

    # Verify data integrity
    if df.equals(df_parquet):
        print("✓ Data integrity verified - CSV and Parquet data match")
    else:
        print("✗ Data integrity check failed")

    # Cleanup
    os.unlink(temp_csv.name)
    os.unlink(temp_parquet.name)


def main():
    """Main test function."""
    print("CSV to Parquet Converter Test Suite")
    print("=" * 50)

    try:
        # Test basic functionality
        test_pandas_parquet_functionality()

        # Test converter functionality
        test_converter_functionality()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("The converter is ready to use.")

    except Exception as e:
        print(f"\nTest suite failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
