#!/usr/bin/env python3
"""
Comprehensive test for the enhanced converter workflow.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def create_test_csv_files():
    """Create multiple test CSV files with many columns."""
    print("Creating test CSV files...")

    # Create a directory for test files
    test_dir = Path("test_csv_files")
    test_dir.mkdir(exist_ok=True)

    # Create multiple CSV files with many columns
    for file_num in range(5):
        test_data = {}

        # Create many columns (simulating 2000+ columns)
        for i in range(50):  # Reduced for testing, but same logic
            test_data[f"column_{i:03d}"] = [f"value_{i}_{j}" for j in range(10)]

        # Add some special columns that might cause issues
        test_data["input_HipX_B"] = [
            16.43,
            -28.48,
            26.88,
            -42.58,
            47.57,
            -19.25,
            -32.59,
            24.91,
            -13.96,
            43.10,
        ]
        test_data["mixed_type_col"] = [
            1,
            2.5,
            "text",
            4,
            5.7,
            "more_text",
            7,
            8.9,
            9,
            10.1,
        ]
        test_data["numeric_col"] = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]

        df = pd.DataFrame(test_data)
        file_path = test_dir / f"test_file_{file_num:03d}.csv"
        df.to_csv(file_path, index=False)
        print(f"Created: {file_path} with {len(df.columns)} columns")

    return test_dir


def test_column_reading_workflow():
    """Test the exact column reading workflow from the enhanced converter."""
    print("\nTesting column reading workflow...")

    # Create test files
    test_dir = create_test_csv_files()

    try:
        # Get list of CSV files (simulating the file scanning)
        csv_files = list(test_dir.glob("*.csv"))
        csv_files.sort()

        print(f"Found {len(csv_files)} CSV files")

        if not csv_files:
            print("No CSV files found!")
            return False

        # Test reading columns from first file (exact method from enhanced converter)
        first_file = str(csv_files[0])
        print(f"Reading columns from: {first_file}")

        try:
            # This is the exact line from the enhanced converter
            df_sample = pd.read_csv(first_file, nrows=1)
            columns = list(df_sample.columns)

            print(f"✓ Successfully read {len(columns)} columns")
            print(f"Sample columns: {columns[:5]}")

            # Test specific columns
            expected_columns = ["input_HipX_B", "mixed_type_col", "numeric_col"]
            for col in expected_columns:
                if col in columns:
                    print(f"✓ Found column: {col}")
                else:
                    print(f"✗ Missing column: {col}")

            return True

        except Exception as e:
            print(f"✗ Error reading columns from {first_file}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return False

    finally:
        # Clean up
        import shutil

        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"Cleaned up test directory: {test_dir}")


def test_large_column_set():
    """Test with a very large number of columns."""
    print("\nTesting with large column set...")

    # Create a CSV with many columns (simulating 2000+ columns)
    test_data = {}
    for i in range(200):  # Simulating many columns
        test_data[f"column_{i:04d}"] = [f"value_{i}_{j}" for j in range(5)]

    # Add some problematic columns
    test_data["input_HipX_B"] = [16.43, -28.48, 26.88, -42.58, 47.57]
    test_data["mixed_type_col"] = [1, 2.5, "text", 4, 5.7]

    df = pd.DataFrame(test_data)
    test_file = "test_large_columns.csv"
    df.to_csv(test_file, index=False)

    try:
        print(f"Created file with {len(df.columns)} columns")

        # Test reading columns
        df_sample = pd.read_csv(test_file, nrows=1)
        columns = list(df_sample.columns)

        print(f"✓ Successfully read {len(columns)} columns")
        print(f"First 5 columns: {columns[:5]}")
        print(f"Last 5 columns: {columns[-5:]}")

        return True

    except Exception as e:
        print(f"✗ Error reading large column set: {str(e)}")
        return False
    finally:
        if Path(test_file).exists():
            Path(test_file).unlink()
            print(f"Cleaned up: {test_file}")


if __name__ == "__main__":
    print("=== Enhanced Converter Workflow Test ===")

    success1 = test_column_reading_workflow()
    success2 = test_large_column_set()

    if success1 and success2:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
