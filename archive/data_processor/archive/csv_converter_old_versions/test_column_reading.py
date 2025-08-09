#!/usr/bin/env python3
"""
Test script to verify column reading functionality in the enhanced converter.
"""

import sys
from pathlib import Path

import pandas as pd


def test_column_reading():
    """Test reading columns from CSV files."""
    print("Testing column reading functionality...")

    # Create a test CSV file with many columns
    test_data = {}
    for i in range(100):  # Create 100 columns
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
    test_data["mixed_type_col"] = [1, 2.5, "text", 4, 5.7, "more_text", 7, 8.9, 9, 10.1]

    df = pd.DataFrame(test_data)
    test_file = "test_columns.csv"
    df.to_csv(test_file, index=False)

    print(f"Created test file: {test_file}")
    print(f"Columns in test file: {len(df.columns)}")
    print(f"First few columns: {list(df.columns[:5])}")

    try:
        # Test reading columns using the same method as the enhanced converter
        df_sample = pd.read_csv(test_file, nrows=1)
        columns = list(df_sample.columns)

        print(f"Successfully read {len(columns)} columns")
        print(f"Sample columns: {columns[:5]}")

        # Test with the specific column that was causing issues
        if "input_HipX_B" in columns:
            print("✓ Found problematic column 'input_HipX_B'")
        else:
            print("✗ Missing column 'input_HipX_B'")

        return True

    except Exception as e:
        print(f"Error reading columns: {str(e)}")
        return False
    finally:
        # Clean up
        if Path(test_file).exists():
            Path(test_file).unlink()
            print(f"Cleaned up test file: {test_file}")


if __name__ == "__main__":
    success = test_column_reading()
    sys.exit(0 if success else 1)
