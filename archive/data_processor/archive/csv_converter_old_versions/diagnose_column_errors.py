#!/usr/bin/env python3
"""
Diagnostic script to identify column reading errors.
Run this script and provide the path to your CSV files to diagnose issues.
"""

import sys
import traceback
from pathlib import Path

import pandas as pd


def diagnose_csv_file(file_path):
    """Diagnose issues with a specific CSV file."""
    print(f"\n=== Diagnosing: {file_path} ===")

    try:
        # Check if file exists
        if not Path(file_path).exists():
            print(f"✗ File does not exist: {file_path}")
            return False

        # Check file size
        file_size = Path(file_path).stat().st_size
        print(f"File size: {file_size:,} bytes")

        if file_size == 0:
            print("✗ File is empty")
            return False

        # Try to read just the first few lines to check format
        print("Reading first few lines...")
        with open(file_path, "r", encoding="utf-8") as f:
            first_lines = [f.readline().strip() for _ in range(3)]

        print("First 3 lines:")
        for i, line in enumerate(first_lines, 1):
            print(f"  {i}: {line[:100]}{'...' if len(line) > 100 else ''}")

        # Try to read with pandas
        print("\nAttempting to read with pandas...")
        try:
            # First, try reading just headers
            df_headers = pd.read_csv(file_path, nrows=0)
            print(f"✓ Successfully read headers: {len(df_headers.columns)} columns")
            print(f"First 5 column names: {list(df_headers.columns[:5])}")

            # Now try reading first row (like the enhanced converter does)
            df_sample = pd.read_csv(file_path, nrows=1)
            print(f"✓ Successfully read first row: {len(df_sample.columns)} columns")

            # Check for any problematic column names
            problematic_columns = []
            for col in df_sample.columns:
                if pd.isna(col) or col == "" or str(col).strip() == "":
                    problematic_columns.append(col)
                elif any(char in str(col) for char in ["\n", "\r", "\t"]):
                    problematic_columns.append(col)

            if problematic_columns:
                print(
                    f"⚠ Found {len(problematic_columns)} potentially problematic columns:"
                )
                for col in problematic_columns:
                    print(f"  - '{col}' (type: {type(col)})")
            else:
                print("✓ No problematic column names found")

            return True

        except Exception as e:
            print(f"✗ Error reading with pandas: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ General error: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main diagnostic function."""
    print("=== CSV Column Reading Diagnostic Tool ===")
    print("This tool will help identify issues with reading columns from CSV files.")

    if len(sys.argv) > 1:
        # Use command line argument
        file_path = sys.argv[1]
        diagnose_csv_file(file_path)
    else:
        # Interactive mode
        print("\nEnter the path to a CSV file to diagnose (or 'quit' to exit):")

        while True:
            file_path = input("\nCSV file path: ").strip()

            if file_path.lower() in ["quit", "exit", "q"]:
                break

            if file_path:
                diagnose_csv_file(file_path)
            else:
                print("Please enter a valid file path.")


if __name__ == "__main__":
    main()
