#!/usr/bin/env python3
"""
Simple test for the enhanced converter to identify issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced converter classes
from csv_to_parquet_converter_enhanced import ColumnSelectionDialog, CSVToParquetConverter
from PyQt6.QtWidgets import QApplication
import pandas as pd
from pathlib import Path

def test_column_dialog():
    """Test the ColumnSelectionDialog with sample data."""
    print("Testing ColumnSelectionDialog...")
    
    # Create sample columns (simulating 2000+ columns)
    columns = []
    for i in range(100):  # Reduced for testing
        columns.append(f'column_{i:03d}')
    
    # Add some problematic columns
    columns.extend(['input_HipX_B', 'mixed_type_col', 'numeric_col'])
    
    print(f"Created {len(columns)} test columns")
    
    # Create QApplication if not already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # Test creating the dialog
        dialog = ColumnSelectionDialog(columns)
        print("✓ Successfully created ColumnSelectionDialog")
        
        # Test populating the list
        dialog._populate_column_list()
        print(f"✓ Successfully populated list with {dialog.column_list.count()} items")
        
        # Test filtering
        dialog.search_edit.setText("input")
        dialog._filter_columns()
        visible_count = sum(1 for i in range(dialog.column_list.count()) 
                          if not dialog.column_list.item(i).isHidden())
        print(f"✓ Filtering works: {visible_count} items visible after filtering")
        
        # Test selection
        dialog._select_all()
        selected_count = sum(1 for i in range(dialog.column_list.count())
                           if dialog.column_list.item(i).checkState().value == 2)
        print(f"✓ Selection works: {selected_count} items selected")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing ColumnSelectionDialog: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_converter():
    """Test the CSVToParquetConverter class."""
    print("\nTesting CSVToParquetConverter...")
    
    # Create QApplication if not already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # Create test CSV file
        test_data = {}
        for i in range(50):
            test_data[f'column_{i:03d}'] = [f'value_{i}_{j}' for j in range(10)]
        test_data['input_HipX_B'] = [16.43, -28.48, 26.88, -42.58, 47.57, -19.25, -32.59, 24.91, -13.96, 43.10]
        
        df = pd.DataFrame(test_data)
        test_file = 'test_converter.csv'
        df.to_csv(test_file, index=False)
        
        print(f"Created test file: {test_file}")
        
        # Test creating the converter
        converter = CSVToParquetConverter()
        print("✓ Successfully created CSVToParquetConverter")
        
        # Test scanning files
        converter._scan_csv_files('.')
        print(f"✓ Found {len(converter.csv_files)} CSV files")
        
        # Test column selection
        if converter.csv_files:
            converter._select_columns()
            print("✓ Column selection method executed successfully")
        
        # Clean up
        if Path(test_file).exists():
            Path(test_file).unlink()
            print(f"Cleaned up: {test_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing CSVToParquetConverter: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Enhanced Converter Simple Test ===")
    
    success1 = test_column_dialog()
    success2 = test_csv_converter()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
        return True
    else:
        print("\n✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
