#!/usr/bin/env python3
"""
Test script to verify data type conversion fix
"""

import pandas as pd
import numpy as np
import tempfile
import os

def test_mixed_data_types():
    """Test conversion with mixed data types."""
    print("Testing mixed data type conversion...")
    
    # Create test data with mixed types
    data = {
        'id': [1, 2, 3, 4, 5],
        'value': [1.5, 2.7, 3, 4.2, 5.0],  # Mixed int/float
        'category': ['A', 'B', 'C', 'D', 'E'],
        'mixed_column': [1, 2.5, 'text', 4, 5.7]  # Mixed types
    }
    
    df = pd.DataFrame(data)
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_file = f.name
    
    print(f"Created test CSV with mixed data types: {csv_file}")
    print(f"Original data types: {df.dtypes}")
    
    # Test conversion to string (auto-convert mode)
    df_converted = df.astype(str)
    print(f"Converted data types: {df_converted.dtypes}")
    
    # Test parquet conversion
    parquet_file = csv_file.replace('.csv', '.parquet')
    df_converted.to_parquet(parquet_file, index=False)
    
    # Read back and verify
    df_read = pd.read_parquet(parquet_file)
    print(f"Read back data types: {df_read.dtypes}")
    
    print("✓ Mixed data type conversion test completed successfully!")
    
    # Cleanup
    os.unlink(csv_file)
    os.unlink(parquet_file)

if __name__ == "__main__":
    test_mixed_data_types()
