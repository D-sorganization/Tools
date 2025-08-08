#!/usr/bin/env python3
"""
Test script for file splitting functionality.
Creates sample data and demonstrates different splitting methods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

def create_sample_data():
    """Create sample data for testing file splitting."""
    print("Creating sample data...")
    
    # Create a large dataset with various data types
    np.random.seed(42)
    n_rows = 100000
    
    # Generate sample data
    data = {
        'id': range(1, n_rows + 1),
        'user_id': np.random.randint(1, 1001, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'value': np.random.normal(100, 20, n_rows),
        'score': np.random.uniform(0, 100, n_rows),
        'status': np.random.choice(['active', 'inactive', 'pending'], n_rows),
        'created_at': [
            datetime.now() - timedelta(days=np.random.randint(0, 365), 
                                     hours=np.random.randint(0, 24))
            for _ in range(n_rows)
        ]
    }
    
    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
    return df

def test_row_splitting(df, output_dir):
    """Test splitting by row count."""
    print("\n=== Testing Row-based Splitting ===")
    
    rows_per_file = 25000  # 4 files of 25k rows each
    total_rows = len(df)
    num_files = (total_rows + rows_per_file - 1) // rows_per_file
    
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, total_rows)
        
        chunk_df = df.iloc[start_idx:end_idx]
        filename = output_dir / f"sample_data_part_{i+1:04d}.parquet"
        
        chunk_df.to_parquet(filename, index=False)
        print(f"Created {filename.name} with {len(chunk_df)} rows")

def test_time_splitting(df, output_dir):
    """Test splitting by time intervals."""
    print("\n=== Testing Time-based Splitting ===")
    
    # Sort by time
    df_sorted = df.sort_values('created_at').reset_index(drop=True)
    
    # Split by monthly intervals
    df_sorted['year_month'] = df_sorted['created_at'].dt.to_period('M')
    
    for period, group in df_sorted.groupby('year_month'):
        if len(group) > 0:
            filename = output_dir / f"sample_data_{period}.parquet"
            group.drop('year_month', axis=1).to_parquet(filename, index=False)
            print(f"Created {filename.name} with {len(group)} rows")

def test_category_splitting(df, output_dir):
    """Test splitting by category."""
    print("\n=== Testing Category-based Splitting ===")
    
    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        filename = output_dir / f"sample_data_category_{category}.parquet"
        category_df.to_parquet(filename, index=False)
        print(f"Created {filename.name} with {len(category_df)} rows")

def main():
    """Main test function."""
    print("File Splitting Test Script")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Create output directory
    output_dir = Path("test_splits")
    output_dir.mkdir(exist_ok=True)
    
    # Test different splitting methods
    test_row_splitting(df, output_dir)
    test_time_splitting(df, output_dir)
    test_category_splitting(df, output_dir)
    
    print(f"\nAll test files created in: {output_dir.absolute()}")
    print("\nYou can now test the file splitting functionality in the GUI:")
    print("1. Launch the enhanced converter")
    print("2. Select the sample data file")
    print("3. Enable 'Split large files'")
    print("4. Click 'Configure Splitting' to set up splitting options")
    print("5. Choose your preferred splitting method")

if __name__ == "__main__":
    main() 