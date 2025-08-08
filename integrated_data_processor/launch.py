#!/usr/bin/env python3
"""
Launch script for the Integrated Data Processor
Self-contained version with all dependencies included.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from Data_Processor_Integrated import IntegratedCSVProcessorApp
    
    if __name__ == "__main__":
        print("Starting Integrated Data Processor...")
        print("This version includes:")
        print("- Original CSV processing functionality")
        print("- Format converter with support for multiple file formats")
        print("- Parquet file analyzer")
        print("- Folder tool functionality")
        print("- All existing plotting and analysis features")
        print()
        
        app = IntegratedCSVProcessorApp()
        app.mainloop()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install customtkinter pandas numpy scipy matplotlib openpyxl Pillow simpledbf pyarrow tables feather-format")
    input("Press Enter to exit...")
except Exception as e:
    print(f"Error starting application: {e}")
    input("Press Enter to exit...")
