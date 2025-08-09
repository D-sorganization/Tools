#!/usr/bin/env python3
# =============================================================================
# Test script for PyQt6 version of the Advanced CSV Time Series Processor
# =============================================================================

import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_dependencies():
    """Test if all required dependencies are available."""
    print("Testing dependencies...")

    required_packages = [
        ("PyQt6", "PyQt6"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("openpyxl", "openpyxl"),
        ("Pillow", "PIL"),
        ("simpledbf", "simpledbf"),
        ("pyarrow", "pyarrow"),
    ]

    missing_packages = []

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - MISSING")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("All dependencies are available!")
    return True


def test_file_format_detection():
    """Test file format detection functionality."""
    print("\nTesting file format detection...")

    from Data_Processor_PyQt6 import FileFormatDetector

    # Create test files
    test_files = {}

    # CSV file
    csv_data = pd.DataFrame(
        {
            "Time": pd.date_range("2024-01-01", periods=100, freq="h"),
            "Value1": np.random.randn(100),
            "Value2": np.random.randn(100),
        }
    )
    csv_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    csv_data.to_csv(csv_file.name, index=False)
    test_files["csv"] = csv_file.name

    # Excel file
    excel_file = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    csv_data.to_excel(excel_file.name, index=False)
    test_files["excel"] = excel_file.name

    # Test format detection
    for expected_format, file_path in test_files.items():
        detected_format = FileFormatDetector.detect_format(file_path)
        if detected_format == expected_format:
            print(f"✓ {expected_format.upper()} format detection: PASSED")
        else:
            print(
                f"✗ {expected_format.upper()} format detection: FAILED (detected: {detected_format})"
            )

    # Cleanup
    for file_path in test_files.values():
        try:
            os.unlink(file_path)
        except:
            pass


def test_data_reading_writing():
    """Test data reading and writing functionality."""
    print("\nTesting data reading and writing...")

    from Data_Processor_PyQt6 import DataReader, DataWriter

    # Create test data
    test_data = pd.DataFrame(
        {
            "Temperature": np.random.normal(20, 5, 50),
            "Humidity": np.random.normal(60, 10, 50),
            "Pressure": np.random.normal(1013, 10, 50),
        }
    )
    test_data.index = pd.date_range("2024-01-01", periods=50, freq="h")

    # Test different formats
    formats_to_test = ["csv", "json"]

    # Check if Excel is available
    try:
        import openpyxl

        formats_to_test.append("excel")
    except ImportError:
        print("⚠️  Excel format test skipped (openpyxl not available)")

    for format_type in formats_to_test:
        try:
            # Create temporary file with proper extension
            if format_type == "excel":
                suffix = ".xlsx"
            else:
                suffix = f".{format_type}"

            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_file.close()

            # Write data
            DataWriter.write_file(test_data, temp_file.name, format_type)

            # Read data back
            read_data = DataReader.read_file(temp_file.name, format_type)

            # Compare
            if len(test_data) == len(read_data) and len(test_data.columns) == len(
                read_data.columns
            ):
                print(f"✓ {format_type.upper()} read/write: PASSED")
            else:
                print(f"✗ {format_type.upper()} read/write: FAILED")

            # Cleanup
            try:
                os.unlink(temp_file.name)
            except:
                pass

        except Exception as e:
            print(f"✗ {format_type.upper()} read/write: FAILED - {e}")
            import traceback

            traceback.print_exc()


def test_processing_function():
    """Test the main processing function."""
    print("\nTesting processing function...")

    from Data_Processor_PyQt6 import process_single_csv_file

    # Create test CSV file
    test_data = pd.DataFrame(
        {
            "Time": pd.date_range("2024-01-01", periods=100, freq="h"),
            "Signal1": np.random.randn(100),
            "Signal2": np.random.randn(100),
            "Signal3": np.random.randn(100),
        }
    )

    temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    test_data.to_csv(temp_file.name, index=False)

    # Test processing settings
    settings = {
        "selected_signals": ["Signal1", "Signal2"],
        "filter_type": "Moving Average",
        "ma_window": 5,
        "integration_signals": ["Signal1"],
        "integration_method": "Trapezoidal",
        "differentiation_signals": ["Signal2"],
        "differentiation_method": "Simple Difference",
        "custom_variables": {"Ratio": "Signal1 / Signal2"},
        "resample_rule": "None",
        "sort_by": "None",
    }

    try:
        # Process the file
        result = process_single_csv_file(temp_file.name, settings)

        if result is not None and len(result) > 0:
            print("✓ Processing function: PASSED")
            print(f"  - Input rows: {len(test_data)}")
            print(f"  - Output rows: {len(result)}")
            print(f"  - Output columns: {list(result.columns)}")
        else:
            print("✗ Processing function: FAILED - No output data")

    except Exception as e:
        print(f"✗ Processing function: FAILED - {e}")

    # Cleanup
    try:
        os.unlink(temp_file.name)
    except:
        pass


def test_application_creation():
    """Test if the PyQt6 application can be created."""
    print("\nTesting application creation...")

    try:
        from Data_Processor_PyQt6 import CSVProcessorAppPyQt6
        from PyQt6.QtWidgets import QApplication

        # Create QApplication instance
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create main window
        window = CSVProcessorAppPyQt6()

        # Check if window was created successfully
        if window is not None:
            print("✓ Application creation: PASSED")
            print(f"  - Window title: {window.windowTitle()}")
            print(f"  - Tab count: {window.tab_widget.count()}")

            # Check tab names
            expected_tabs = [
                "Processing",
                "Plotting & Analysis",
                "Plots List",
                "Format Converter",
                "Folder Tool",
                "DAT File Import",
                "Help",
            ]

            for i, expected_tab in enumerate(expected_tabs):
                if i < window.tab_widget.count():
                    actual_tab = window.tab_widget.tabText(i)
                    if actual_tab == expected_tab:
                        print(f"  - Tab {i+1}: {actual_tab} ✓")
                    else:
                        print(
                            f"  - Tab {i+1}: {actual_tab} (expected: {expected_tab}) ✗"
                        )
                else:
                    print(f"  - Tab {i+1}: MISSING ✗")
        else:
            print("✗ Application creation: FAILED")

    except Exception as e:
        print(f"✗ Application creation: FAILED - {e}")


def test_thread_classes():
    """Test the thread classes."""
    print("\nTesting thread classes...")

    from Data_Processor_PyQt6 import (ConversionThread, FolderProcessingThread,
                                      ProcessingThread)

    # Test ProcessingThread
    try:
        thread = ProcessingThread([], {}, "")
        print("✓ ProcessingThread creation: PASSED")
    except Exception as e:
        print(f"✗ ProcessingThread creation: FAILED - {e}")

    # Test ConversionThread
    try:
        thread = ConversionThread([], "csv", False, "")
        print("✓ ConversionThread creation: PASSED")
    except Exception as e:
        print(f"✗ ConversionThread creation: FAILED - {e}")

    # Test FolderProcessingThread
    try:
        thread = FolderProcessingThread([], "", "Combine")
        print("✓ FolderProcessingThread creation: PASSED")
    except Exception as e:
        print(f"✗ FolderProcessingThread creation: FAILED - {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PyQt6 Data Processor - Comprehensive Test Suite")
    print("=" * 60)

    # Run tests
    tests = [
        test_dependencies,
        test_file_format_detection,
        test_data_reading_writing,
        test_processing_function,
        test_thread_classes,
        test_application_creation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The PyQt6 version is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
