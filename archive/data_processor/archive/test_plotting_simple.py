#!/usr/bin/env python3
"""
Simple test script to verify plotting functionality fixes.
This script tests the core logic without complex GUI mocking.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the current directory to the path so we can import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_processed_files_initialization():
    """Test that processed_files attribute is properly initialized."""
    print("Testing processed_files initialization...")

    # Import the main module
    from CSV_Processor_Rev5_Complete import CSVProcessorApp

    # Create an instance
    app = CSVProcessorApp()

    # Check if processed_files is initialized
    assert hasattr(
        app, "processed_files"
    ), "processed_files attribute should be initialized"
    assert isinstance(
        app.processed_files, dict
    ), "processed_files should be a dictionary"
    assert len(app.processed_files) == 0, "processed_files should be empty initially"

    print("✓ processed_files initialization test passed")


def test_get_data_for_plotting_logic():
    """Test the get_data_for_plotting method logic."""
    print("Testing get_data_for_plotting method logic...")

    from CSV_Processor_Rev5_Complete import CSVProcessorApp

    app = CSVProcessorApp()

    # Test with non-existent file
    result = app.get_data_for_plotting("nonexistent.csv")
    assert result is None, "Should return None for non-existent file"

    # Test with empty processed_files
    result = app.get_data_for_plotting("test.csv")
    assert result is None, "Should return None when file not in processed_files"

    # Test with file in processed_files
    test_data = pd.DataFrame(
        {
            "Time": pd.date_range("2024-01-01", periods=10, freq="H"),
            "Signal1": np.random.randn(10),
            "Signal2": np.random.randn(10),
        }
    )
    app.processed_files["test.csv"] = test_data

    result = app.get_data_for_plotting("test.csv")
    assert result is not None, "Should return data when file is in processed_files"
    assert isinstance(result, pd.DataFrame), "Should return a DataFrame"
    assert len(result) == 10, "Should return correct number of rows"

    print("✓ get_data_for_plotting method logic test passed")


def test_process_files_stores_data():
    """Test that process_files stores data in processed_files."""
    print("Testing process_files data storage...")

    from CSV_Processor_Rev5_Complete import CSVProcessorApp

    app = CSVProcessorApp()

    # Mock the _process_single_file method to return test data
    def mock_process_single_file(file_path, settings):
        return pd.DataFrame(
            {
                "Time": pd.date_range("2024-01-01", periods=10, freq="H"),
                "Signal1": np.random.randn(10),
                "Signal2": np.random.randn(10),
            }
        )

    app._process_single_file = mock_process_single_file

    # Mock input file paths
    app.input_file_paths = ["/path/to/test1.csv", "/path/to/test2.csv"]

    # Mock signal_vars
    app.signal_vars = {
        "Time": {
            "var": type("MockVar", (), {"get": lambda *args: True})(),
            "checkbox": type(
                "MockCheckbox", (), {"configure": lambda **kwargs: None}
            )(),
        },
        "Signal1": {
            "var": type("MockVar", (), {"get": lambda *args: True})(),
            "checkbox": type(
                "MockCheckbox", (), {"configure": lambda **kwargs: None}
            )(),
        },
        "Signal2": {
            "var": type("MockVar", (), {"get": lambda *args: True})(),
            "checkbox": type(
                "MockCheckbox", (), {"configure": lambda **kwargs: None}
            )(),
        },
    }

    # Mock other required attributes
    app.filter_type_var = type("MockVar", (), {"get": lambda *args: "None"})()
    app.resample_var = type("MockVar", (), {"get": lambda *args: False})()
    app.ma_value_entry = type("MockEntry", (), {"get": lambda *args: "10"})()
    app.bw_order_entry = type("MockEntry", (), {"get": lambda *args: "3"})()
    app.bw_cutoff_entry = type("MockEntry", (), {"get": lambda *args: "0.1"})()
    app.median_kernel_entry = type("MockEntry", (), {"get": lambda *args: "5"})()
    app.savgol_window_entry = type("MockEntry", (), {"get": lambda *args: "11"})()
    app.savgol_polyorder_entry = type("MockEntry", (), {"get": lambda *args: "2"})()
    app.trim_date_entry = type("MockEntry", (), {"get": lambda *args: ""})()
    app.trim_start_entry = type("MockEntry", (), {"get": lambda *args: ""})()
    app.trim_end_entry = type("MockEntry", (), {"get": lambda *args: ""})()

    # Mock the _export_processed_files method
    app._export_processed_files = lambda *args, **kwargs: None

    # Mock messagebox
    app.messagebox = type(
        "MockMessagebox", (), {"showinfo": lambda *args, **kwargs: None}
    )()

    # Call process_files
    app.process_files()

    # Check that data was stored in processed_files
    assert "test1.csv" in app.processed_files, "test1.csv should be in processed_files"
    assert "test2.csv" in app.processed_files, "test2.csv should be in processed_files"
    assert isinstance(
        app.processed_files["test1.csv"], pd.DataFrame
    ), "Stored data should be DataFrame"
    assert isinstance(
        app.processed_files["test2.csv"], pd.DataFrame
    ), "Stored data should be DataFrame"

    print("✓ process_files data storage test passed")


def test_plot_file_menu_update_logic():
    """Test that plot file menu update logic works correctly."""
    print("Testing plot file menu update logic...")

    from CSV_Processor_Rev5_Complete import CSVProcessorApp

    app = CSVProcessorApp()

    # Mock plot_file_menu
    menu_configured = False
    menu_values = []

    def mock_configure(*args, **kwargs):
        nonlocal menu_configured, menu_values
        menu_configured = True
        menu_values = kwargs.get("values", [])

    app.plot_file_menu = type("MockMenu", (), {"configure": mock_configure})()

    # Mock input file paths
    app.input_file_paths = ["/path/to/file1.csv", "/path/to/file2.csv"]

    # Mock update_signal_list method
    app.update_signal_list = lambda *args, **kwargs: None

    # Call load_signals_from_files
    app.load_signals_from_files()

    # Check that plot_file_menu was configured with correct values
    assert menu_configured, "plot_file_menu.configure should have been called"
    expected_values = ["Select a file...", "file1.csv", "file2.csv"]
    assert (
        menu_values == expected_values
    ), f"Expected {expected_values}, got {menu_values}"

    print("✓ plot file menu update logic test passed")


if __name__ == "__main__":
    print("Running plotting functionality tests...\n")

    try:
        test_processed_files_initialization()
        test_get_data_for_plotting_logic()
        test_process_files_stores_data()
        test_plot_file_menu_update_logic()

        print(
            "\n🎉 All tests passed! The plotting functionality should now work correctly."
        )
        print("\nKey fixes implemented:")
        print("1. ✓ Initialized self.processed_files attribute in __init__")
        print(
            "2. ✓ Modified process_files to store processed data in self.processed_files"
        )
        print("3. ✓ Verified get_data_for_plotting can access processed data")
        print("4. ✓ Confirmed plot file menu is properly updated")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
