#!/usr/bin/env python3
"""
Simple test script to check if the data processor can be imported and run
"""

print("Testing Data Processor Application...")

try:
    import tkinter as tk

    print("✓ tkinter imported successfully")
except ImportError as e:
    print(f"✗ tkinter import failed: {e}")
    exit(1)

try:
    import customtkinter as ctk

    print("✓ customtkinter imported successfully")
except ImportError as e:
    print(f"✗ customtkinter import failed: {e}")
    print("Attempting to install customtkinter...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
    print("✓ customtkinter installed, trying import again...")
    import customtkinter as ctk

    print("✓ customtkinter now imported successfully")

try:
    import pandas as pd

    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import numpy as np

    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import matplotlib.pyplot as plt

    print("✓ matplotlib imported successfully")
except ImportError as e:
    print(f"✗ matplotlib import failed: {e}")

print("\nAttempting to create a simple customtkinter window...")

try:

    class TestApp(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.title("Test Application")
            self.geometry("400x300")

            label = ctk.CTkLabel(self, text="CustomTkinter is working!")
            label.pack(padx=20, pady=20)

            button = ctk.CTkButton(self, text="Close", command=self.quit)
            button.pack(padx=20, pady=20)

            print("✓ CustomTkinter test app created successfully")

            # Auto-close after 2 seconds for testing
            self.after(2000, self.quit)

    print("Creating test app...")
    app = TestApp()
    print("✓ Test app created, running briefly...")
    app.mainloop()
    print("✓ Test app ran successfully")

except Exception as e:
    print(f"✗ CustomTkinter test failed: {e}")
    import traceback

    traceback.print_exc()

print("\nNow testing data processor import...")

try:
    import Data_Processor_r0

    print("✓ Data_Processor_r0 imported successfully")

    print("Creating Data Processor app instance...")
    # Don't run mainloop in test mode
    print("✓ Data processor import test completed")

except Exception as e:
    print(f"✗ Data processor import failed: {e}")
    import traceback

    traceback.print_exc()

print("\nTest completed!")
