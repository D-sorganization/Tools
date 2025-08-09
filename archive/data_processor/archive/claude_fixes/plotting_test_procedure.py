# Test Script for Data Processor Plotting Functionality
# Run this as a separate test to verify plotting works

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_csv():
    """Create a test CSV file for plotting verification."""
    # Generate test data
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    time_points = [start_time + timedelta(seconds=i) for i in range(1000)]
    
    # Create signals
    signal1 = np.sin(np.linspace(0, 4 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
    signal2 = np.cos(np.linspace(0, 4 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
    signal3 = np.linspace(0, 10, 1000) + np.random.normal(0, 0.5, 1000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': time_points,
        'Signal_1_Sine': signal1,
        'Signal_2_Cosine': signal2,
        'Signal_3_Linear': signal3
    })
    
    # Save to CSV
    df.to_csv('test_plot_data.csv', index=False)
    print("Test CSV created: test_plot_data.csv")
    return 'test_plot_data.csv'

def test_plotting_standalone():
    """Test matplotlib plotting independently."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import tkinter as tk
    
    # Create test window
    root = tk.Tk()
    root.title("Plot Test")
    
    # Create figure
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Plot
    ax.plot(x, y, label='Test Signal')
    ax.set_title('Matplotlib Test Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    
    # Create canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    print("If you see a sine wave, matplotlib is working correctly.")
    
    # Add update button
    def update_plot():
        ax.clear()
        y_new = np.cos(x)
        ax.plot(x, y_new, label='Updated Signal', color='red')
        ax.set_title('Updated Plot')
        ax.legend()
        ax.grid(True)
        canvas.draw_idle()
        print("Plot updated")
    
    tk.Button(root, text="Update Plot", command=update_plot).pack()
    
    root.mainloop()

def verify_data_processor_plotting():
    """Steps to verify plotting in the main application."""
    print("""
    Manual Testing Procedure for Data Processor:
    
    1. Start the Data Processor application
    
    2. In the Processing tab:
       - Click "Select Input CSV File(s)"
       - Choose the test_plot_data.csv file
       - Select all signals
       - Click "Process & Batch Export Files"
    
    3. Switch to "Plotting & Analysis" tab
    
    4. Check for these elements:
       - File dropdown should show your file
       - X-Axis dropdown should show "Time" or first column
       - Signal checkboxes should appear on the left
    
    5. Select one or more signals
    
    6. Look for the plot to appear automatically
       - If not, click "Update Plot" button
    
    7. Test these features:
       - Zoom: Click and drag on the plot
       - Pan: Click the pan button and drag
       - Change plot type dropdown
       - Add a trendline
       - Change colors
    
    8. Check the console for any error messages
    
    Common Issues:
    - If plot is blank: Check console for errors
    - If signals don't appear: Verify file was processed
    - If plot freezes: Use Force Refresh button
    """)

if __name__ == "__main__":
    # Create test data
    test_file = create_test_csv()
    
    # Test matplotlib independently
    print("\nTesting matplotlib independently...")
    print("Close the test window to continue.")
    test_plotting_standalone()
    
    # Print verification steps
    verify_data_processor_plotting()