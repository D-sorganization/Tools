# Restoring Plotting Functionality in Data Processor r0

## Summary of Issues

The r0 version has plotting problems due to:
1. Overly broad exception handling that masks errors
2. Potential timing issues with canvas updates
3. Missing or incorrect attribute checks

## Recommended Fixes

### 1. Replace the `update_plot` Method
Use the fixed version provided in the code artifact above. Key improvements:
- More granular error handling
- Explicit canvas updates with `draw_idle()`
- Better validation of data and columns
- Clearer error messages

### 2. Check Canvas Initialization
Add this method to ensure canvas is properly initialized:

```python
def _ensure_plot_canvas_ready(self):
    """Ensure plot canvas is properly initialized."""
    if not hasattr(self, 'plot_canvas') or self.plot_canvas is None:
        print("ERROR: Plot canvas not initialized!")
        return False
    
    if not hasattr(self, 'plot_ax') or self.plot_ax is None:
        print("ERROR: Plot axes not initialized!")
        return False
        
    # Force a draw to ensure canvas is ready
    try:
        self.plot_canvas.draw()
        return True
    except Exception as e:
        print(f"ERROR: Canvas draw failed - {e}")
        return False
```

### 3. Fix the Plot Initialization
In the `create_plotting_tab` method, ensure proper initialization:

```python
# After creating the plot canvas
self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_canvas_frame)
self.plot_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

# Force initial draw
self.plot_canvas.draw()

# Add a short delay to ensure canvas is ready
self.after(100, lambda: self._ensure_plot_canvas_ready())
```

### 4. Debug Mode
Add a debug mode to track plotting issues:

```python
def enable_plot_debugging(self):
    """Enable verbose debugging for plot operations."""
    self.plot_debug = True
    
def debug_print(self, message):
    """Print debug message if debugging is enabled."""
    if hasattr(self, 'plot_debug') and self.plot_debug:
        print(f"[PLOT DEBUG] {message}")
```

### 5. Common Fixes for Specific Issues

#### If plots don't appear:
1. Check console for error messages
2. Verify data is loaded correctly
3. Ensure at least one signal is selected
4. Try clicking "Update Plot" manually

#### If canvas errors occur:
1. Restart the application
2. Check matplotlib backend compatibility
3. Ensure all dependencies are properly installed

#### If data doesn't plot correctly:
1. Verify datetime parsing is working
2. Check that numeric columns are properly converted
3. Ensure filter parameters are valid

### 6. Alternative Minimal Fix
If the above doesn't work, try this minimal fix that removes all advanced error handling:

```python
def update_plot_minimal(self):
    """Minimal update plot without advanced error handling."""
    if not hasattr(self, 'plot_canvas'):
        return
        
    selected_file = self.plot_file_menu.get()
    if selected_file == "Select a file...":
        return
        
    df = self.get_data_for_plotting(selected_file)
    if df is None:
        return
        
    self.plot_ax.clear()
    
    # Simple plot
    for col in df.columns[1:]:  # Skip first column (usually time)
        self.plot_ax.plot(df.iloc[:, 0], df[col], label=col)
    
    self.plot_ax.legend()
    self.plot_ax.grid(True)
    self.plot_canvas.draw()
```

## Testing Procedure

1. Start the application
2. Load a CSV file
3. Switch to "Plotting & Analysis" tab
4. Select a file from dropdown
5. Select one or more signals
6. Click "Update Plot" if needed
7. Check console for any error messages

## If Problems Persist

1. Compare the exact differences between r0 and Functional Baseline
2. Check Python and library versions
3. Test with a simple CSV file first
4. Disable all filters and advanced features
5. Use the minimal plotting function as a baseline