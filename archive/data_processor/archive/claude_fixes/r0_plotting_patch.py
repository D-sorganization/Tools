# Critical Plotting Patches for Data_Processor_r0.py
# Apply these changes to restore plotting functionality


# PATCH 1: Add this method to the DataProcessor class
def _safe_plot_update(self):
    """Safely update plot with better error isolation."""
    # Ensure canvas exists
    if not hasattr(self, "plot_canvas") or not hasattr(self, "plot_ax"):
        print("ERROR: Plot canvas not initialized")
        return False

    # Test canvas responsiveness
    try:
        self.plot_canvas.draw()
    except Exception as e:
        print(f"ERROR: Canvas not responsive - {e}")
        # Try to recreate canvas
        self._reinitialize_plot_canvas()
        return False

    return True


# PATCH 2: Add canvas reinitialization method
def _reinitialize_plot_canvas(self):
    """Reinitialize the plot canvas if it becomes unresponsive."""
    try:
        # Find the parent frame
        if hasattr(self, "plot_canvas"):
            parent = self.plot_canvas.get_tk_widget().master
            self.plot_canvas.get_tk_widget().destroy()

            # Recreate
            self.plot_fig = Figure(figsize=(5, 4), dpi=100)
            self.plot_ax = self.plot_fig.add_subplot(111)
            self.plot_fig.tight_layout()

            self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=parent)
            self.plot_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

            # Recreate toolbar if it exists
            if hasattr(self, "plot_toolbar"):
                self.plot_toolbar.destroy()
                self.plot_toolbar = NavigationToolbar2Tk(
                    self.plot_canvas, parent, pack_toolbar=False
                )
                self.plot_toolbar.grid(row=0, column=0, sticky="ew")

            print("Plot canvas reinitialized successfully")
            return True
    except Exception as e:
        print(f"ERROR: Failed to reinitialize canvas - {e}")
        return False


# PATCH 3: Replace the beginning of update_plot method
def update_plot(self):
    """Update the plot with better error handling."""
    # First check if we can safely update
    if not self._safe_plot_update():
        self.status_label.configure(text="Plot canvas error - trying to recover...")
        return

    selected_file = self.plot_file_menu.get()
    x_axis_col = self.plot_xaxis_menu.get()

    if selected_file == "Select a file..." or not x_axis_col:
        return

    # Get data separately to isolate data errors from plot errors
    df = None
    try:
        df = self.get_data_for_plotting(selected_file)
    except Exception as e:
        print(f"Data loading error: {e}")
        self.status_label.configure(text=f"Data error: {str(e)[:50]}...")
        return

    if df is None or df.empty:
        self.plot_ax.clear()
        self.plot_ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        self.plot_canvas.draw_idle()  # Use draw_idle instead of draw
        return

    # Continue with rest of plotting...
    # [Rest of the update_plot method remains the same but with draw_idle() instead of draw()]


# PATCH 4: Add this to the __init__ method after UI creation
# Enable matplotlib interactive mode for better responsiveness
plt.ion()

# PATCH 5: Modify all canvas.draw() calls to canvas.draw_idle()
# This prevents blocking and improves responsiveness
# Search and replace: self.plot_canvas.draw() -> self.plot_canvas.draw_idle()


# PATCH 6: Add validation to on_plot_file_select method
def on_plot_file_select(self, selected_file):
    """Handle file selection with validation."""
    if selected_file == "Select a file...":
        return

    # Validate file exists in loaded data
    if selected_file not in self.loaded_files:
        print(f"ERROR: Selected file {selected_file} not in loaded files")
        self.status_label.configure(text="Error: File not found in loaded data")
        return

    # Original method code continues here...
    # Update signal checkboxes based on selected file


# PATCH 7: Add this method to handle stuck plots
def force_plot_refresh(self):
    """Force a complete refresh of the plot."""
    if hasattr(self, "plot_ax"):
        self.plot_ax.clear()

    if hasattr(self, "plot_canvas"):
        self.plot_canvas.draw_idle()

    # Re-run update_plot
    self.update_plot()


# PATCH 8: Add a "Force Refresh" button in the plotting UI
# In create_plotting_tab method, add:
# ctk.CTkButton(plot_control_frame, text="Force Refresh",
#               command=self.force_plot_refresh).grid(row=0, column=7, padx=5, pady=10)
