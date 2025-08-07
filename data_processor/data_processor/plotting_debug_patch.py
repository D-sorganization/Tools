
# PLOTTING DEBUGGING ENHANCEMENTS
# Add these methods to the CSVProcessorApp class


    def _debug_plot_state(self):
        """Debug helper to print current plotting state."""
        print("\n=== PLOT DEBUG STATE ===")
        print(f"plot_file_menu: {getattr(self, 'plot_file_menu', None)}")
        if hasattr(self, 'plot_file_menu'):
            print(f"  selected file: {self.plot_file_menu.get()}")
        
        print(f"plot_xaxis_menu: {getattr(self, 'plot_xaxis_menu', None)}")
        if hasattr(self, 'plot_xaxis_menu'):
            print(f"  selected x-axis: {self.plot_xaxis_menu.get()}")
        
        print(f"plot_signal_vars: {getattr(self, 'plot_signal_vars', None)}")
        if hasattr(self, 'plot_signal_vars'):
            print(f"  number of signals: {len(self.plot_signal_vars)}")
            selected = [s for s, data in self.plot_signal_vars.items() if data['var'].get()]
            print(f"  selected signals: {selected}")
        
        print(f"plot_canvas: {getattr(self, 'plot_canvas', None)}")
        print(f"plot_ax: {getattr(self, 'plot_ax', None)}")
        print(f"processed_files: {len(getattr(self, 'processed_files', {})) if hasattr(self, 'processed_files') else 'None'}")
        print(f"loaded_data_cache: {len(getattr(self, 'loaded_data_cache', {})) if hasattr(self, 'loaded_data_cache') else 'None'}")
        print("========================\n")

    def _force_signal_selection(self):
        """Force select at least one signal for debugging."""
        if hasattr(self, 'plot_signal_vars') and self.plot_signal_vars:
            # Check if any signals are selected
            selected = [s for s, data in self.plot_signal_vars.items() if data['var'].get()]
            if not selected:
                # Auto-select first non-time signal
                for signal, data in self.plot_signal_vars.items():
                    if not any(word in signal.lower() for word in ['time', 'date', 'timestamp']):
                        data['var'].set(True)
                        print(f"DEBUG: Force-selected signal: {signal}")
                        break


# REPLACE the existing update_plot method with this enhanced version:

    def update_plot(self, selected_signals=None):
        """The main function to draw/redraw the plot with all selected options."""
        print("\n" + "="*50)
        print("DEBUG: update_plot called")
        print("="*50)
        
        if not hasattr(self, 'plot_canvas') or not hasattr(self, 'plot_ax'):
            print("DEBUG: plot_canvas or plot_ax not found")
            print(f"  plot_canvas exists: {hasattr(self, 'plot_canvas')}")
            print(f"  plot_ax exists: {hasattr(self, 'plot_ax')}")
            return
        
        try:
            # Debug current state
            self._debug_plot_state()
            
            # Clear any previous error messages
            if hasattr(self, 'status_label'):
                self.status_label.configure(text="Updating plot...")
            print(f"DEBUG: Starting plot update. Selected signals: {selected_signals}")
            
            zoom_state = None
            if hasattr(self, '_preserve_zoom_during_update'):
                zoom_state = self._preserve_zoom_during_update()
                
            selected_file = self.plot_file_menu.get() if hasattr(self, 'plot_file_menu') else None
            x_axis_col = self.plot_xaxis_menu.get() if hasattr(self, 'plot_xaxis_menu') else None
            print(f"DEBUG: Selected file: {selected_file}, X-axis: {x_axis_col}")

            if not selected_file or selected_file == "Select a file..." or not x_axis_col:
                print("DEBUG: No file or x-axis selected")
                self.plot_ax.clear()
                self.plot_ax.text(0.5, 0.5, "Please select a file and x-axis column", ha='center', va='center')
                self.plot_canvas.draw()
                return

            df = self.get_data_for_plotting(selected_file)
            if df is None or df.empty:
                print("DEBUG: No data retrieved for plotting")
                self.plot_ax.clear()
                self.plot_ax.text(0.5, 0.5, "Could not load or plot data.", ha='center', va='center')
                self.plot_canvas.draw()
                return
            
            print(f"DEBUG: Data loaded successfully. Shape: {df.shape}, Columns: {list(df.columns)[:5]}")

            if x_axis_col not in df.columns:
                print(f"DEBUG: X-axis column '{x_axis_col}' not in data columns")
                if len(df.columns) > 0:
                    self.plot_xaxis_menu.set(df.columns[0])
                    x_axis_col = df.columns[0]
                    print(f"DEBUG: Set x-axis to first column: {x_axis_col}")
                else:
                    self.plot_ax.clear()
                    self.plot_ax.text(0.5, 0.5, "No valid columns found for plotting.", ha='center', va='center')
                    self.plot_canvas.draw()
                    return

            # Force signal selection if none selected
            self._force_signal_selection()

            signals_to_plot = selected_signals if selected_signals else [s for s, data in self.plot_signal_vars.items() if data['var'].get()]
            print(f"DEBUG: Signals to plot: {signals_to_plot}")
            print(f"DEBUG: plot_signal_vars keys: {list(self.plot_signal_vars.keys())[:5]}")
            
            # Debug: Check which signals are actually selected
            if not selected_signals and hasattr(self, 'plot_signal_vars'):
                print("DEBUG: Checking plot_signal_vars for selected signals:")
                for signal, data in self.plot_signal_vars.items():
                    is_selected = data['var'].get()
                    print(f"  {signal}: {is_selected}")
            
            self.plot_ax.clear()

            if not signals_to_plot:
                print("DEBUG: No signals selected for plotting")
                self.plot_ax.text(0.5, 0.5, "Select one or more signals to plot", ha='center', va='center')
            else:
                print(f"DEBUG: Plotting {len(signals_to_plot)} signals")
                
                # Get plot settings with defaults
                show_both = getattr(self, 'show_both_signals_var', None)
                show_both = show_both.get() if show_both else False
                
                plot_filter = getattr(self, 'plot_filter_type', None)
                plot_filter = plot_filter.get() if plot_filter else "None"
                
                plot_style = getattr(self, 'plot_type_var', None)
                plot_style = plot_style.get() if plot_style else "Line"
                
                print(f"DEBUG: Plot settings - show_both: {show_both}, filter: {plot_filter}, style: {plot_style}")
                
                # Set up plot style
                style_args = {"linestyle": "-", "marker": ""}
                if plot_style == "Line with Markers":
                    style_args = {"linestyle": "-", "marker": ".", "markersize": 4}
                elif plot_style == "Markers Only (Scatter)":
                    style_args = {"linestyle": "None", "marker": ".", "markersize": 5}
                
                line_width = getattr(self, 'line_width_var', None)
                line_width = float(line_width.get()) if line_width else 1.0
                style_args["linewidth"] = line_width
                
                # Set up colors
                color_scheme = getattr(self, 'color_scheme_var', None)
                color_scheme = color_scheme.get() if color_scheme else "Auto (Matplotlib)"
                
                if color_scheme == "Custom Colors" and hasattr(self, 'custom_colors'):
                    colors = [self.custom_colors[i % len(self.custom_colors)] for i in range(len(signals_to_plot))]
                else:
                    cmap = plt.get_cmap("tab10")
                    colors = cmap(np.linspace(0, 1, len(signals_to_plot)))

                print(f"DEBUG: Starting to plot {len(signals_to_plot)} signals")
                plotted_count = 0
                
                for i, signal in enumerate(signals_to_plot):
                    if signal not in df.columns: 
                        print(f"DEBUG: Signal '{signal}' not in data columns, skipping")
                        continue
                    
                    try:
                        plot_df = df[[x_axis_col, signal]].dropna()
                        print(f"DEBUG: Plotting signal '{signal}' with {len(plot_df)} data points")
                        
                        if len(plot_df) == 0:
                            print(f"DEBUG: No data points for signal '{signal}' after dropping NaN")
                            continue
                        
                        plot_style_final = {**style_args, "color": colors[i]}
                        signal_label = getattr(self, 'custom_legend_entries', {}).get(signal, signal)
                        
                        self.plot_ax.plot(plot_df[x_axis_col], plot_df[signal], label=signal_label, **plot_style_final)
                        plotted_count += 1
                        print(f"DEBUG: Successfully plotted signal '{signal}'")
                        
                    except Exception as e:
                        print(f"DEBUG: Error plotting signal '{signal}': {e}")
                        continue

                print(f"DEBUG: Successfully plotted {plotted_count} out of {len(signals_to_plot)} signals")

            # Set plot labels and title
            title = getattr(self, 'plot_title_entry', None)
            title = title.get() if title else f"Signals from {selected_file}"
            
            xlabel = getattr(self, 'plot_xlabel_entry', None)
            xlabel = xlabel.get() if xlabel else x_axis_col
            
            ylabel = getattr(self, 'plot_ylabel_entry', None)
            ylabel = ylabel.get() if ylabel else "Value"
            
            self.plot_ax.set_title(title, fontsize=14)
            self.plot_ax.set_xlabel(xlabel)
            self.plot_ax.set_ylabel(ylabel)

            # Add legend
            legend_position = getattr(self, 'legend_position_var', None)
            legend_position = legend_position.get() if legend_position else "best"
            
            if legend_position == "outside right":
                self.plot_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                self.plot_ax.legend(loc=legend_position)
            
            self.plot_ax.grid(True, linestyle='--', alpha=0.6)

            # Handle datetime formatting
            if hasattr(pd.api.types, 'is_datetime64_any_dtype') and pd.api.types.is_datetime64_any_dtype(df[x_axis_col]):
                self.plot_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                self.plot_ax.tick_params(axis='x', rotation=0)

            # Restore zoom if available
            if zoom_state and hasattr(self, '_apply_zoom_state'):
                self._apply_zoom_state(zoom_state)

            self.plot_canvas.draw()
            
            if hasattr(self, 'status_label'):
                self.status_label.configure(text="Plot updated successfully")
            
            print("DEBUG: Plot update completed successfully")
            
        except Exception as e:
            print(f"ERROR in update_plot: {e}")
            import traceback
            traceback.print_exc()
            
            self.plot_ax.clear()
            self.plot_ax.text(0.5, 0.5, f"Error plotting data:\n{str(e)}", 
                             ha='center', va='center', wrap=True)
            self.plot_canvas.draw()
            
            if hasattr(self, 'status_label'):
                self.status_label.configure(text="Plot error - check console for details")

