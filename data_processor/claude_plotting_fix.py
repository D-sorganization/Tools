    def _add_trendline(self, df, signal, x_axis_col):
        """Add trendline to the plot."""
        trend_type = self.trendline_type_var.get()
        
        if trend_type == "None":
            return
            
        plot_df = df[[x_axis_col, signal]].dropna()
        if len(plot_df) < 2:
            return
        
        # Apply time window filtering based on selected mode
        window_mode = self.trendline_window_mode.get()
        
        if window_mode == "Manual Entry":
            start_str = self.trendline_start_entry.get().strip()
            end_str = self.trendline_end_entry.get().strip()
            
            if start_str or end_str:
                try:
                    if pd.api.types.is_datetime64_any_dtype(plot_df[x_axis_col]):
                        # Convert to datetime
                        if start_str:
                            start_time = pd.to_datetime(start_str)
                            plot_df = plot_df[plot_df[x_axis_col] >= start_time]
                        if end_str:
                            end_time = pd.to_datetime(end_str)
                            plot_df = plot_df[plot_df[x_axis_col] <= end_time]
                    else:
                        # Numeric data
                        if start_str:
                            start_val = float(start_str)
                            plot_df = plot_df[plot_df[x_axis_col] >= start_val]
                        if end_str:
                            end_val = float(end_str)
                            plot_df = plot_df[plot_df[x_axis_col] <= end_val]
                except (ValueError, TypeError):
                    messagebox.showwarning("Warning", "Invalid time window format. Using full range.")
        
        elif window_mode == "Visual Selection":
            if hasattr(self, 'trendline_selection_start') and hasattr(self, 'trendline_selection_end'):
                if self.trendline_selection_start is not None and self.trendline_selection_end is not None:
                    try:
                        if pd.api.types.is_datetime64_any_dtype(plot_df[x_axis_col]):
                            # Convert numeric selection back to datetime
                            x_min = plot_df[x_axis_col].min()
                            start_time = x_min + pd.Timedelta(seconds=self.trendline_selection_start)
                            end_time = x_min + pd.Timedelta(seconds=self.trendline_selection_end)
                            plot_df = plot_df[(plot_df[x_axis_col] >= start_time) & (plot_df[x_axis_col] <= end_time)]
                        else:
                            # Numeric data
                            plot_df = plot_df[(plot_df[x_axis_col] >= self.trendline_selection_start) & 
                                            (plot_df[x_axis_col] <= self.trendline_selection_end)]
                    except Exception as e:
                        print(f"Error applying visual selection: {e}")
        
        # Check if we still have enough data after filtering
        if len(plot_df) < 2:
            messagebox.showwarning("Warning", "Not enough data points in selected time window for trendline.")
            return
            
        x_data = plot_df[x_axis_col].values
        y_data = plot_df[signal].values
        
        # Convert datetime to numeric for fitting
        if pd.api.types.is_datetime64_any_dtype(plot_df[x_axis_col]):
            x_numeric = (plot_df[x_axis_col] - plot_df[x_axis_col].min()).dt.total_seconds().values
        else:
            x_numeric = x_data.astype(float)
            
        try:
            if trend_type == "Linear":
                coeffs = np.polyfit(x_numeric, y_data, 1)
                trend = np.poly1d(coeffs)
                trendline = trend(x_numeric)
                equation = f"y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}"
                
            elif trend_type == "Exponential":
                # Log-linear fit for exponential
                y_positive = y_data[y_data > 0]
                x_positive = x_numeric[y_data > 0]
                if len(y_positive) > 1:
                    log_y = np.log(y_positive)
                    coeffs = np.polyfit(x_positive, log_y, 1)
                    a = np.exp(coeffs[1])
                    b = coeffs[0]
                    trendline = a * np.exp(b * x_numeric)
                    equation = f"y = {a:.4f} * e^({b:.4f}x)"
                else:
                    messagebox.showwarning("Warning", "Not enough positive values for exponential trendline.")
                    return
                    
            elif trend_type == "Power":
                # Log-log fit for power law
                mask = (y_data > 0) & (x_numeric > 0)
                y_positive = y_data[mask]
                x_positive = x_numeric[mask]
                if len(y_positive) > 1:
                    log_x = np.log(x_positive)
                    log_y = np.log(y_positive)
                    coeffs = np.polyfit(log_x, log_y, 1)
                    a = np.exp(coeffs[1])
                    b = coeffs[0]
                    trendline = a * (x_numeric ** b)
                    equation = f"y = {a:.4f} * x^({b:.4f})"
                else:
                    messagebox.showwarning("Warning", "Not enough positive values for power trendline.")
                    return
                    
            elif trend_type == "Polynomial":
                order = int(self.poly_order_entry.get() or "2")
                order = max(2, min(6, order))  # Limit to 2-6
                coeffs = np.polyfit(x_numeric, y_data, order)
                trend = np.poly1d(coeffs)
                trendline = trend(x_numeric)
                # Build equation string
                terms = []
                for i in range(order+1):
                    power = order - i
                    coeff = coeffs[i]
                    if power == 0:
                        terms.append(f"{coeff:.4f}")
                    elif power == 1:
                        terms.append(f"{coeff:.4f}x")
                    else:
                        terms.append(f"{coeff:.4f}x^{power}")
                equation = f"Polynomial (order {order}): " + " + ".join(terms)
            
            # Plot trendline - use the original x_data for plotting
            self.plot_ax.plot(plot_df[x_axis_col], trendline, '--', color='red', linewidth=2, 
                            label=f'{signal} Trendline ({trend_type})', alpha=0.8)
            
            # Force redraw the legend
            handles, labels = self.plot_ax.get_legend_handles_labels()
            legend_position = self.legend_position_var.get()
            if legend_position == "outside right":
                self.plot_ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                self.plot_ax.legend(handles, labels, loc=legend_position)
            
            # Update trendline textbox
            self.trendline_textbox.delete("1.0", tk.END)
            self.trendline_textbox.insert("1.0", equation)
            
            # Redraw the canvas
            self.plot_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Trendline Error", f"Error adding trendline: {str(e)}")
            print(f"Error adding trendline: {e}")
            import traceback
            traceback.print_exc()