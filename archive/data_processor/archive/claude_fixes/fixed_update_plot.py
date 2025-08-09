def update_plot(self):
    """Update the plot with fixed error handling and canvas management."""
    # Check if plot canvas is initialized
    if not hasattr(self, "plot_canvas") or not hasattr(self, "plot_ax"):
        print("Warning: Plot canvas not initialized")
        return

    selected_file = self.plot_file_menu.get()
    x_axis_col = self.plot_xaxis_menu.get()

    if selected_file == "Select a file..." or not x_axis_col:
        return

    # Clear the plot first
    self.plot_ax.clear()

    # Get data with specific error handling
    df = None
    try:
        df = self.get_data_for_plotting(selected_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        self.plot_ax.text(
            0.5,
            0.5,
            f"Error loading data:\n{str(e)}",
            ha="center",
            va="center",
            wrap=True,
        )
        self.plot_canvas.draw()
        self.status_label.configure(text="Error loading data - check console")
        return

    if df is None or df.empty:
        self.plot_ax.text(
            0.5, 0.5, "No data available for plotting", ha="center", va="center"
        )
        self.plot_canvas.draw()
        return

    # Validate x_axis_col
    if x_axis_col not in df.columns:
        if len(df.columns) > 0:
            x_axis_col = df.columns[0]
            self.plot_xaxis_menu.set(x_axis_col)
        else:
            self.plot_ax.text(
                0.5,
                0.5,
                "No valid columns found for plotting.",
                ha="center",
                va="center",
            )
            self.plot_canvas.draw()
            return

    # Get signals to plot
    signals_to_plot = [
        s for s, data in self.plot_signal_vars.items() if data["var"].get()
    ]

    if not signals_to_plot:
        self.plot_ax.text(
            0.5, 0.5, "Select one or more signals to plot", ha="center", va="center"
        )
        self.plot_canvas.draw()
        return

    # Now do the actual plotting with more granular error handling
    try:
        # Check if we should show both raw and filtered signals
        show_both = self.show_both_signals_var.get()
        plot_filter = self.plot_filter_type.get()

        # Chart customization
        plot_style = self.plot_type_var.get()
        style_args = {"linestyle": "-", "marker": ""}
        if plot_style == "Line with Markers":
            style_args = {"linestyle": "-", "marker": ".", "markersize": 4}
        elif plot_style == "Markers Only (Scatter)":
            style_args = {"linestyle": "None", "marker": "."}

        # Apply filter if needed
        plot_df = df.copy()
        if plot_filter != "None" and not show_both:
            try:
                plot_df = self._apply_plot_filter(plot_df, signals_to_plot, x_axis_col)
            except Exception as e:
                print(f"Warning: Filter failed - {e}")
                # Continue with unfiltered data

        # Plot signals
        for i, signal in enumerate(signals_to_plot):
            if signal not in plot_df.columns:
                print(f"Warning: Signal {signal} not found in data")
                continue

            signal_data = plot_df[[x_axis_col, signal]].dropna()
            if len(signal_data) == 0:
                print(f"Warning: No valid data for signal {signal}")
                continue

            try:
                # Get color
                color_scheme = self.color_scheme_var.get()
                if color_scheme == "Default":
                    color = plt.cm.tab10(i % 10)
                elif color_scheme == "Colorblind-friendly":
                    colors = [
                        "#0173B2",
                        "#DE8F05",
                        "#029E73",
                        "#CC78BC",
                        "#CA9161",
                        "#FBAFE4",
                        "#949494",
                        "#ECE133",
                        "#56B4E9",
                    ]
                    color = colors[i % len(colors)]
                else:
                    color = self.custom_colors[i % len(self.custom_colors)]

                # Plot with custom legend if available
                label = self.custom_legend_entries.get(signal, signal)
                line_width = self.line_width_var.get()

                self.plot_ax.plot(
                    signal_data[x_axis_col],
                    signal_data[signal],
                    label=label,
                    color=color,
                    linewidth=line_width,
                    **style_args,
                )

                # Show both raw and filtered if requested
                if show_both and plot_filter != "None":
                    raw_label = f"{label} (raw)"
                    self.plot_ax.plot(
                        df[x_axis_col],
                        df[signal],
                        label=raw_label,
                        color=color,
                        alpha=0.3,
                        linewidth=line_width * 0.7,
                    )

            except Exception as e:
                print(f"Error plotting signal {signal}: {e}")
                continue

        # Add trendline if configured
        try:
            trendline_signal = self.trendline_signal_var.get()
            trendline_type = self.trendline_type_var.get()

            if trendline_signal != "None" and trendline_signal in signals_to_plot:
                self._add_trendline(
                    plot_df, x_axis_col, trendline_signal, trendline_type
                )
        except Exception as e:
            print(f"Warning: Trendline failed - {e}")

        # Configure plot appearance
        title = self.plot_title_entry.get() or f"Signals from {selected_file}"
        xlabel = self.plot_xlabel_entry.get() or x_axis_col
        ylabel = self.plot_ylabel_entry.get() or "Value"

        self.plot_ax.set_title(title, fontsize=14)
        self.plot_ax.set_xlabel(xlabel)
        self.plot_ax.set_ylabel(ylabel)

        # Legend
        legend_position = self.legend_position_var.get()
        if legend_position == "outside right":
            self.plot_ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            self.plot_ax.legend(loc=legend_position)

        self.plot_ax.grid(True, linestyle="--", alpha=0.6)

        # Format datetime axis if applicable
        if pd.api.types.is_datetime64_any_dtype(df[x_axis_col]):
            self.plot_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            self.plot_ax.tick_params(axis="x", rotation=0)

        # Apply saved zoom state if available
        zoom_state = getattr(self, "saved_zoom_state", None)
        if zoom_state:
            try:
                self._apply_zoom_state(zoom_state)
            except Exception as e:
                print(f"Warning: Could not apply zoom state - {e}")

        # Force canvas update
        self.plot_canvas.draw_idle()
        self.status_label.configure(text="Plot updated successfully")

    except Exception as e:
        print(f"Error in plotting: {e}")
        import traceback

        traceback.print_exc()

        # Show error on plot
        self.plot_ax.clear()
        self.plot_ax.text(
            0.5,
            0.5,
            f"Error creating plot:\n{str(e)}",
            ha="center",
            va="center",
            wrap=True,
        )
        self.plot_canvas.draw()
        self.status_label.configure(text="Plot error - check console for details")
