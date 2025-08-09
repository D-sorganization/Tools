#!/usr/bin/env python3
"""
Simple test to launch the Data Processor and test basic plotting functionality
"""

print("Testing Data Processor launch and plotting components...")

try:
    import Data_Processor_r0

    print("✓ Module imported successfully")

    # Create app instance
    app = Data_Processor_r0.CSVProcessorApp()
    print("✓ App instance created")

    # Check if plotting components exist
    has_canvas = hasattr(app, "plot_canvas")
    has_ax = hasattr(app, "plot_ax")
    has_fig = hasattr(app, "plot_fig")

    print(f"plot_canvas exists: {has_canvas}")
    print(f"plot_ax exists: {has_ax}")
    print(f"plot_fig exists: {has_fig}")

    if has_canvas and has_ax:
        print("✓ Plotting components initialized successfully")

        # Test if we can modify the plot
        try:
            app.plot_ax.clear()
            app.plot_ax.text(
                0.5, 0.5, "Test from external script", ha="center", va="center"
            )
            app.plot_ax.set_title("External Test Plot")
            app.plot_canvas.draw()
            print("✓ Successfully modified plot from external script")
        except Exception as e:
            print(f"✗ Failed to modify plot: {e}")
    else:
        print("✗ Plotting components not properly initialized")

    # Check if file selection works
    if hasattr(app, "input_file_paths"):
        print(f"input_file_paths exists: {len(app.input_file_paths)} files")

    # Set up some test data if no files loaded
    import os

    data_dir = "Half Ton Data"
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        if csv_files:
            test_file = os.path.join(data_dir, csv_files[0])
            app.input_file_paths = [test_file]
            print(f"✓ Set test file: {csv_files[0]}")

            # Test file selection in plotting
            if hasattr(app, "plot_file_menu"):
                app.plot_file_menu.configure(values=[csv_files[0]])
                app.plot_file_menu.set(csv_files[0])
                print("✓ Set plot file menu")

                # Try to trigger file selection
                try:
                    print("Attempting to trigger file selection...")
                    app.on_plot_file_select(csv_files[0])
                    print("✓ File selection completed")

                    # Check if signals were created
                    if hasattr(app, "plot_signal_vars") and app.plot_signal_vars:
                        selected_count = sum(
                            1
                            for data in app.plot_signal_vars.values()
                            if data["var"].get()
                        )
                        print(
                            f"✓ plot_signal_vars created with {len(app.plot_signal_vars)} signals"
                        )
                        print(f"✓ {selected_count} signals selected")

                        if selected_count > 0:
                            print("Attempting manual plot update...")
                            app.update_plot()
                            print("✓ Manual plot update completed")
                        else:
                            print("⚠ No signals selected for plotting")
                    else:
                        print("✗ plot_signal_vars not created")

                except Exception as e:
                    print(f"✗ File selection failed: {e}")
                    import traceback

                    traceback.print_exc()

    print("\nRunning application... Close the window to continue.")
    app.mainloop()
    print("✓ Application closed successfully")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
