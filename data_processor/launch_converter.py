#!/usr/bin/env python3
"""
Standalone launcher for the Format Converter tool.
Starts the integrated app and switches focus to the 'Format Converter' tab.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Data_Processor_Integrated import IntegratedCSVProcessorApp

if __name__ == "__main__":
    app = IntegratedCSVProcessorApp()
    try:
        # Ensure the tab exists, then select it
        if hasattr(app, "main_tab_view"):
            try:
                # Some tabviews use .set(label), others use .select(tab)
                app.main_tab_view.set("Format Converter")
            except Exception:
                try:
                    app.main_tab_view.select(app.main_tab_view.tab("Format Converter"))
                except Exception:
                    pass
    except Exception:
        pass
    app.mainloop()
