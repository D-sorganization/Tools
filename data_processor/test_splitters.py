#!/usr/bin/env python3
"""
Test script to verify splitter functionality and layout persistence.
"""

import tkinter as tk
import customtkinter as ctk
import json
import os

class TestSplitterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Layout persistence variables
        self.layout_config_file = os.path.join(os.path.expanduser("~"), ".test_splitter_layout.json")
        self.splitters = {}
        self.layout_data = self._load_layout_config()
        
        self.title("Splitter Test")
        self.geometry("800x600")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Set up closing handler
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Create main tab view
        self.main_tab_view = ctk.CTkTabview(self)
        self.main_tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.main_tab_view.add("Test Tab")
        self.create_test_tab(self.main_tab_view.tab("Test Tab"))
        
        # Status bar
        self.status_label = ctk.CTkLabel(self, text="Ready - Test the splitters!")
        self.status_label.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
    
    def create_test_tab(self, tab):
        """Create a test tab with splitters."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        def create_left_content(left_panel):
            """Create left panel content."""
            left_panel.grid_rowconfigure(0, weight=1)
            left_panel.grid_columnconfigure(0, weight=1)
            
            # Add some test content
            test_frame = ctk.CTkFrame(left_panel)
            test_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            test_frame.grid_columnconfigure(0, weight=1)
            
            ctk.CTkLabel(test_frame, text="Left Panel", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=10)
            ctk.CTkLabel(test_frame, text="This is the left panel content.\nTry dragging the splitter handle\nto resize this panel.").grid(row=1, column=0, padx=10, pady=10)
            
            # Add a button to test functionality
            ctk.CTkButton(test_frame, text="Test Button", command=self._test_button_click).grid(row=2, column=0, padx=10, pady=10)
        
        def create_right_content(right_panel):
            """Create right panel content."""
            right_panel.grid_rowconfigure(0, weight=1)
            right_panel.grid_columnconfigure(0, weight=1)
            
            # Add some test content
            test_frame = ctk.CTkFrame(right_panel)
            test_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            test_frame.grid_columnconfigure(0, weight=1)
            
            ctk.CTkLabel(test_frame, text="Right Panel", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=10)
            ctk.CTkLabel(test_frame, text="This is the right panel content.\nThe splitter position should be\nsaved and restored when you\nrestart the application.").grid(row=1, column=0, padx=10, pady=10)
            
            # Add a text area to show current splitter width
            self.width_label = ctk.CTkLabel(test_frame, text="Left panel width: Loading...")
            self.width_label.grid(row=2, column=0, padx=10, pady=10)
            
            # Update width display
            self._update_width_display()
        
        # Create the splitter
        splitter_frame = self._create_splitter(tab, create_left_content, create_right_content, 'test_left_width', 300)
        splitter_frame.grid(row=0, column=0, sticky="nsew")
    
    def _create_splitter(self, parent, left_creator, right_creator, splitter_key, default_left_width):
        """Create a splitter with left and right panels."""
        splitter_frame = ctk.CTkFrame(parent)
        splitter_frame.grid_columnconfigure(2, weight=1)
        splitter_frame.grid_rowconfigure(0, weight=1)
        
        # Get saved width or use default
        left_width = self.layout_data.get(splitter_key, default_left_width)
        
        # Create left panel
        left_panel = ctk.CTkFrame(splitter_frame, width=left_width)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(5, 0))
        left_panel.grid_propagate(False)
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(0, weight=1)
        left_creator(left_panel)
        
        # Create splitter handle
        splitter_handle = ctk.CTkFrame(splitter_frame, width=8, fg_color="#666666")
        splitter_handle.grid(row=0, column=1, sticky="ns", padx=1)
        
        # Bind events for dragging
        splitter_handle.bind("<Enter>", lambda e, h=splitter_handle: self._on_splitter_enter(e, h))
        splitter_handle.bind("<Leave>", lambda e, h=splitter_handle: self._on_splitter_leave(e, h))
        splitter_handle.bind("<Button-1>", lambda e, h=splitter_handle: self._start_splitter_drag(e, h, left_panel, splitter_key))
        splitter_handle.bind("<B1-Motion>", lambda e, h=splitter_handle: self._drag_splitter(e, h, left_panel, splitter_key))
        splitter_handle.bind("<ButtonRelease-1>", lambda e: self._end_splitter_drag())
        
        # Create right panel
        right_panel = ctk.CTkFrame(splitter_frame)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=(0, 5))
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=1)
        right_creator(right_panel)
        
        # Store splitter reference
        self.splitters[splitter_key] = left_panel
        
        return splitter_frame
    
    def _on_splitter_enter(self, event, handle):
        """Handle mouse enter on splitter handle."""
        handle.configure(fg_color="#888888")
        handle.configure(cursor="sb_h_double_arrow")
    
    def _on_splitter_leave(self, event, handle):
        """Handle mouse leave on splitter handle."""
        if not hasattr(self, 'dragging_splitter') or not self.dragging_splitter:
            handle.configure(fg_color="#666666")
    
    def _start_splitter_drag(self, event, handle, left_panel, splitter_key):
        """Start dragging the splitter."""
        self.dragging_splitter = True
        self.drag_splitter_key = splitter_key
        self.drag_left_panel = left_panel
        self.drag_start_x = event.x_root
        self.drag_start_width = left_panel.winfo_width()
        handle.configure(fg_color="#AAAAAA")
    
    def _drag_splitter(self, event, handle, left_panel, splitter_key):
        """Drag the splitter."""
        if hasattr(self, 'dragging_splitter') and self.dragging_splitter:
            delta_x = event.x_root - self.drag_start_x
            new_width = max(150, min(800, self.drag_start_width + delta_x))
            left_panel.configure(width=new_width)
            self._update_width_display()
    
    def _end_splitter_drag(self):
        """End dragging the splitter."""
        if hasattr(self, 'dragging_splitter') and self.dragging_splitter:
            if hasattr(self, 'drag_splitter_key') and hasattr(self, 'drag_left_panel'):
                self.layout_data[self.drag_splitter_key] = self.drag_left_panel.winfo_width()
                self._save_layout_config()
        
        self.dragging_splitter = False
        # Reset handle color
        for splitter_key, splitter in self.splitters.items():
            if hasattr(splitter, 'master') and hasattr(splitter.master, 'winfo_children'):
                for child in splitter.master.winfo_children():
                    if isinstance(child, ctk.CTkFrame) and child.winfo_width() == 8:
                        child.configure(fg_color="#666666")
    
    def _update_width_display(self):
        """Update the width display label."""
        if hasattr(self, 'splitters') and 'test_left_width' in self.splitters:
            width = self.splitters['test_left_width'].winfo_width()
            self.width_label.configure(text=f"Left panel width: {width}px")
    
    def _test_button_click(self):
        """Test button click handler."""
        self.status_label.configure(text="Test button clicked! Splitter functionality is working.")
    
    def _load_layout_config(self):
        """Load layout configuration from file."""
        try:
            if os.path.exists(self.layout_config_file):
                with open(self.layout_config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading layout config: {e}")
        return {}
    
    def _save_layout_config(self):
        """Save layout configuration to file."""
        try:
            self.layout_data['window_width'] = self.winfo_width()
            self.layout_data['window_height'] = self.winfo_height()
            
            for splitter_key, splitter in self.splitters.items():
                if hasattr(splitter, 'winfo_width'):
                    self.layout_data[splitter_key] = splitter.winfo_width()
            
            with open(self.layout_config_file, 'w') as f:
                json.dump(self.layout_data, f, indent=2)
        except Exception as e:
            print(f"Error saving layout config: {e}")
    
    def _on_closing(self):
        """Handle application closing."""
        self._save_layout_config()
        self.quit()

if __name__ == "__main__":
    print("Starting Splitter Test Application...")
    app = TestSplitterApp()
    app.mainloop() 