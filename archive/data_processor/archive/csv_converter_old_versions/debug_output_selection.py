#!/usr/bin/env python3
"""
Debug script to test output selection logic
"""

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QCheckBox, QLabel

def test_output_selection():
    app = QApplication(sys.argv)
    
    widget = QWidget()
    layout = QVBoxLayout()
    
    # Create checkbox
    checkbox = QCheckBox("Combine all files into single Parquet file")
    checkbox.setChecked(True)
    
    # Create label to show state
    state_label = QLabel("Checkbox state: True")
    
    # Create button to test output selection
    def test_browse():
        checked = checkbox.isChecked()
        state_label.setText(f"Checkbox state: {checked}")
        
        if checked:
            print("Should show file save dialog")
            from PyQt6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                widget, "Save Parquet File", "", "Parquet Files (*.parquet)"
            )
            if file_path:
                print(f"Selected file: {file_path}")
        else:
            print("Should show folder selection dialog")
            from PyQt6.QtWidgets import QFileDialog
            folder_path = QFileDialog.getExistingDirectory(
                widget, "Select Output Folder", ""
            )
            if folder_path:
                print(f"Selected folder: {folder_path}")
    
    browse_btn = QPushButton("Test Browse Output")
    browse_btn.clicked.connect(test_browse)
    
    # Create button to toggle checkbox
    def toggle_checkbox():
        checkbox.setChecked(not checkbox.isChecked())
        state_label.setText(f"Checkbox state: {checkbox.isChecked()}")
    
    toggle_btn = QPushButton("Toggle Checkbox")
    toggle_btn.clicked.connect(toggle_checkbox)
    
    layout.addWidget(checkbox)
    layout.addWidget(state_label)
    layout.addWidget(browse_btn)
    layout.addWidget(toggle_btn)
    
    widget.setLayout(layout)
    widget.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    test_output_selection()
