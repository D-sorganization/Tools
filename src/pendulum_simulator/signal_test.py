#!/usr/bin/env python3
"""Minimal test to verify PyQt signals work."""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PyQt6.QtCore import pyqtSignal

class TestWindow(QMainWindow):
    run_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Test")
        self.setGeometry(100, 100, 300, 200)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        button = QPushButton("Click Me")
        button.clicked.connect(self.run_requested.emit)
        layout.addWidget(button)
        
        self.setCentralWidget(widget)
        
        # Connect our own handler
        self.run_requested.connect(self.on_run)
    
    def on_run(self):
        print("[TEST] Signal received!")
        QMessageBox.information(self, "Success", "Signal was received!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())
