# Note: Qt imported but unused; remove to satisfy linter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListView,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
from threads import FolderProcessingThread


class FolderToolTab(QWidget):
    """Tab widget encapsulating folder operations."""

    def __init__(self, status_bar=None, parent=None):
        super().__init__(parent)
        self.status_bar = status_bar
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Source folders group
        source_group = QGroupBox("Source Folders")
        source_layout = QVBoxLayout(source_group)

        source_buttons_layout = QHBoxLayout()
        self.folder_select_source_btn = QPushButton("Add Folders")
        self.folder_select_source_btn.clicked.connect(self.folder_select_source_folders)
        source_buttons_layout.addWidget(self.folder_select_source_btn)

        self.folder_clear_source_btn = QPushButton("Clear All")
        self.folder_clear_source_btn.clicked.connect(self.folder_clear_source_folders)
        source_buttons_layout.addWidget(self.folder_clear_source_btn)

        source_layout.addLayout(source_buttons_layout)

        self.folder_source_list = QListWidget()
        source_layout.addWidget(self.folder_source_list)
        layout.addWidget(source_group)

        # Operation selection
        operation_group = QGroupBox("Operation")
        operation_layout = QVBoxLayout(operation_group)

        from PyQt6.QtWidgets import (  # imported here to avoid circular import ordering
            QComboBox,
        )

        self.folder_operation_combo = QComboBox()
        self.folder_operation_combo.addItems(
            ["Combine", "Flatten", "Prune", "Deduplicate", "Analyze"]
        )
        self.folder_operation_combo.currentTextChanged.connect(
            self.on_folder_operation_changed
        )
        operation_layout.addWidget(self.folder_operation_combo)
        layout.addWidget(operation_group)

        # Destination folder
        dest_group = QGroupBox("Destination")
        dest_layout = QHBoxLayout(dest_group)

        self.folder_dest_edit = QLineEdit()
        self.folder_dest_edit.setPlaceholderText("Select destination folder...")
        dest_layout.addWidget(self.folder_dest_edit)

        self.folder_select_dest_btn = QPushButton("Browse")
        self.folder_select_dest_btn.clicked.connect(self.folder_select_dest_folder)
        dest_layout.addWidget(self.folder_select_dest_btn)
        layout.addWidget(dest_group)

        # Run button and progress
        self.folder_run_btn = QPushButton("Run Operation")
        self.folder_run_btn.clicked.connect(self.folder_run_processing)
        layout.addWidget(self.folder_run_btn)

        self.folder_progress = QProgressBar()
        layout.addWidget(self.folder_progress)

    # ------------------------------------------------------------------
    # Slot implementations
    # ------------------------------------------------------------------
    def folder_select_source_folders(self):
        """Select source folders for folder operations."""
        dialog = QFileDialog(self, "Select Source Folders")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        list_view = dialog.findChild(QListView, "listView")
        if list_view:
            list_view.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        tree_view = dialog.findChild(QTreeView)
        if tree_view:
            tree_view.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

        if dialog.exec():
            for folder in dialog.selectedFiles():
                self.folder_source_list.addItem(folder)

    def folder_clear_source_folders(self):
        """Clear all source folders."""
        self.folder_source_list.clear()

    def on_folder_operation_changed(self, operation):
        """Handle folder operation change."""
        requires_dest = operation in {"Combine", "Flatten", "Prune", "Deduplicate"}
        self.folder_dest_edit.setEnabled(requires_dest)
        self.folder_select_dest_btn.setEnabled(requires_dest)
        if not requires_dest:
            self.folder_dest_edit.clear()
        if self.status_bar is not None:
            self.status_bar.showMessage(f"Operation set to {operation}")

    def folder_select_dest_folder(self):
        """Select destination folder for folder operations."""
        folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        if folder:
            self.folder_dest_edit.setText(folder)

    def folder_run_processing(self):
        """Run the folder processing operation."""
        if self.folder_source_list.count() == 0:
            QMessageBox.warning(self, "Warning", "Please select source folders.")
            return

        operation = self.folder_operation_combo.currentText()
        if operation != "Analyze" and not self.folder_dest_edit.text():
            QMessageBox.warning(self, "Warning", "Please select a destination folder.")
            return

        source_folders = self.get_folder_source_list()
        dest_folder = self.folder_dest_edit.text()

        self.folder_thread = FolderProcessingThread(
            source_folders, dest_folder, operation
        )
        self.folder_thread.progress_updated.connect(self.update_folder_progress)
        self.folder_thread.finished.connect(self.folder_processing_finished)
        self.folder_thread.start()

    def get_folder_source_list(self):
        """Get list of source folders from widget."""
        return [
            self.folder_source_list.item(i).text()
            for i in range(self.folder_source_list.count())
        ]

    def update_folder_progress(self, value):
        """Update folder processing progress bar."""
        self.folder_progress.setValue(value)

    def folder_processing_finished(self):
        """Handle folder processing completion."""
        QMessageBox.information(self, "Complete", "Folder processing completed!")
        if self.status_bar is not None:
            self.status_bar.showMessage("Folder processing completed")
