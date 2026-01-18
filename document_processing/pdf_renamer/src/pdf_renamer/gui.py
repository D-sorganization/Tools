"""Modern PyQt6 GUI for PDF Renamer."""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .api_mode import APIRenameManager, RenameProposal
from .cache import ResultCache
from .config import get_user_preferences, save_user_preferences, update_last_directory
from .deduper import DuplicateFinder
from .llm_layer import GeminiTitleLLM
from .transaction_log import TransactionLog
from .worker import process_single_file

logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """Background thread for PDF processing."""

    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    log_message = pyqtSignal(str, str)  # message, level
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(
        self,
        directory: Path,
        dry_run: bool,
        style: str,
        delete_dups: bool,
        use_llm: bool,
        recursive: bool,
        workers: int,
        db_path: Path,
        include_author: bool = False,
        move_failed: bool = True,
        failed_folder: str = "failed_renames",
    ):
        super().__init__()
        self.directory = directory
        self.dry_run = dry_run
        self.style = style
        self.delete_dups = delete_dups
        self.use_llm = use_llm
        self.recursive = recursive
        self.workers = workers
        self.db_path = db_path
        self.include_author = include_author
        self.move_failed = move_failed
        self.failed_folder = failed_folder
        self._is_cancelled = False

    def cancel(self) -> None:
        """Cancel the processing."""
        self._is_cancelled = True

    def run(self) -> None:
        """Run the processing in background thread."""
        try:
            self.log_message.emit(f"Starting processing in: {self.directory}", "INFO")
            self.log_message.emit(
                f"Configuration: Style={self.style}, DryRun={self.dry_run}, "
                f"DeleteDups={self.delete_dups}, UseLLM={self.use_llm}, "
                f"Recursive={self.recursive}, Workers={self.workers}",
                "INFO",
            )

            # 1. Find and handle duplicates
            if self._is_cancelled:
                return

            self.log_message.emit("Scanning for duplicates...", "INFO")
            finder = DuplicateFinder(self.directory, recursive=self.recursive)
            duplicates = finder.find_duplicates()

            if duplicates:
                self.log_message.emit(
                    f"Found {len(duplicates)} sets of duplicates", "WARNING"
                )

                for _file_hash, paths in duplicates.items():
                    if self._is_cancelled:
                        return  # type: ignore[unreachable]

                    if self.delete_dups:
                        keep = paths[0]
                        to_delete = paths[1:]
                        self.log_message.emit(f"Keeping: {keep.name}", "SUCCESS")

                        for p in to_delete:
                            if self.dry_run:
                                self.log_message.emit(
                                    f"[DRY RUN] Would delete: {p.name}", "WARNING"
                                )
                            else:
                                try:
                                    p.unlink()
                                    self.log_message.emit(
                                        f"Deleted duplicate: {p.name}", "WARNING"
                                    )
                                except Exception as e:
                                    self.log_message.emit(
                                        f"Failed to delete {p.name}: {e}", "ERROR"
                                    )
                    else:
                        self.log_message.emit(
                            f"Duplicate set: {[p.name for p in paths]}", "WARNING"
                        )
            else:
                self.log_message.emit("No duplicates found", "SUCCESS")

            # 2. Process PDF files
            if self._is_cancelled:
                return  # type: ignore[unreachable]

            self.log_message.emit("Scanning for PDF files...", "INFO")
            pattern = "**/*.pdf" if self.recursive else "*.pdf"
            pdf_files = list(self.directory.glob(pattern))
            # Filter out symlinks
            pdf_files = [f for f in pdf_files if f.is_file() and not f.is_symlink()]
            total_files = len(pdf_files)

            if total_files == 0:
                self.log_message.emit("No PDF files found", "WARNING")
                self.finished.emit(True, "No files to process")
                return

            self.log_message.emit(
                f"Found {total_files} PDF files. Starting processing...", "INFO"
            )

            # Initialize components
            cache = ResultCache(self.db_path)
            transaction_log = TransactionLog()
            llm = GeminiTitleLLM() if self.use_llm else None

            if self.use_llm and llm and not llm.genai:
                self.log_message.emit(
                    "LLM requested but not available. Falling back to local extraction.",
                    "WARNING",
                )
                llm = None

            # Process files in parallel
            processed_count = 0
            success_count = 0
            fail_count = 0

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(
                        process_single_file,
                        pdf_file,
                        cache,
                        transaction_log,
                        llm,
                        self.dry_run,
                        self.style,
                        self.include_author,
                        self.move_failed,
                        self.failed_folder,
                    ): pdf_file
                    for pdf_file in pdf_files
                }

                # Process results as they complete
                for future in as_completed(future_to_file):
                    if self._is_cancelled:
                        return  # type: ignore[unreachable]

                    processed_count += 1
                    pdf_file = future_to_file[future]

                    try:
                        result = future.result()

                        # Determine log level
                        if result.success:
                            level = "SUCCESS"
                            success_count += 1
                        else:
                            level = "ERROR"
                            fail_count += 1

                        self.log_message.emit(result.message, level)

                    except Exception as e:
                        fail_count += 1
                        self.log_message.emit(
                            f"Executor failed for {pdf_file.name}: {e}", "ERROR"
                        )

                    self.progress_updated.emit(
                        processed_count,
                        total_files,
                        f"Processed {processed_count}/{total_files} files",
                    )

            # Summary
            summary = (
                f"Processing complete! {success_count} succeeded, {fail_count} failed"
            )
            self.log_message.emit(summary, "SUCCESS")
            self.finished.emit(True, summary)

        except Exception as e:
            error_msg = f"Critical error: {e}"
            self.log_message.emit(error_msg, "ERROR")
            self.finished.emit(False, error_msg)


class PDFRenamerGUI(QMainWindow):
    """Main GUI window for PDF Renamer."""

    def __init__(self) -> None:
        super().__init__()
        self.processing_thread: ProcessingThread | None = None
        self.preferences = get_user_preferences()
        self.api_manager: APIRenameManager | None = None
        self.init_ui()
        self.load_preferences()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("PDF Renamer Pro - AI-Powered Document Management")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        header_label = QLabel("📄 PDF Renamer Pro")
        header_font = QFont("Arial", 20, QFont.Weight.Bold)
        header_label.setFont(header_font)
        header_label.setStyleSheet(
            "color: #2c3e50; padding: 15px; background: #ecf0f1; border-radius: 5px;"
        )
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_label)

        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_batch_tab()
        self.create_api_tab()
        self.create_settings_tab()

    def create_batch_tab(self) -> None:
        """Create the batch processing tab."""
        batch_widget = QWidget()
        layout = QVBoxLayout(batch_widget)

        # Directory selection
        dir_group = QGroupBox("📁 Directory Selection")
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select directory containing PDF files...")
        dir_layout.addWidget(self.dir_input)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.browse_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Processing settings
        settings_group = QGroupBox("⚙️ Processing Settings")
        settings_layout = QVBoxLayout()

        # Naming style
        style_layout = QHBoxLayout()
        style_label = QLabel("Naming Style:")
        style_layout.addWidget(style_label)
        self.style_standard = QRadioButton("Standard (Title.pdf)")
        self.style_standard.setChecked(True)
        style_layout.addWidget(self.style_standard)
        self.style_snake = QRadioButton("Snake Case (title_here.pdf)")
        style_layout.addWidget(self.style_snake)
        self.style_kebab = QRadioButton("Kebab Case (title-here.pdf)")
        style_layout.addWidget(self.style_kebab)
        style_layout.addStretch()
        settings_layout.addLayout(style_layout)

        # Options Row 1
        options_layout = QHBoxLayout()
        self.dry_run_check = QCheckBox("🔍 Dry Run (Preview only)")
        self.dry_run_check.setChecked(True)
        options_layout.addWidget(self.dry_run_check)
        self.delete_dups_check = QCheckBox("🗑️ Delete Duplicates")
        options_layout.addWidget(self.delete_dups_check)
        self.use_llm_check = QCheckBox("🤖 Use AI (Gemini)")
        options_layout.addWidget(self.use_llm_check)
        self.recursive_check = QCheckBox("📂 Include Subfolders")
        self.recursive_check.setChecked(True)
        options_layout.addWidget(self.recursive_check)
        options_layout.addStretch()
        settings_layout.addLayout(options_layout)

        # Options Row 2
        options_layout2 = QHBoxLayout()
        self.include_author_check = QCheckBox("👤 Include Author (Author - Title.pdf)")
        options_layout2.addWidget(self.include_author_check)
        self.move_failed_check = QCheckBox("📁 Move Failed Files to Subfolder")
        self.move_failed_check.setChecked(True)
        options_layout2.addWidget(self.move_failed_check)
        options_layout2.addStretch()
        settings_layout.addLayout(options_layout2)

        # Workers and failed folder
        workers_layout = QHBoxLayout()
        workers_label = QLabel("Parallel Workers:")
        workers_layout.addWidget(workers_label)
        self.workers_spin = QSpinBox()
        self.workers_spin.setMinimum(1)
        self.workers_spin.setMaximum(16)
        self.workers_spin.setValue(4)
        workers_layout.addWidget(self.workers_spin)

        workers_layout.addWidget(QLabel("Failed Folder:"))
        self.failed_folder_input = QLineEdit("failed_renames")
        workers_layout.addWidget(self.failed_folder_input)
        workers_layout.addStretch()
        settings_layout.addLayout(workers_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Progress
        progress_group = QGroupBox("📊 Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("🚀 Start Processing")
        self.start_btn.setStyleSheet(
            "background-color: #27ae60; color: white; padding: 12px; font-size: 14px; font-weight: bold; border-radius: 5px;"
        )
        self.start_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("⏹️ Cancel")
        self.cancel_btn.setStyleSheet(
            "background-color: #e74c3c; color: white; padding: 12px; font-size: 14px; font-weight: bold; border-radius: 5px;"
        )
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        button_layout.addWidget(self.cancel_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Log output
        log_group = QGroupBox("📋 Execution Log")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.tab_widget.addTab(batch_widget, "🔄 Batch Processing")

    def create_api_tab(self) -> None:
        """Create the API-only processing tab."""
        api_widget = QWidget()
        layout = QVBoxLayout(api_widget)

        # Info label
        info_label = QLabel(
            "🤖 API-Only Mode: Generate rename proposals using AI, then review and approve manually."
        )
        info_label.setStyleSheet(
            "background: #e8f4fd; padding: 10px; border-radius: 5px; color: #2c3e50;"
        )
        layout.addWidget(info_label)

        # Directory and settings
        dir_group = QGroupBox("📁 Directory & Settings")
        dir_layout = QVBoxLayout()

        # Directory selection
        dir_select_layout = QHBoxLayout()
        self.api_dir_input = QLineEdit()
        self.api_dir_input.setPlaceholderText("Select directory for API processing...")
        dir_select_layout.addWidget(self.api_dir_input)
        self.api_browse_btn = QPushButton("Browse...")
        self.api_browse_btn.clicked.connect(self.api_browse_directory)
        dir_select_layout.addWidget(self.api_browse_btn)
        dir_layout.addLayout(dir_select_layout)

        # API settings
        api_settings_layout = QHBoxLayout()
        api_settings_layout.addWidget(QLabel("Style:"))
        self.api_style_combo = QComboBox()
        self.api_style_combo.addItems(["Standard", "Snake Case", "Kebab Case"])
        api_settings_layout.addWidget(self.api_style_combo)

        self.api_include_author = QCheckBox("Include Author")
        api_settings_layout.addWidget(self.api_include_author)

        self.api_recursive = QCheckBox("Include Subfolders")
        self.api_recursive.setChecked(True)
        api_settings_layout.addWidget(self.api_recursive)
        api_settings_layout.addStretch()
        dir_layout.addLayout(api_settings_layout)

        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Control buttons
        api_button_layout = QHBoxLayout()
        self.generate_btn = QPushButton("🔍 Generate Proposals")
        self.generate_btn.setStyleSheet(
            "background-color: #3498db; color: white; padding: 10px; font-size: 14px; font-weight: bold; border-radius: 5px;"
        )
        self.generate_btn.clicked.connect(self.generate_proposals)
        api_button_layout.addWidget(self.generate_btn)

        self.export_btn = QPushButton("📤 Export CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_proposals)
        api_button_layout.addWidget(self.export_btn)

        self.execute_btn = QPushButton("✅ Execute Approved")
        self.execute_btn.setEnabled(False)
        self.execute_btn.clicked.connect(self.execute_approved)
        api_button_layout.addWidget(self.execute_btn)

        api_button_layout.addStretch()
        layout.addLayout(api_button_layout)

        # Proposals table
        table_group = QGroupBox("📋 Rename Proposals")
        table_layout = QVBoxLayout()
        self.proposals_table = QTableWidget()
        self.proposals_table.setColumnCount(6)
        self.proposals_table.setHorizontalHeaderLabels(
            [
                "Current Name",
                "Proposed Name",
                "Confidence",
                "Status",
                "Approve",
                "Reject",
            ]
        )
        if self.proposals_table.horizontalHeader() is not None:
             self.proposals_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
        table_layout.addWidget(self.proposals_table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        self.tab_widget.addTab(api_widget, "🤖 API Mode")

    def create_settings_tab(self) -> None:
        """Create the settings and preferences tab."""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)

        # Preferences group
        prefs_group = QGroupBox("🔧 User Preferences")
        prefs_layout = QVBoxLayout()

        # Remember settings
        self.remember_settings_check = QCheckBox("Remember settings between sessions")
        self.remember_settings_check.setChecked(True)
        prefs_layout.addWidget(self.remember_settings_check)

        # Default failed folder
        failed_layout = QHBoxLayout()
        failed_layout.addWidget(QLabel("Default failed folder name:"))
        self.default_failed_input = QLineEdit("failed_renames")
        failed_layout.addWidget(self.default_failed_input)
        failed_layout.addStretch()
        prefs_layout.addLayout(failed_layout)

        prefs_group.setLayout(prefs_layout)
        layout.addWidget(prefs_group)

        # API Configuration
        api_group = QGroupBox("🔑 API Configuration")
        api_layout = QVBoxLayout()

        api_info = QLabel(
            "Configure your Gemini API key for AI-powered title extraction.\n"
            "The API key is stored securely in your user profile."
        )
        api_layout.addWidget(api_info)

        api_button_layout = QHBoxLayout()
        self.setup_api_btn = QPushButton("🔧 Setup API Key")
        self.setup_api_btn.clicked.connect(self.setup_api_key)
        api_button_layout.addWidget(self.setup_api_btn)

        self.test_api_btn = QPushButton("🧪 Test API Key")
        self.test_api_btn.clicked.connect(self.test_api_key)
        api_button_layout.addWidget(self.test_api_btn)
        api_button_layout.addStretch()
        api_layout.addLayout(api_button_layout)

        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        # Save preferences button
        save_layout = QHBoxLayout()
        self.save_prefs_btn = QPushButton("💾 Save Preferences")
        self.save_prefs_btn.setStyleSheet(
            "background-color: #27ae60; color: white; padding: 10px; font-size: 14px; font-weight: bold; border-radius: 5px;"
        )
        self.save_prefs_btn.clicked.connect(self.save_preferences)
        save_layout.addWidget(self.save_prefs_btn)
        save_layout.addStretch()
        layout.addLayout(save_layout)

        layout.addStretch()
        self.tab_widget.addTab(settings_widget, "⚙️ Settings")

    def load_preferences(self) -> None:
        """Load user preferences into the GUI."""
        # Set last directory
        if self.preferences.get("last_directory"):
            self.dir_input.setText(self.preferences["last_directory"])
            self.api_dir_input.setText(self.preferences["last_directory"])

        # Set other preferences
        self.workers_spin.setValue(self.preferences.get("default_workers", 4))
        self.failed_folder_input.setText(
            self.preferences.get("failed_folder_name", "failed_renames")
        )
        self.default_failed_input.setText(
            self.preferences.get("failed_folder_name", "failed_renames")
        )
        self.remember_settings_check.setChecked(
            self.preferences.get("remember_settings", True)
        )

    def save_preferences(self) -> None:
        """Save current preferences."""
        self.preferences["default_workers"] = self.workers_spin.value()
        self.preferences["failed_folder_name"] = self.default_failed_input.text()
        self.preferences["remember_settings"] = self.remember_settings_check.isChecked()

        save_user_preferences(self.preferences)
        QMessageBox.information(
            self, "Preferences Saved", "Your preferences have been saved successfully!"
        )

    def browse_directory(self) -> None:
        """Open directory browser dialog for batch processing."""
        start_dir = self.preferences.get("last_directory", str(Path.home()))
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory Containing PDFs", start_dir
        )
        if directory:
            self.dir_input.setText(directory)
            update_last_directory(directory)
            self.preferences["last_directory"] = directory

    def api_browse_directory(self) -> None:
        """Open directory browser dialog for API processing."""
        start_dir = self.preferences.get("last_directory", str(Path.home()))
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory for API Processing", start_dir
        )
        if directory:
            self.api_dir_input.setText(directory)
            update_last_directory(directory)
            self.preferences["last_directory"] = directory

    def get_selected_style(self) -> str:
        """Get the selected naming style."""
        if self.style_snake.isChecked():
            return "snake_case"
        elif self.style_kebab.isChecked():
            return "kebab_case"
        return "standard"

    def append_log(self, message: str, level: str = "INFO") -> None:
        """Append message to log with color coding."""
        colors = {
            "INFO": "black",
            "SUCCESS": "green",
            "WARNING": "orange",
            "ERROR": "red",
        }
        color = colors.get(level, "black")
        self.log_output.setTextColor(Qt.GlobalColor.black)
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_output.setTextCursor(cursor)
        self.log_output.insertHtml(
            f'<span style="color:{color};">[{level}] {message}</span><br>'
        )
        self.log_output.ensureCursorVisible()

    def update_progress(self, current: int, total: int, message: str) -> None:
        """Update progress bar and status."""
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
        self.status_label.setText(message)

    def start_processing(self) -> None:
        """Start the PDF processing."""
        directory = self.dir_input.text()
        if not directory or not Path(directory).exists():
            QMessageBox.warning(
                self, "Invalid Directory", "Please select a valid directory."
            )
            return

        # Clear log
        self.log_output.clear()

        # Disable start button, enable cancel
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Get settings
        dry_run = self.dry_run_check.isChecked()
        style = self.get_selected_style()
        delete_dups = self.delete_dups_check.isChecked()
        use_llm = self.use_llm_check.isChecked()
        recursive = self.recursive_check.isChecked()
        workers = self.workers_spin.value()
        include_author = self.include_author_check.isChecked()
        move_failed = self.move_failed_check.isChecked()
        failed_folder = self.failed_folder_input.text() or "failed_renames"

        # Start processing thread
        db_path = Path.cwd() / "pdf_titles.sqlite"
        self.processing_thread = ProcessingThread(
            Path(directory),
            dry_run,
            style,
            delete_dups,
            use_llm,
            recursive,
            workers,
            db_path,
            include_author,
            move_failed,
            failed_folder,
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.log_message.connect(self.append_log)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()

    def cancel_processing(self) -> None:
        """Cancel the processing."""
        if self.processing_thread:
            self.processing_thread.cancel()
            self.append_log("Cancelling processing...", "WARNING")

    def processing_finished(self, success: bool, message: str) -> None:
        """Handle processing completion."""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)

        if success:
            QMessageBox.information(self, "Processing Complete", message)
        else:
            QMessageBox.critical(self, "Processing Failed", message)

    def generate_proposals(self) -> None:
        """Generate API-based rename proposals."""
        directory = self.api_dir_input.text()
        if not directory or not Path(directory).exists():
            QMessageBox.warning(
                self, "Invalid Directory", "Please select a valid directory."
            )
            return

        # Check if API key is available
        from .config import get_api_key

        if not get_api_key():
            QMessageBox.warning(
                self,
                "API Key Required",
                "Please configure your Gemini API key in the Settings tab first.",
            )
            return

        try:
            # Create API manager
            cache = ResultCache(Path.cwd() / "pdf_titles.sqlite")
            llm = GeminiTitleLLM()

            style_map = {
                "Standard": "standard",
                "Snake Case": "snake_case",
                "Kebab Case": "kebab_case",
            }
            style = style_map[self.api_style_combo.currentText()]

            self.api_manager = APIRenameManager(
                directory=Path(directory),
                cache=cache,
                llm=llm,
                style=style,
                include_author=self.api_include_author.isChecked(),
                recursive=self.api_recursive.isChecked(),
            )

            # Generate proposals
            self.generate_btn.setEnabled(False)
            self.generate_btn.setText("🔄 Generating...")

            proposals = self.api_manager.generate_proposals()

            # Populate table
            self.populate_proposals_table(proposals)

            # Enable buttons
            self.export_btn.setEnabled(True)
            self.execute_btn.setEnabled(True)
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("🔍 Generate Proposals")

            QMessageBox.information(
                self,
                "Proposals Generated",
                f"Generated {len(proposals)} rename proposals. Review and approve them below.",
            )

        except Exception as e:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("🔍 Generate Proposals")
            QMessageBox.critical(self, "Error", f"Failed to generate proposals: {e}")
            logger.error(f"Error generating proposals: {e}")

    def populate_proposals_table(self, proposals: list[RenameProposal]) -> None:
        """Populate the proposals table with data."""
        self.proposals_table.setRowCount(len(proposals))

        for i, proposal in enumerate(proposals):
            # Current name
            self.proposals_table.setItem(i, 0, QTableWidgetItem(proposal.current_name))

            # Proposed name
            self.proposals_table.setItem(i, 1, QTableWidgetItem(proposal.proposed_name))

            # Confidence
            confidence_item = QTableWidgetItem(f"{proposal.confidence:.2f}")
            self.proposals_table.setItem(i, 2, confidence_item)

            # Status
            status_item = QTableWidgetItem("Pending")
            self.proposals_table.setItem(i, 3, status_item)

            # Approve button
            approve_btn = QPushButton("✅ Approve")
            approve_btn.clicked.connect(
                lambda checked, idx=i: self.approve_proposal(idx)
            )
            self.proposals_table.setCellWidget(i, 4, approve_btn)

            # Reject button
            reject_btn = QPushButton("❌ Reject")
            reject_btn.clicked.connect(lambda checked, idx=i: self.reject_proposal(idx))
            self.proposals_table.setCellWidget(i, 5, reject_btn)

    def approve_proposal(self, index: int) -> None:
        """Approve a proposal."""
        if self.api_manager and self.api_manager.approve_proposal(index):
            status_item = QTableWidgetItem("✅ Approved")
            status_item.setBackground(Qt.GlobalColor.green)
            self.proposals_table.setItem(index, 3, status_item)

    def reject_proposal(self, index: int) -> None:
        """Reject a proposal."""
        if self.api_manager and self.api_manager.reject_proposal(index):
            status_item = QTableWidgetItem("❌ Rejected")
            status_item.setBackground(Qt.GlobalColor.red)
            self.proposals_table.setItem(index, 3, status_item)

    def export_proposals(self) -> None:
        """Export proposals to CSV."""
        if not self.api_manager:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Proposals", "rename_proposals.csv", "CSV Files (*.csv)"
        )
        if filename:
            try:
                self.api_manager.export_proposals_csv(Path(filename))
                QMessageBox.information(
                    self, "Export Complete", f"Proposals exported to: {filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed", f"Failed to export proposals: {e}"
                )

    def execute_approved(self) -> None:
        """Execute approved rename operations."""
        if not self.api_manager:
            return

        approved = self.api_manager.get_approved_proposals()
        if not approved:
            QMessageBox.warning(
                self, "No Approved Proposals", "Please approve some proposals first."
            )
            return

        reply = QMessageBox.question(
            self,
            "Execute Renames",
            f"Execute {len(approved)} approved rename operations?\n\nThis will actually rename the files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                results = self.api_manager.execute_approved_renames(dry_run=False)
                QMessageBox.information(
                    self,
                    "Execution Complete",
                    f"Rename operations completed:\n"
                    f"✅ Success: {results['success']}\n"
                    f"❌ Failed: {results['failed']}\n"
                    f"⏭️ Skipped: {results['skipped']}",
                )

                # Refresh the table to show completed operations
                for i in range(self.proposals_table.rowCount()):
                    status_item = self.proposals_table.item(i, 3)
                    if status_item and status_item.text() == "✅ Approved":
                        status_item.setText("✅ Completed")

            except Exception as e:
                QMessageBox.critical(
                    self, "Execution Failed", f"Failed to execute renames: {e}"
                )

    def setup_api_key(self) -> None:
        """Setup API key interactively."""
        from .config import setup_api_key_interactive

        try:
            success = setup_api_key_interactive()
            if success:
                QMessageBox.information(
                    self, "API Key Setup", "API key configured successfully!"
                )
            else:
                QMessageBox.information(
                    self, "API Key Setup", "API key setup was cancelled or failed."
                )
        except Exception as e:
            QMessageBox.critical(self, "Setup Error", f"Failed to setup API key: {e}")

    def test_api_key(self) -> None:
        """Test the configured API key."""
        from .config import get_api_key

        api_key = get_api_key()
        if not api_key:
            QMessageBox.warning(
                self, "No API Key", "No API key found. Please setup your API key first."
            )
            return

        try:
            # Test with a simple LLM call
            llm = GeminiTitleLLM()
            if llm.genai:
                QMessageBox.information(
                    self, "API Key Test", "✅ API key is working correctly!"
                )
            else:
                QMessageBox.warning(
                    self,
                    "API Key Test",
                    "❌ API key test failed. Please check your configuration.",
                )
        except Exception as e:
            QMessageBox.critical(self, "API Key Test", f"❌ API key test failed: {e}")


def main() -> None:
    """Main entry point for GUI."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look
    window = PDFRenamerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
