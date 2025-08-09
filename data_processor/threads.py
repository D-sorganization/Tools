import os
import shutil
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from file_utils import FileFormatDetector, DataReader, DataWriter


class ProcessingThread(QThread):
    """Thread for processing files."""
    progress_updated = pyqtSignal(int, str)

    def __init__(self, files, settings, output_folder, processor=None):
        super().__init__()
        self.files = files
        self.settings = settings
        self.output_folder = output_folder
        if processor is None:
            from Data_Processor_PyQt6 import process_single_csv_file as _processor
            self.processor = _processor
        else:
            self.processor = processor

    def run(self):
        for i, file_path in enumerate(self.files):
            self.progress_updated.emit(
                int((i / len(self.files)) * 100),
                f"Processing {os.path.basename(file_path)}...",
            )
            result = self.processor(file_path, self.settings)
            if result is not None:
                output_path = os.path.join(
                    self.output_folder, f"processed_{os.path.basename(file_path)}"
                )
                result.to_csv(output_path)
        self.progress_updated.emit(100, "Processing completed")


class ConversionThread(QThread):
    """Thread for file conversion."""
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)

    def __init__(self, files, output_format, combine_files, output_dir):
        super().__init__()
        self.files = files
        self.output_format = output_format
        self.combine_files = combine_files
        self.output_dir = output_dir

    def run(self):
        try:
            if self.combine_files:
                self.convert_combined_files()
            else:
                self.convert_separate_files()
        except Exception as e:  # pragma: no cover
            self.log_updated.emit(f"Error: {e}")

    def convert_combined_files(self):
        self.log_updated.emit("Starting combined file conversion...")
        combined_data = []
        for i, file_path in enumerate(self.files):
            self.progress_updated.emit(int((i / len(self.files)) * 50))
            self.log_updated.emit(f"Reading {os.path.basename(file_path)}...")
            try:
                format_type = FileFormatDetector.detect_format(file_path)
                df = DataReader.read_file(file_path, format_type)
                combined_data.append(df)
            except Exception as e:
                self.log_updated.emit(f"Error reading {file_path}: {e}")
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            output_path = os.path.join(
                self.output_dir, f"combined_data.{self.output_format}"
            )
            self.log_updated.emit(f"Saving combined file to {output_path}...")
            DataWriter.write_file(combined_df, output_path, self.output_format)
            self.log_updated.emit("Combined file conversion completed!")
        self.progress_updated.emit(100)

    def convert_separate_files(self):
        self.log_updated.emit("Starting separate file conversion...")
        for i, file_path in enumerate(self.files):
            self.progress_updated.emit(int((i / len(self.files)) * 100))
            self.log_updated.emit(f"Converting {os.path.basename(file_path)}...")
            try:
                format_type = FileFormatDetector.detect_format(file_path)
                df = DataReader.read_file(file_path, format_type)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(
                    self.output_dir, f"{base_name}.{self.output_format}"
                )
                DataWriter.write_file(df, output_path, self.output_format)
                self.log_updated.emit(f"Converted: {os.path.basename(output_path)}")
            except Exception as e:
                self.log_updated.emit(f"Error converting {file_path}: {e}")
        self.log_updated.emit("Separate file conversion completed!")


class FolderProcessingThread(QThread):
    """Thread for folder processing operations."""
    progress_updated = pyqtSignal(int)

    def __init__(self, source_folders, dest_folder, operation):
        super().__init__()
        self.source_folders = source_folders
        self.dest_folder = dest_folder
        self.operation = operation

    def run(self):
        try:
            if self.operation == "Combine":
                self.combine_operation()
            elif self.operation == "Flatten":
                self.flatten_operation()
            elif self.operation == "Prune":
                self.prune_operation()
            elif self.operation == "Deduplicate":
                self.deduplicate_operation()
            elif self.operation == "Analyze":
                self.analyze_operation()
        except Exception as e:  # pragma: no cover
            print(f"Error in folder processing: {e}")

    def combine_operation(self):
        self.progress_updated.emit(0)
        os.makedirs(self.dest_folder, exist_ok=True)
        file_count = 0
        total_files = sum(
            len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            for folder in self.source_folders
        )
        for folder in self.source_folders:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    dest_path = os.path.join(self.dest_folder, file)
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(
                                self.dest_folder, f"{base}_{counter}{ext}"
                            )
                            counter += 1
                    shutil.copy2(file_path, dest_path)
                    file_count += 1
                    self.progress_updated.emit(int((file_count / total_files) * 100))
        self.progress_updated.emit(100)

    def flatten_operation(self):
        self.progress_updated.emit(0)
        os.makedirs(self.dest_folder, exist_ok=True)
        file_count = 0
        total_files = 0
        for folder in self.source_folders:
            for root, _, files in os.walk(folder):
                total_files += len(files)
        for folder in self.source_folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    dest_path = os.path.join(self.dest_folder, file)
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(
                                self.dest_folder, f"{base}_{counter}{ext}"
                            )
                            counter += 1
                    shutil.copy2(file_path, dest_path)
                    file_count += 1
                    self.progress_updated.emit(int((file_count / total_files) * 100))
        self.progress_updated.emit(100)

    def prune_operation(self):
        self.progress_updated.emit(0)
        os.makedirs(self.dest_folder, exist_ok=True)
        file_count = 0
        total_files = sum(
            len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            for folder in self.source_folders
        )
        seen_files = set()
        for folder in self.source_folders:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_key = (file, file_size)
                    if file_key not in seen_files:
                        seen_files.add(file_key)
                        dest_path = os.path.join(self.dest_folder, file)
                        shutil.copy2(file_path, dest_path)
                    file_count += 1
                    self.progress_updated.emit(int((file_count / total_files) * 100))
        self.progress_updated.emit(100)

    def deduplicate_operation(self):
        self.progress_updated.emit(0)
        os.makedirs(self.dest_folder, exist_ok=True)
        file_count = 0
        total_files = sum(
            len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            for folder in self.source_folders
        )
        seen_hashes = set()
        for folder in self.source_folders:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    try:
                        import hashlib
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        if file_hash not in seen_hashes:
                            seen_hashes.add(file_hash)
                            dest_path = os.path.join(self.dest_folder, file)
                            shutil.copy2(file_path, dest_path)
                    except Exception:
                        pass
                    file_count += 1
                    self.progress_updated.emit(int((file_count / total_files) * 100))
        self.progress_updated.emit(100)

    def analyze_operation(self):
        self.progress_updated.emit(0)
        os.makedirs(self.dest_folder, exist_ok=True)
        analysis_results = []
        folder_count = 0
        total_folders = len(self.source_folders)
        for folder in self.source_folders:
            folder_stats = {
                'folder': folder,
                'total_files': 0,
                'total_size': 0,
                'file_types': {},
                'subfolders': 0,
            }
            for root, dirs, files in os.walk(folder):
                folder_stats['subfolders'] += len(dirs)
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        folder_stats['total_files'] += 1
                        folder_stats['total_size'] += file_size
                        ext = os.path.splitext(file)[1].lower()
                        folder_stats['file_types'][ext] = (
                            folder_stats['file_types'].get(ext, 0) + 1
                        )
                    except Exception:
                        pass
            analysis_results.append(folder_stats)
            folder_count += 1
            self.progress_updated.emit(int((folder_count / total_folders) * 100))
        report_path = os.path.join(self.dest_folder, "folder_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("Folder Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            for stats in analysis_results:
                f.write(f"Folder: {stats['folder']}\n")
                f.write(f"Total Files: {stats['total_files']}\n")
                f.write(
                    f"Total Size: {stats['total_size'] / (1024*1024):.2f} MB\n"
                )
                f.write(f"Subfolders: {stats['subfolders']}\n")
                f.write("File Types:\n")
                for ext, count in stats['file_types'].items():
                    f.write(f"  {ext}: {count}\n")
                f.write("\n")
        self.progress_updated.emit(100)
