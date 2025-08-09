# Data Processor Tools

A collection of data processing tools and utilities organized into separate projects.

## Project Structure

This workspace has been organized into the following projects:

### 📁 csv_converter/
**CSV to Parquet Converter** - A PyQt6-based GUI application for converting multiple CSV files to Parquet format with advanced features like column selection, bulk processing, and memory optimization.

**Key Features:**
- Bulk conversion of CSV files to Parquet
- Advanced column selection with search/filtering
- Save/load column selection lists
- Memory-efficient processing for large datasets
- Progress tracking and error handling

**Main File:** `csv_to_parquet_converter_enhanced.py`

### 📁 data_processor/
**Data Processor** - A comprehensive data processing application with plotting and analysis capabilities.

**Key Features:**
- Advanced CSV data processing and analysis
- Interactive plotting with zoom, pan, and export
- Signal list management and filtering
- Multiple export formats (CSV, Excel, MATLAB)
- Performance optimized for large datasets

**Main File:** `Data_Processor_r0.py`

### 📁 tests/
All test files for both projects, organized by functionality:
- CSV converter tests
- Data processor tests
- Performance tests
- Workflow tests

### 📁 docs/
Project documentation, including:
- Feature documentation
- Performance analysis
- Implementation guides
- Log files

### 📁 data/
Sample data files and datasets for testing and development.

### 📁 archive/
Historical versions, backups, and development artifacts:
- Previous versions of applications
- Claude fixes and improvements
- Performance analysis reports

### 📁 scripts/
Utility scripts and launchers for the projects.

## Quick Start

### CSV Converter
```bash
cd csv_converter
python csv_to_parquet_converter_enhanced.py
```

### Data Processor
```bash
cd data_processor
python launch_app.py
```

## Dependencies

Each project has its own `requirements.txt` file. Install dependencies for each project separately:

```bash
# For CSV Converter
cd csv_converter
pip install -r requirements.txt

# For Data Processor
cd data_processor
pip install -r requirements.txt
```

## Development

- **Tests**: Run tests from the `tests/` directory
- **Documentation**: Check `docs/` for detailed documentation
- **Archives**: Previous versions and development history in `archive/`

## Organization Benefits

This structure provides:
- **Clear separation** of different tools and their dependencies
- **Independent development** of each project
- **Easy maintenance** with organized test suites
- **Historical preservation** of development progress
- **Clean documentation** for each component 