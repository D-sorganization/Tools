# Chapter 7 — Data and Document Processing

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)

---

## 7.1 Data Processor

**Source:** `src/data_processing/data_processor/`
**GUI:** PyQt6 + Web
**Status:** ✅ Implemented

### 7.1.1 Purpose

General-purpose data processing tool for loading, transforming, filtering, and analyzing tabular datasets.

### 7.1.2 Capabilities

- CSV/Excel data import
- Data cleaning and preprocessing
- Statistical analysis
- Signal processing integration (via Signal Toolkit)
- Data visualization with matplotlib
- Export to multiple formats
- PyQt6 desktop GUI
- Web interface (Streamlit/Flask)

### 7.1.3 Statistical Operations

**Descriptive Statistics:**

$$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$$

$$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}$$

**Correlation:**

$$r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \cdot \sum(y_i - \bar{y})^2}}$$

---

## 7.2 PDF Renamer

**Source:** `src/document_processing/pdf_renamer/`
**Status:** ✅ Implemented

### 7.2.1 Purpose

Automated PDF file renaming tool that extracts metadata and content from PDF documents to generate standardized filenames.

### 7.2.2 Capabilities

- PDF metadata extraction (title, author, date)
- Text content extraction
- Pattern-based filename generation
- Batch processing
- Configurable naming templates
- Dry-run mode for preview

### 7.2.3 Components

| Module         | Description                          |
| -------------- | ------------------------------------ |
| `extractor.py` | PDF text/metadata extraction         |
| `utils.py`     | Filename sanitization and formatting |
| `tests/`       | Test suite with conftest.py          |

---

_[← Robotics & 3D](./06_robotics_3d.md) | [Back to Manual](./TOOLS_USER_MANUAL.md) | [Next: Web Applications →](./08_web_applications.md)_
