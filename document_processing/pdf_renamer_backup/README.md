# PDF Renamer

A professional tool to bulk rename PDF files based on their metadata (Author - Title), with smart duplicate handling.

## Features

- **Metadata Extraction**: Extracts Author and Title from PDF metadata.
- **Smart Renaming**: Renames files to `Author - Title.pdf` using title case (ignoring minor words).
- **Duplicate Management**: Identifies duplicates by content (size + hash) and offers deletion.
- **Safety**: Dry-run mode to preview changes.

## Usage

```bash
python3 src/pdf_renamer/main.py /path/to/pdfs --dry-run
```
