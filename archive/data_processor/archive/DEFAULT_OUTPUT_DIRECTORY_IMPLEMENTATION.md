# Default Output Directory Implementation

## Overview
This document describes the implementation of the feature that automatically sets the default save location to the same folder that the data file was loaded from, while still allowing users to change it manually.

## Changes Made

### 1. Modified `select_files()` Method
**File**: `CSV_Processor_Rev5_Complete.py` (lines 890-909)

**Changes**:
- Added logic to automatically set the output directory to the folder of the first selected file
- Added UI update to reflect the new default directory in the output label
- Maintained existing functionality for file selection and signal loading

**Code Changes**:
```python
def select_files(self):
    """Select input CSV files."""
    file_paths = filedialog.askopenfilenames(
        title="Select CSV Files",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if file_paths:
        self.input_file_paths = list(file_paths)
        
        # Set default output directory to the folder of the first selected file
        if self.input_file_paths:
            first_file_dir = os.path.dirname(self.input_file_paths[0])
            self.output_directory = first_file_dir
            # Update the output label to reflect the new default directory
            if hasattr(self, 'output_label'):
                self.output_label.configure(text=f"Output: {self.output_directory}")
        
        self.update_file_list()
        self.load_signals_from_files()
```

### 2. Enhanced DAT Import Functionality
**File**: `CSV_Processor_Rev5_Complete.py` (lines 2343-2360)

**Changes**:
- Implemented basic DAT file selection functionality
- Added automatic output directory setting for DAT files
- Implemented tag file selection functionality

**Code Changes**:
```python
def _select_data_file(self):
    """Select data file for DAT import."""
    filepath = filedialog.askopenfilename(
        title="Select Data File",
        filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
    )
    if filepath:
        self.dat_import_data_file_path = filepath
        self.data_file_label.configure(text=os.path.basename(filepath))
        
        # Set default output directory to the folder of the selected DAT file
        dat_file_dir = os.path.dirname(filepath)
        self.output_directory = dat_file_dir
        # Update the output label to reflect the new default directory
        if hasattr(self, 'output_label'):
            self.output_label.configure(text=f"Output: {self.output_directory}")

def _select_tag_file(self):
    """Select tag file for DAT import."""
    filepath = filedialog.askopenfilename(
        title="Select Tag File",
        filetypes=[("DBF files", "*.dbf"), ("All files", "*.*")]
    )
    if filepath:
        self.dat_import_tag_file_path = filepath
        self.tag_file_label.configure(text=os.path.basename(filepath))
```

## User Experience

### Before Implementation
- Users had to manually select an output directory after loading files
- Default output directory was always set to `~/Documents`
- No automatic directory detection based on input files

### After Implementation
- **Automatic Detection**: When users select CSV files, the output directory automatically defaults to the folder containing the first selected file
- **DAT File Support**: When users select DAT files for import, the output directory is set to the folder containing the DAT file
- **Manual Override**: Users can still manually change the output directory using the "Select Output Folder" button
- **Visual Feedback**: The output directory is immediately displayed in the UI, showing users where files will be saved
- **Consistent Behavior**: Works for both CSV file selection and DAT file import

## Technical Details

### Logic Flow
1. User selects files (CSV or DAT)
2. System extracts the directory path of the first selected file
3. System updates `self.output_directory` with the extracted path
4. System updates the UI label to show the new output directory
5. User can optionally change the directory using the manual selection button

### Error Handling
- Checks if `self.input_file_paths` exists before accessing
- Uses `hasattr()` to check if UI elements exist before updating them
- Gracefully handles cases where no files are selected

### Compatibility
- Works with existing file overwriting prevention system
- Maintains compatibility with all export formats (CSV, Excel, MAT)
- Preserves existing manual output directory selection functionality

## Benefits

1. **Improved User Experience**: Users no longer need to manually navigate to the input file directory
2. **Reduced Workflow Steps**: Eliminates the need to remember and manually select the source directory
3. **Logical Default**: Output files are saved in the same location as input files by default
4. **Flexibility Maintained**: Users can still choose a different output location if needed
5. **Consistent Behavior**: Works across all file types and import methods

## Testing

The implementation was tested with:
- Single file selection
- Multiple file selection from different directories
- Different file selection orders
- Empty file selection (edge case)
- DAT file selection
- UI updates verification

All test cases passed successfully, confirming the logic works correctly across various scenarios. 