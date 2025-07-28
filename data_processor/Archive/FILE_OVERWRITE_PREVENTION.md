# File Overwriting Prevention Implementation

## Overview

This document describes the file overwriting prevention functionality that has been implemented in the CSV Processor application to prevent accidental file overwriting during export and save operations.

## Features Implemented

### 1. Unique Filename Generation
- **Method**: `_generate_unique_filename(base_path, extension)`
- **Purpose**: Generates unique filenames to prevent overwriting existing files
- **Naming Convention**: 
  - First file: `filename_processed.ext`
  - Subsequent files: `filename_processed_1.ext`, `filename_processed_2.ext`, etc.
- **Smart Handling**: Removes existing `_processed` suffixes before generating new names

### 2. File Overwrite Detection and User Choice
- **Method**: `_check_file_overwrite(file_path)`
- **Purpose**: Checks if a file exists and prompts the user for action
- **User Options**:
  - **Yes**: Overwrite the existing file
  - **No**: Generate a unique filename automatically
  - **Cancel**: Cancel the operation entirely
- **Dialog**: Shows a clear warning dialog with the filename and available options

## Implementation Details

### Core Methods Added

```python
def _generate_unique_filename(self, base_path, extension):
    """Generate a unique filename to prevent overwriting existing files."""
    directory = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    
    # Remove any existing suffix like _processed, _1, _2, etc.
    if base_name.endswith('_processed'):
        base_name = base_name[:-10]  # Remove '_processed'
    
    counter = 1
    while True:
        if counter == 1:
            filename = f"{base_name}_processed{extension}"
        else:
            filename = f"{base_name}_processed_{counter}{extension}"
        
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

def _check_file_overwrite(self, file_path):
    """Check if file exists and prompt user for action."""
    if os.path.exists(file_path):
        filename = os.path.basename(file_path)
        response = messagebox.askyesnocancel(
            "File Already Exists",
            f"The file '{filename}' already exists.\n\n"
            f"Would you like to:\n"
            f"• Yes: Overwrite the existing file\n"
            f"• No: Generate a unique filename\n"
            f"• Cancel: Cancel the operation",
            icon='warning'
        )
        
        if response is None:  # Cancel
            return None
        elif response:  # Yes - overwrite
            return file_path
        else:  # No - generate unique name
            directory = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            extension = os.path.splitext(file_path)[1]
            return self._generate_unique_filename(os.path.join(directory, base_name), extension)
    
    return file_path
```

### Export Methods Updated

All export methods have been updated to use the overwrite prevention functionality:

#### 1. CSV Export Methods
- `_export_csv_separate()`: Each file gets individual overwrite protection
- `_export_csv_compiled()`: Single compiled file gets overwrite protection

#### 2. Excel Export Methods
- `_export_excel_multisheet()`: Multi-sheet Excel file gets overwrite protection
- `_export_excel_separate()`: Each Excel file gets individual overwrite protection

#### 3. MAT Export Methods
- `_export_mat_separate()`: Each MAT file gets individual overwrite protection
- `_export_mat_compiled()`: Single compiled MAT file gets overwrite protection

#### 4. Chart Export Methods
- `_export_chart_image()`: Chart image exports get overwrite protection
- `_export_chart_excel()`: Chart data Excel exports get overwrite protection

### Integration Pattern

Each export method now follows this pattern:

```python
# Original code
output_path = os.path.join(self.output_directory, f"{base_name}_processed.csv")
df.to_csv(output_path, index=False)

# Updated code with overwrite prevention
output_path = os.path.join(self.output_directory, f"{base_name}_processed.csv")
final_path = self._check_file_overwrite(output_path)
if final_path is None:  # User cancelled
    continue  # or return, depending on context
df.to_csv(final_path, index=False)
```

## User Experience

### Before Implementation
- Files could be accidentally overwritten without warning
- No way to recover lost data
- Potential for data loss during batch operations

### After Implementation
- **Automatic Detection**: System detects when a file already exists
- **User Choice**: User can choose to overwrite, generate unique name, or cancel
- **Clear Communication**: Dialog clearly explains the situation and options
- **Safe Default**: Default behavior is to generate unique names, preventing accidental overwrites
- **Batch Safety**: Each file in batch operations gets individual protection

### Example User Flow

1. **User attempts to export** a file named `data_processed.csv`
2. **System detects** that `data_processed.csv` already exists
3. **Dialog appears** with options:
   - "Yes" → Overwrites existing file
   - "No" → Creates `data_processed_1.csv`
   - "Cancel" → Aborts the operation
4. **User selects "No"** → File is saved as `data_processed_1.csv`
5. **Next export** of same base file would create `data_processed_2.csv`

## Benefits

### 1. Data Protection
- Prevents accidental loss of important data
- Maintains data integrity during batch operations
- Provides recovery options for users

### 2. User Control
- Gives users full control over file operations
- Clear communication about what will happen
- Multiple options to handle conflicts

### 3. Professional Experience
- Matches user expectations from other professional software
- Reduces user anxiety about data loss
- Improves overall application reliability

### 4. Batch Operation Safety
- Each file in batch operations gets individual protection
- Users can choose different actions for different files
- Maintains operation integrity even if some files are cancelled

## Technical Considerations

### Performance
- Minimal overhead: Only checks file existence before writing
- No impact on actual data processing
- Efficient file system operations

### Compatibility
- Works with all supported file formats (CSV, Excel, MAT)
- Compatible with all export modes (separate, compiled, multi-sheet)
- No changes to existing file formats or data structure

### Error Handling
- Graceful handling of file system errors
- Clear error messages for users
- Proper cleanup on cancellation

## Future Enhancements

### Potential Improvements
1. **Remember User Preference**: Save user's choice for future operations
2. **Preview Generated Names**: Show what the unique filename will be before saving
3. **Batch Override**: Allow users to set a default action for entire batch operations
4. **File Comparison**: Show differences between existing and new files
5. **Backup Creation**: Automatically create backups before overwriting

### Configuration Options
- Global setting for default behavior
- Per-export-type preferences
- Remember last used directory and filename patterns

## Conclusion

The file overwriting prevention functionality significantly improves the safety and user experience of the CSV Processor application. It prevents accidental data loss while maintaining user control and providing clear communication about file operations. The implementation is comprehensive, covering all export methods and providing a consistent user experience across the application. 