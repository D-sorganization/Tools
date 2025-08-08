# Signal List Management Feature

## Overview

The Signal List Management feature allows users to save, load, and apply predefined lists of signals across different data files. This eliminates the need to manually select the same signals every time similar files are loaded, significantly improving workflow efficiency.

## Features

### 1. Save Current Signal List
- **Purpose**: Save the currently selected signals as a reusable signal list
- **Location**: Setup & Process Tab → Setup Sub-tab → Signal List Management section
- **Button**: "Save Current Signal List"
- **Process**:
  1. User selects signals from the current file
  2. Clicks "Save Current Signal List"
  3. Enters a name for the signal list
  4. Chooses a save location (JSON format)
  5. Signal list is saved with metadata (name, signals, creation date)

### 2. Load Saved Signal List
- **Purpose**: Load a previously saved signal list from file
- **Button**: "Load Saved Signal List"
- **Process**:
  1. User clicks "Load Saved Signal List"
  2. Selects a JSON file containing a saved signal list
  3. System validates the file format and loads the signals
  4. Status is updated to show loaded signal list name and count

### 3. Apply Saved Signals
- **Purpose**: Apply the loaded signal list to the current file's available signals
- **Button**: "Apply Saved Signals"
- **Process**:
  1. User loads a file with signals
  2. Clicks "Apply Saved Signals"
  3. System matches saved signals with available signals
  4. Automatically selects matching signals and deselects others
  5. Shows detailed feedback about which signals were applied and which are missing

## User Interface

### Signal List Management Frame
Located in the Setup & Process Tab → Setup Sub-tab, the Signal List Management frame contains:

- **Title**: "Signal List Management"
- **Three Buttons**:
  - Save Current Signal List
  - Load Saved Signal List  
  - Apply Saved Signals
- **Status Label**: Shows current state (e.g., "No saved signal list loaded", "Loaded: My Signals (5 signals)")

### Status Feedback
The status label provides real-time feedback:
- **Gray**: No signal list loaded
- **Green**: Signal list successfully loaded
- **Blue**: Signal list applied to current file

## File Format

Signal lists are saved as JSON files with the following structure:

```json
{
  "name": "My Signal List",
  "signals": ["Time", "Signal1", "Signal2", "Signal3"],
  "created_date": "2024-01-15T10:30:00"
}
```

### Fields:
- **name**: User-defined name for the signal list
- **signals**: Array of signal names to be selected
- **created_date**: ISO timestamp of when the list was created

## Error Handling

### Save Signal List
- **No signals available**: Warns user to load a file first
- **No signals selected**: Warns user to select signals before saving
- **File save error**: Shows error message with details

### Load Signal List
- **Invalid file format**: Shows error for malformed JSON
- **Missing required fields**: Validates file structure
- **File read error**: Shows error message with details

### Apply Saved Signals
- **No signal list loaded**: Warns user to load a signal list first
- **No signals available**: Warns user to load a file first
- **Missing signals**: Shows detailed list of signals not found in current file

## Workflow Example

### Typical Usage Scenario:

1. **Load a file** with signals (e.g., "sensor_data_001.csv")
2. **Select desired signals** (e.g., Time, Temperature, Pressure, Flow_Rate)
3. **Save signal list** as "Temperature Sensors"
4. **Load a new file** (e.g., "sensor_data_002.csv")
5. **Apply saved signals** - system automatically selects Temperature, Pressure, Flow_Rate if available
6. **Review feedback** - see which signals were applied and which are missing

### Benefits:
- **Time Savings**: No need to manually select signals for each file
- **Consistency**: Ensures same signals are selected across similar files
- **Error Prevention**: Reduces chance of missing important signals
- **Workflow Efficiency**: Streamlines repetitive tasks

## Technical Implementation

### Key Methods:
- `save_signal_list()`: Saves currently selected signals to JSON file
- `load_signal_list()`: Loads signal list from JSON file with validation
- `apply_saved_signals()`: Applies loaded signals to current file with error handling

### Data Structures:
- `self.saved_signal_list`: List of signal names from loaded file
- `self.saved_signal_list_name`: Name of the loaded signal list
- `self.signal_list_status_label`: UI element for status feedback

### Integration:
- Seamlessly integrates with existing signal selection system
- Uses same `signal_vars` structure for consistency
- Updates status bar and UI elements for user feedback

## Testing

The functionality has been thoroughly tested with unit tests covering:
- Signal list creation and saving
- File format validation
- Signal matching logic
- Error handling scenarios
- Signal selection application

All tests pass successfully, ensuring reliable operation.

## Future Enhancements

Potential improvements could include:
- Multiple signal list management
- Signal list categories/tags
- Import/export of signal lists between users
- Automatic signal list suggestions based on file patterns
- Integration with processing templates 