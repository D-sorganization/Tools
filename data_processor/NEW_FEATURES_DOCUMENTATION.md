# Data Processor - New Plot Integration Features

## Overview
Two new features have been added to improve the workflow between plotting and data processing:

## Feature 1: Copy Plot Range to Time Trimming

### Location
- **Setup & Process Tab** → **Time Trimming Section**
- New button: **"Copy Plot Range to Times"**

### How it Works
1. Go to the **Plotting & Analysis** tab
2. Load and plot your data
3. Use the matplotlib navigation toolbar to zoom into the desired time range
4. Go back to the **Setup & Process** tab
5. Click **"Copy Plot Range to Times"** button
6. The current x-axis (time) range from the plot will be automatically copied to:
   - Date field
   - Start Time field  
   - End Time field

### Benefits
- Visual selection of time ranges using the plot zoom functionality
- No need to manually type in exact timestamps
- Seamless workflow from visual analysis to data processing

## Feature 2: Custom Plot View State Management

### Location
- **Plotting & Analysis Tab** → **Plot Time Range Section**
- New buttons:
  - **"Save Current View"**
  - **"Copy Current View to Processing"**

### How it Works

#### Save Current View
1. Navigate and zoom to your desired plot view
2. Click **"Save Current View"** button
3. The current x-axis and y-axis limits are saved
4. Now when you click the **Home button** on the matplotlib toolbar, it will return to your saved view instead of the default full view

#### Copy Current View to Processing
1. Navigate and zoom to your desired plot view
2. Click **"Copy Current View to Processing"** button
3. The current x-axis time range is copied to the Processing tab time trimming fields
4. The app automatically switches to the **Setup & Process** tab
5. You can now process data using the visually selected time range

### Technical Details

#### Matplotlib Integration
- The features integrate with matplotlib's date handling system
- Automatically converts matplotlib date numbers to readable date/time format
- Works with the existing NavigationToolbar2Tk

#### Error Handling
- Checks if plot data exists before attempting to copy ranges
- Provides informative error messages for common issues
- Fallback behavior for the home button if no saved view exists

## Usage Workflow Example

1. **Load Data**: Import your CSV files in Setup & Process tab
2. **Initial Plot**: Go to Plotting & Analysis tab and create your plot  
3. **Explore Data**: Use zoom, pan tools to explore different time periods
4. **Save Interesting View**: When you find an interesting time period, click "Save Current View"
5. **Copy for Processing**: Click "Copy Current View to Processing" to set up time trimming
6. **Process Data**: The app switches to Setup & Process tab with time range pre-filled
7. **Return to Saved View**: Use Home button to return to your saved interesting view anytime

## Benefits

- **Visual Workflow**: Select time ranges visually rather than typing timestamps
- **Improved Efficiency**: Seamless integration between analysis and processing
- **User-Friendly**: Intuitive button placement and clear functionality
- **Flexible**: Maintains all existing functionality while adding new capabilities

## Compatibility

- Works with existing CSV time series data
- Compatible with all existing filtering and processing options
- Maintains backward compatibility with previous versions
