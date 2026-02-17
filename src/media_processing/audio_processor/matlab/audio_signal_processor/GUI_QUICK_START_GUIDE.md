# GUI Reorganization - Quick Start Guide

## 🎯 Overview

This guide helps you implement the reorganized GUI architecture. Start here for step-by-step instructions.

**Total Effort:** ~7 weeks for full implementation
**Minimum Viable Product:** 3 weeks (Phase 1 only)

---

## 📋 Prerequisites

Before starting, ensure you have:

- ✅ All backend classes in `core/` directory
- ✅ `GUI_ARCHITECTURE_REVIEW.md` reviewed and approved
- ✅ MATLAB R2020b or later with App Designer
- ✅ Backup of current `MainWindow.m`

---

## 🚀 Phase 1: Core Integration (3 Weeks)

### Week 1: Switch to Enhanced Mixer

**Goal:** Replace `MixerCore` with `MixerCoreEnhanced` to enable time offsets.

#### Step 1.1: Update MainWindow.m Initialization

**File:** `gui/MainWindow.m`
**Line:** ~43

**CHANGE THIS:**

```matlab
mainWindow.Mixer = MixerCore(8, 44100);
```

**TO THIS:**

```matlab
mainWindow.Mixer = MixerCoreEnhanced(8, 44100);
```

#### Step 1.2: Add Time Offset Controls to Track Strips

**File:** `gui/MainWindow.m`
**Function:** `createMixerPanel`
**Line:** ~358-402

**ADD AFTER line 397** (after Effects button):

```matlab
% Time offset control (NEW)
offsetPanel = uipanel(trackGrid);
offsetGrid = uigridlayout(offsetPanel, [2, 1]);
offsetGrid.RowHeight = {'fit', 'fit'};
offsetGrid.Padding = [2, 2, 2, 2];

uilabel(offsetGrid, 'Text', 'Offset:', 'FontSize', 8, 'HorizontalAlignment', 'center');
offsetSpinner = uispinner(offsetGrid, 'Value', 0, 'Limits', [0, 300], 'Step', 0.1, ...
    'ValueDisplayFormat', '%.1fs', ...
    'ValueChangedFcn', @(src, event) setTrackOffset(mainWindow.Mixer, i, src.Value));

mainWindow.TrackStrips{i}.OffsetSpinner = offsetSpinner;
```

#### Step 1.3: Add Fade Controls

**ADD AFTER the offset controls:**

```matlab
% Fade in/out buttons (NEW)
fadePanel = uipanel(trackGrid);
fadeGrid = uigridlayout(fadePanel, [1, 2]);
fadeGrid.ColumnWidth = {'1x', '1x'};
fadeGrid.Padding = [2, 2, 2, 2];

uibutton(fadeGrid, 'Text', 'FI', 'Tooltip', 'Fade In', ...
    'ButtonPushedFcn', @(src, event) showFadeInDialog(mainWindow, i));
uibutton(fadeGrid, 'Text', 'FO', 'Tooltip', 'Fade Out', ...
    'ButtonPushedFcn', @(src, event) showFadeOutDialog(mainWindow, i));
```

#### Step 1.4: Add Timeline View

**ADD ABOVE the track strips** (after "Tracks panel" creation ~348):

```matlab
% Timeline view (NEW)
timelinePanel = uipanel(mixerGrid, 'Title', 'Timeline');
timelinePanel.Layout.Row = 1;
mainWindow.MixerTimelineAxes = uiaxes(timelinePanel);
mainWindow.MixerTimelineAxes.XLabel.String = 'Time (s)';
mainWindow.MixerTimelineAxes.YLabel.String = 'Track';
mainWindow.MixerTimelineAxes.Title.String = 'Multi-track Timeline';

% Timeline controls
timelineControlPanel = uipanel(mixerGrid);
timelineControlPanel.Layout.Row = 2;
timelineControlGrid = uigridlayout(timelineControlPanel, [1, 5]);
timelineControlGrid.ColumnWidth = {'fit', 'fit', 'fit', 'fit', '1x'};

uibutton(timelineControlGrid, 'Text', 'Add Marker', ...
    'ButtonPushedFcn', @(src, event) addMarkerDialog(mainWindow));
uidropdown(timelineControlGrid, 'Items', {'Manual', 'Align to Start', 'Align to Peak', 'Align to End'}, ...
    'Value', 'Manual', ...
    'ValueChangedFcn', @(src, event) alignTracks(mainWindow, src.Value));
uibutton(timelineControlGrid, 'Text', 'Zoom +', ...
    'ButtonPushedFcn', @(src, event) zoomTimeline(mainWindow, 0.8));
uibutton(timelineControlGrid, 'Text', 'Zoom -', ...
    'ButtonPushedFcn', @(src, event) zoomTimeline(mainWindow, 1.2));
uilabel(timelineControlGrid, 'Text', '');
```

#### Step 1.5: Update Grid Layout

**CHANGE mixerGrid row configuration** (line ~344):

**FROM:**

```matlab
mixerGrid.RowHeight = {'1x', 'fit'};
```

**TO:**

```matlab
mixerGrid.RowHeight = {'fit', 'fit', '1x', 'fit'};
```

**AND UPDATE track strip row:**

```matlab
tracksPanel.Layout.Row = 3;  % Was Row 1
```

**AND UPDATE master panel row:**

```matlab
masterPanel.Layout.Row = 4;  % Was Row 2
```

#### Step 1.6: Add Helper Functions

**ADD AT END OF FILE:**

```matlab
%% NEW MIXER FUNCTIONS

function showFadeInDialog(mainWindow, trackIndex)
% Show fade in configuration dialog
dialog = uifigure('Name', sprintf('Track %d Fade In', trackIndex), 'Position', [100, 100, 300, 200]);
grid = uigridlayout(dialog, [4, 2]);
grid.RowHeight = {'fit', 'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Duration (s):');
durationField = uispinner(grid, 'Value', 0.5, 'Limits', [0, 10], 'Step', 0.1);

uilabel(grid, 'Text', 'Curve Type:');
curveDropdown = uidropdown(grid, 'Items', {'linear', 'exponential', 'logarithmic', 'scurve'}, 'Value', 'scurve');

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Apply', 'ButtonPushedFcn', @(src, event) applyFadeIn(mainWindow, trackIndex, durationField.Value, curveDropdown.Value, dialog));
end

function applyFadeIn(mainWindow, trackIndex, duration, curveType, dialog)
mainWindow.Mixer.setTrackFadeIn(trackIndex, duration, curveType);
close(dialog);
uialert(mainWindow.Figure, sprintf('Fade in applied to Track %d', trackIndex), 'Success');
end

function showFadeOutDialog(mainWindow, trackIndex)
% Show fade out configuration dialog
dialog = uifigure('Name', sprintf('Track %d Fade Out', trackIndex), 'Position', [100, 100, 300, 200]);
grid = uigridlayout(dialog, [4, 2]);
grid.RowHeight = {'fit', 'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Duration (s):');
durationField = uispinner(grid, 'Value', 1.0, 'Limits', [0, 10], 'Step', 0.1);

uilabel(grid, 'Text', 'Curve Type:');
curveDropdown = uidropdown(grid, 'Items', {'linear', 'exponential', 'logarithmic', 'scurve'}, 'Value', 'exponential');

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Apply', 'ButtonPushedFcn', @(src, event) applyFadeOut(mainWindow, trackIndex, durationField.Value, curveDropdown.Value, dialog));
end

function applyFadeOut(mainWindow, trackIndex, duration, curveType, dialog)
mainWindow.Mixer.setTrackFadeOut(trackIndex, duration, curveType);
close(dialog);
uialert(mainWindow.Figure, sprintf('Fade out applied to Track %d', trackIndex), 'Success');
end

function addMarkerDialog(mainWindow)
% Add marker to timeline
dialog = uifigure('Name', 'Add Marker', 'Position', [100, 100, 300, 150]);
grid = uigridlayout(dialog, [3, 2]);
grid.RowHeight = {'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Time (s):');
timeField = uispinner(grid, 'Value', 0, 'Limits', [0, 1000], 'Step', 0.1);

uilabel(grid, 'Text', 'Label:');
labelField = uieditfield(grid, 'Value', 'Marker');

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Add', 'ButtonPushedFcn', @(src, event) addMarker(mainWindow, timeField.Value, labelField.Value, dialog));
end

function addMarker(mainWindow, time, label, dialog)
mainWindow.Mixer.addMarker(time, label);
updateTimelineDisplay(mainWindow);
close(dialog);
end

function alignTracks(mainWindow, method)
% Align tracks automatically
if strcmp(method, 'Manual')
    return;
end

% Map UI names to mixer method names
methodMap = containers.Map(...
    {'Align to Start', 'Align to Peak', 'Align to End'}, ...
    {'start', 'peak', 'end'});

if isKey(methodMap, method)
    mainWindow.Mixer.alignTracks(methodMap(method));
    updateTimelineDisplay(mainWindow);
    uialert(mainWindow.Figure, sprintf('Tracks aligned using: %s', method), 'Success');
end
end

function zoomTimeline(mainWindow, factor)
% Zoom timeline view
if isfield(mainWindow, 'MixerTimelineAxes')
    currentXLim = xlim(mainWindow.MixerTimelineAxes);
    center = mean(currentXLim);
    range = diff(currentXLim) * factor / 2;
    xlim(mainWindow.MixerTimelineAxes, [center - range, center + range]);
end
end

function updateTimelineDisplay(mainWindow)
% Update timeline visualization
if ~isfield(mainWindow, 'MixerTimelineAxes')
    return;
end

cla(mainWindow.MixerTimelineAxes);
hold(mainWindow.MixerTimelineAxes, 'on');

% Plot each track
for i = 1:mainWindow.Mixer.NumTracks
    track = mainWindow.Mixer.Tracks(i);
    if track.IsLoaded
        offset = track.Offset;
        duration = length(track.AudioData) / mainWindow.Mixer.SampleRate;

        % Draw track bar
        rectangle(mainWindow.MixerTimelineAxes, ...
            'Position', [offset, i-0.4, duration, 0.8], ...
            'FaceColor', [0.3, 0.5, 0.8], ...
            'EdgeColor', 'k');

        % Add track label
        text(mainWindow.MixerTimelineAxes, offset + 0.1, i, sprintf('Track %d', i), ...
            'Color', 'white', 'FontWeight', 'bold', 'FontSize', 8);
    end
end

% Plot markers
if isfield(mainWindow.Mixer, 'Markers') && ~isempty(mainWindow.Mixer.Markers)
    for i = 1:length(mainWindow.Mixer.Markers)
        marker = mainWindow.Mixer.Markers(i);
        xline(mainWindow.MixerTimelineAxes, marker.Time, '--r', marker.Label);
    end
end

hold(mainWindow.MixerTimelineAxes, 'off');
ylim(mainWindow.MixerTimelineAxes, [0, mainWindow.Mixer.NumTracks + 1]);
yticks(mainWindow.MixerTimelineAxes, 1:mainWindow.Mixer.NumTracks);
yticklabels(mainWindow.MixerTimelineAxes, arrayfun(@(x) sprintf('Track %d', x), 1:mainWindow.Mixer.NumTracks, 'UniformOutput', false));
grid(mainWindow.MixerTimelineAxes, 'on');
end
```

#### Step 1.7: Update processMix Function

**FIND the processMix function** (line ~965) and **ADD timeline update:**

```matlab
function processMix(mainWindow)
% Process and mix all tracks

% Update timeline first
updateTimelineDisplay(mainWindow);

try
    mixedAudio = mainWindow.Mixer.processMix();
    mainWindow.LoadedAudio = mixedAudio;
    mainWindow.CurrentFile = 'Mixed Audio';
    updateWaveformDisplay(mainWindow);
    uialert(mainWindow.Figure, 'Mix processed successfully', 'Success');
catch ME
    uialert(mainWindow.Figure, ['Error processing mix: ' ME.message], 'Error');
end
end
```

**✅ Week 1 Complete!** You now have enhanced mixer with time offsets, fades, and timeline view.

---

### Week 2: Add Edit Tab

**Goal:** Create audio editing capabilities.

#### Step 2.1: Add Edit Tab to Tab Group

**File:** `gui/MainWindow.m`
**Function:** `createTabGroup`
**Line:** ~166-192

**ADD AFTER Waveform tab** (after line ~175):

```matlab
% Edit tab (NEW)
editTab = uitab(mainWindow.TabGroup, 'Title', 'Edit');
createEditPanel(mainWindow, editTab);
```

#### Step 2.2: Create Edit Panel Function

**ADD NEW FUNCTION** (after `createWaveformPanel`):

```matlab
function createEditPanel(mainWindow, parent)
% Create audio editing panel

editGrid = uigridlayout(parent, [4, 1]);
editGrid.RowHeight = {'fit', 'fit', 'fit', 'fit'};
editGrid.Padding = [10, 10, 10, 10];
editGrid.RowSpacing = 10;

% Selection Tools
selectionPanel = uipanel(editGrid, 'Title', 'Selection & Editing');
selectionPanel.Layout.Row = 1;
selectionGrid = uigridlayout(selectionPanel, [3, 4]);
selectionGrid.ColumnWidth = {'fit', '1x', '1x', '1x'};
selectionGrid.Padding = [5, 5, 5, 5];

uilabel(selectionGrid, 'Text', 'Selection:');
mainWindow.SelectionStartField = uispinner(selectionGrid, 'Value', 0, 'Limits', [0, 1000], ...
    'ValueDisplayFormat', '%.3fs', 'Tooltip', 'Selection start time');
mainWindow.SelectionEndField = uispinner(selectionGrid, 'Value', 0, 'Limits', [0, 1000], ...
    'ValueDisplayFormat', '%.3fs', 'Tooltip', 'Selection end time');
mainWindow.SelectionDurationLabel = uilabel(selectionGrid, 'Text', 'Duration: 0.000s');

uilabel(selectionGrid, 'Text', 'Actions:');
uibutton(selectionGrid, 'Text', 'Trim', 'ButtonPushedFcn', @(src, event) trimAudio(mainWindow), ...
    'Tooltip', 'Keep selection, delete rest');
uibutton(selectionGrid, 'Text', 'Cut', 'ButtonPushedFcn', @(src, event) cutAudio(mainWindow), ...
    'Tooltip', 'Cut selection to clipboard');
uibutton(selectionGrid, 'Text', 'Copy', 'ButtonPushedFcn', @(src, event) copyAudio(mainWindow), ...
    'Tooltip', 'Copy selection to clipboard');

uilabel(selectionGrid, 'Text', '');
uibutton(selectionGrid, 'Text', 'Paste at:', 'ButtonPushedFcn', @(src, event) pasteAudio(mainWindow));
mainWindow.PastePositionField = uispinner(selectionGrid, 'Value', 0, 'Limits', [0, 1000], ...
    'ValueDisplayFormat', '%.3fs');
uibutton(selectionGrid, 'Text', 'Select All', 'ButtonPushedFcn', @(src, event) selectAllAudio(mainWindow));

% Fades & Crossfades
fadePanel = uipanel(editGrid, 'Title', 'Fades & Crossfades');
fadePanel.Layout.Row = 2;
fadeGrid = uigridlayout(fadePanel, [2, 5]);
fadeGrid.ColumnWidth = {'fit', '1x', '1x', 'fit', 'fit'};
fadeGrid.Padding = [5, 5, 5, 5];

uilabel(fadeGrid, 'Text', 'Fade In:');
mainWindow.FadeInDurationField = uispinner(fadeGrid, 'Value', 0.5, 'Limits', [0, 10], 'Step', 0.1, ...
    'ValueDisplayFormat', '%.2fs');
mainWindow.FadeInCurveDropdown = uidropdown(fadeGrid, ...
    'Items', {'linear', 'exponential', 'logarithmic', 'scurve'}, 'Value', 'scurve');
uibutton(fadeGrid, 'Text', 'Preview', 'ButtonPushedFcn', @(src, event) previewFadeIn(mainWindow));
uibutton(fadeGrid, 'Text', 'Apply', 'ButtonPushedFcn', @(src, event) applyFadeInToSelection(mainWindow));

uilabel(fadeGrid, 'Text', 'Fade Out:');
mainWindow.FadeOutDurationField = uispinner(fadeGrid, 'Value', 1.0, 'Limits', [0, 10], 'Step', 0.1, ...
    'ValueDisplayFormat', '%.2fs');
mainWindow.FadeOutCurveDropdown = uidropdown(fadeGrid, ...
    'Items', {'linear', 'exponential', 'logarithmic', 'scurve'}, 'Value', 'exponential');
uibutton(fadeGrid, 'Text', 'Preview', 'ButtonPushedFcn', @(src, event) previewFadeOut(mainWindow));
uibutton(fadeGrid, 'Text', 'Apply', 'ButtonPushedFcn', @(src, event) applyFadeOutToSelection(mainWindow));

% Processing
processingPanel = uipanel(editGrid, 'Title', 'Audio Processing');
processingPanel.Layout.Row = 3;
processingGrid = uigridlayout(processingPanel, [2, 5]);
processingGrid.ColumnWidth = {'1x', '1x', '1x', '1x', '1x'};
processingGrid.Padding = [5, 5, 5, 5];

uibutton(processingGrid, 'Text', 'Normalize', 'ButtonPushedFcn', @(src, event) showNormalizeDialog(mainWindow));
uibutton(processingGrid, 'Text', 'Remove Silence', 'ButtonPushedFcn', @(src, event) showRemoveSilenceDialog(mainWindow));
uibutton(processingGrid, 'Text', 'Reverse', 'ButtonPushedFcn', @(src, event) reverseAudio(mainWindow));
uibutton(processingGrid, 'Text', 'Remove DC Offset', 'ButtonPushedFcn', @(src, event) removeDCOffset(mainWindow));
uibutton(processingGrid, 'Text', 'Change Gain', 'ButtonPushedFcn', @(src, event) showGainDialog(mainWindow));

% History
historyPanel = uipanel(editGrid, 'Title', 'Undo/Redo History');
historyPanel.Layout.Row = 4;
historyGrid = uigridlayout(historyPanel, [1, 4]);
historyGrid.ColumnWidth = {'fit', 'fit', '1x', 'fit'};
historyGrid.Padding = [5, 5, 5, 5];

mainWindow.UndoButton = uibutton(historyGrid, 'Text', '◀ Undo', ...
    'ButtonPushedFcn', @(src, event) undoEdit(mainWindow), ...
    'Enable', 'off');
mainWindow.RedoButton = uibutton(historyGrid, 'Text', 'Redo ▶', ...
    'ButtonPushedFcn', @(src, event) redoEdit(mainWindow), ...
    'Enable', 'off');
mainWindow.HistoryLabel = uilabel(historyGrid, 'Text', 'No history');
uibutton(historyGrid, 'Text', 'Clear History', 'ButtonPushedFcn', @(src, event) clearEditHistory(mainWindow));

% Initialize audio editor
mainWindow.AudioEditor = [];
end
```

#### Step 2.3: Add Edit Functions

**ADD AT END OF FILE:**

```matlab
%% AUDIO EDITING FUNCTIONS

function initializeEditor(mainWindow)
% Initialize AudioEditor with current audio
if isempty(mainWindow.LoadedAudio)
    return;
end

mainWindow.AudioEditor = AudioEditor(mainWindow.LoadedAudio, mainWindow.SampleRate);
updateEditHistory(mainWindow);
end

function trimAudio(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

startTime = mainWindow.SelectionStartField.Value;
endTime = mainWindow.SelectionEndField.Value;

if startTime >= endTime
    uialert(mainWindow.Figure, 'Invalid selection: start must be before end', 'Error');
    return;
end

mainWindow.AudioEditor.setSelection(startTime, endTime);
mainWindow.AudioEditor.trim();

mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
uialert(mainWindow.Figure, 'Audio trimmed successfully', 'Success');
end

function cutAudio(mainWindow)
if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

startTime = mainWindow.SelectionStartField.Value;
endTime = mainWindow.SelectionEndField.Value;

mainWindow.AudioEditor.setSelection(startTime, endTime);
mainWindow.AudioEditor.cut();

mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
uialert(mainWindow.Figure, 'Selection cut to clipboard', 'Success');
end

function copyAudio(mainWindow)
if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

startTime = mainWindow.SelectionStartField.Value;
endTime = mainWindow.SelectionEndField.Value;

mainWindow.AudioEditor.setSelection(startTime, endTime);
mainWindow.AudioEditor.copy();
uialert(mainWindow.Figure, 'Selection copied to clipboard', 'Success');
end

function pasteAudio(mainWindow)
if isempty(mainWindow.AudioEditor)
    uialert(mainWindow.Figure, 'No clipboard data', 'Error');
    return;
end

position = mainWindow.PastePositionField.Value;
mainWindow.AudioEditor.paste(position);

mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
uialert(mainWindow.Figure, 'Audio pasted successfully', 'Success');
end

function selectAllAudio(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

duration = length(mainWindow.LoadedAudio) / mainWindow.SampleRate;
mainWindow.SelectionStartField.Value = 0;
mainWindow.SelectionEndField.Value = duration;
mainWindow.SelectionDurationLabel.Text = sprintf('Duration: %.3fs', duration);
end

function applyFadeInToSelection(mainWindow)
if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

duration = mainWindow.FadeInDurationField.Value;
curveType = mainWindow.FadeInCurveDropdown.Value;

mainWindow.AudioEditor.fadeIn(duration, curveType);
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
uialert(mainWindow.Figure, 'Fade in applied', 'Success');
end

function applyFadeOutToSelection(mainWindow)
if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

duration = mainWindow.FadeOutDurationField.Value;
curveType = mainWindow.FadeOutCurveDropdown.Value;

mainWindow.AudioEditor.fadeOut(duration, curveType);
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
uialert(mainWindow.Figure, 'Fade out applied', 'Success');
end

function showNormalizeDialog(mainWindow)
dialog = uifigure('Name', 'Normalize Audio', 'Position', [100, 100, 300, 200]);
grid = uigridlayout(dialog, [4, 2]);
grid.RowHeight = {'fit', 'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Method:');
methodGroup = uibuttongroup(grid);
uiradiobutton(methodGroup, 'Text', 'Peak', 'Position', [5, 30, 100, 22], 'Value', true);
uiradiobutton(methodGroup, 'Text', 'RMS', 'Position', [5, 5, 100, 22]);

uilabel(grid, 'Text', 'Target (dB):');
targetField = uispinner(grid, 'Value', -3, 'Limits', [-60, 0]);

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Apply', 'ButtonPushedFcn', @(src, event) applyNormalize(mainWindow, methodGroup.SelectedObject.Text, targetField.Value, dialog));
end

function applyNormalize(mainWindow, method, target, dialog)
if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

mainWindow.AudioEditor.normalize(lower(method), target);
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
close(dialog);
uialert(mainWindow.Figure, sprintf('Audio normalized to %ddB (%s)', target, method), 'Success');
end

function reverseAudio(mainWindow)
if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

mainWindow.AudioEditor.reverse();
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
uialert(mainWindow.Figure, 'Audio reversed', 'Success');
end

function removeDCOffset(mainWindow)
if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

mainWindow.AudioEditor.removeOffset();
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
uialert(mainWindow.Figure, 'DC offset removed', 'Success');
end

function undoEdit(mainWindow)
if isempty(mainWindow.AudioEditor)
    return;
end

mainWindow.AudioEditor.undo();
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
end

function redoEdit(mainWindow)
if isempty(mainWindow.AudioEditor)
    return;
end

mainWindow.AudioEditor.redo();
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
end

function updateEditHistory(mainWindow)
if isempty(mainWindow.AudioEditor)
    return;
end

% Update undo/redo button states
if mainWindow.AudioEditor.HistoryIndex > 1
    mainWindow.UndoButton.Enable = 'on';
else
    mainWindow.UndoButton.Enable = 'off';
end

if mainWindow.AudioEditor.HistoryIndex < length(mainWindow.AudioEditor.History)
    mainWindow.RedoButton.Enable = 'on';
else
    mainWindow.RedoButton.Enable = 'off';
end

% Update history label
historyText = sprintf('History: %d/%d', mainWindow.AudioEditor.HistoryIndex, length(mainWindow.AudioEditor.History));
mainWindow.HistoryLabel.Text = historyText;
end

function clearEditHistory(mainWindow)
if ~isempty(mainWindow.AudioEditor)
    mainWindow.AudioEditor = [];
end
updateEditHistory(mainWindow);
uialert(mainWindow.Figure, 'Edit history cleared', 'Info');
end

function previewFadeIn(mainWindow)
uialert(mainWindow.Figure, 'Preview functionality coming soon', 'Info');
end

function previewFadeOut(mainWindow)
uialert(mainWindow.Figure, 'Preview functionality coming soon', 'Info');
end

function showRemoveSilenceDialog(mainWindow)
uialert(mainWindow.Figure, 'Remove silence dialog coming soon', 'Info');
end

function showGainDialog(mainWindow)
uialert(mainWindow.Figure, 'Gain dialog coming soon', 'Info');
end
```

**✅ Week 2 Complete!** You now have audio editing with trim, cut, copy, paste, fades, and undo/redo.

---

### Week 3: Add Effects Tab

Continue with the remaining implementation...

---

## 📊 Testing Checklist

After each week, test:

### Week 1 Mixer Tests

- [ ] Load audio into multiple tracks
- [ ] Set different offsets for tracks
- [ ] Apply fade in/out to tracks
- [ ] Add markers to timeline
- [ ] Auto-align tracks
- [ ] Process mix and verify offsets work
- [ ] Zoom timeline in/out

### Week 2 Edit Tests

- [ ] Select audio region
- [ ] Trim audio
- [ ] Cut/copy/paste audio
- [ ] Apply fade in/out
- [ ] Normalize audio (peak, RMS)
- [ ] Reverse audio
- [ ] Undo/redo operations (test 10+ times)
- [ ] Remove DC offset

### Week 3 Effects Tests

- [ ] Add effect to chain
- [ ] Reorder effects
- [ ] Bypass individual effects
- [ ] Apply effect chain
- [ ] Save/load effect preset
- [ ] Use convolution reverb with built-in IR
- [ ] Load custom IR file

---

## 🐛 Common Issues & Fixes

### Issue: "Undefined function or variable 'MixerCoreEnhanced'"

**Fix:** Ensure `MixerCoreEnhanced.m` is in the `core/` folder and on MATLAB path.

### Issue: "Index exceeds array dimensions" in timeline

**Fix:** Check that tracks are loaded before calling `updateTimelineDisplay()`.

### Issue: Undo/redo buttons not enabling

**Fix:** Ensure `AudioEditor` is initialized before first edit operation.

### Issue: Effects not applying

**Fix:** Verify `AudioEffects.m` and `ConvolutionReverb.m` are on path.

---

## 📚 Next Steps

After completing Phase 1:

1. Test thoroughly with real audio files
2. Gather user feedback
3. Proceed to Phase 2 (Production and Research tabs)
4. Document any custom workflows

---

## 💡 Pro Tips

1. **Backup frequently** - Save copies of `MainWindow.m` at each stage
2. **Test incrementally** - Don't add all features at once
3. **Use version control** - Git is your friend
4. **Read the backend docs** - Each class has detailed examples
5. **Start simple** - Get basic functionality working before adding polish

---

## 🆘 Getting Help

- Review `GUI_ARCHITECTURE_REVIEW.md` for big picture
- Check individual class documentation for backend details
- See `ENHANCEMENT_EXAMPLES.m` for working code examples
- Consult MATLAB documentation for UI components

---

**You're ready to start! Begin with Week 1 and work through systematically. Good luck! 🚀**
