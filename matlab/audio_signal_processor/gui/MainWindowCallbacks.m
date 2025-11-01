% MAINWINDOWCALLBACKS - All callback functions for MainWindow
% This file contains all the callback and helper functions for the
% main audio processor GUI. Include this file after MainWindow.m or
% append these functions to MainWindow.m.
%
% This is Part 2 of the complete GUI implementation.

%% BASIC WINDOW FUNCTIONS
function show(mainWindow)
mainWindow.Figure.Visible = 'on';
end

function hide(mainWindow)
mainWindow.Figure.Visible = 'off';
end

function close(mainWindow)
delete(mainWindow.Figure);
end

%% AUDIO LOADING AND FILE MANAGEMENT
function loadAudio(mainWindow, filename)
try
    [audioData, sampleRate, ~] = AudioLoader(filename);

    mainWindow.LoadedAudio = audioData;
    mainWindow.SampleRate = sampleRate;
    mainWindow.CurrentFile = filename;

    % Update all relevant displays
    updateWaveformDisplay(mainWindow);
    updateFileInfo(mainWindow);
    updateAAInfo(mainWindow);

    % Reset editor
    mainWindow.AudioEditor = [];

    mainWindow.StatusText.Text = sprintf('Loaded: %s', filename);

catch ME
    uialert(mainWindow.Figure, sprintf('Error loading audio: %s', ME.message), 'Load Error');
end
end

function loadAudioDialog(mainWindow)
[filename, pathname] = uigetfile({'*.wav;*.mp3;*.flac;*.ogg;*.m4a', 'Audio Files'}, 'Load Audio');
if filename ~= 0
    loadAudio(mainWindow, fullfile(pathname, filename));
end
end

function exportAudioDialog(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio to export', 'Warning');
    return;
end

[file, path] = uiputfile({'*.wav', 'WAV File'; '*.mp3', 'MP3 File'}, 'Export Audio');
if file == 0
    return;
end

try
    AudioExporter(mainWindow.LoadedAudio, fullfile(path, file), ...
        'SampleRate', mainWindow.SampleRate, 'BitDepth', 24);
    uialert(mainWindow.Figure, 'Audio exported successfully', 'Success');
catch ME
    uialert(mainWindow.Figure, ['Error exporting: ' ME.message], 'Error');
end
end

function exportWithEffects(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Warning');
    return;
end

if isempty(mainWindow.EffectChain)
    uialert(mainWindow.Figure, 'No effects in chain. Use Export Audio instead.', 'Info');
    return;
end

% Apply effect chain
processed = applyEffectChainToAudio(mainWindow, mainWindow.LoadedAudio);

% Export
[file, path] = uiputfile({'*.wav', 'WAV File'}, 'Export with Effects');
if file ~= 0
    AudioExporter(processed, fullfile(path, file), ...
        'SampleRate', mainWindow.SampleRate, 'BitDepth', 24);
    uialert(mainWindow.Figure, 'Audio with effects exported successfully', 'Success');
end
end

%% WAVEFORM DISPLAY
function updateWaveformDisplay(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

audioData = mainWindow.LoadedAudio;
sampleRate = mainWindow.SampleRate;

% Create time vector
time = (0:size(audioData, 1)-1) / sampleRate;

% Plot waveform
cla(mainWindow.WaveformAxes);
plot(mainWindow.WaveformAxes, time, audioData);
mainWindow.WaveformAxes.XLabel.String = 'Time (s)';
mainWindow.WaveformAxes.YLabel.String = 'Amplitude';
mainWindow.WaveformAxes.Title.String = sprintf('Audio Waveform - %s', mainWindow.CurrentFile);
grid(mainWindow.WaveformAxes, 'on');

% Update time display
duration = length(audioData) / sampleRate;
mainWindow.TimeDisplay.Text = sprintf('00:00 / %02d:%02d', floor(duration/60), mod(floor(duration), 60));
end

function updateFileInfo(mainWindow)
if isempty(mainWindow.LoadedAudio)
    mainWindow.FileInfoLabel.Text = 'No audio loaded';
    return;
end

[~, name, ext] = fileparts(mainWindow.CurrentFile);
duration = length(mainWindow.LoadedAudio) / mainWindow.SampleRate;
channels = size(mainWindow.LoadedAudio, 2);

mainWindow.FileInfoLabel.Text = sprintf('%s%s | %.2fs | %dHz | %dch', ...
    name, ext, duration, mainWindow.SampleRate, channels);
end

%% TRANSPORT CONTROLS
function play(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Playback Error');
    return;
end

try
    sound(mainWindow.LoadedAudio, mainWindow.SampleRate);
    mainWindow.IsPlaying = true;
    mainWindow.StatusText.Text = 'Playing...';
catch ME
    uialert(mainWindow.Figure, sprintf('Playback error: %s', ME.message), 'Error');
end
end

function pause(mainWindow)
clear sound;
mainWindow.IsPlaying = false;
mainWindow.StatusText.Text = 'Paused';
end

function stop(mainWindow)
clear sound;
mainWindow.IsPlaying = false;
mainWindow.StatusText.Text = 'Stopped';
end

function playSelection(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

startTime = mainWindow.SelectionStartField.Value;
endTime = mainWindow.SelectionEndField.Value;

if startTime >= endTime
    uialert(mainWindow.Figure, 'Invalid selection', 'Error');
    return;
end

startSample = round(startTime * mainWindow.SampleRate) + 1;
endSample = round(endTime * mainWindow.SampleRate);

if endSample > length(mainWindow.LoadedAudio)
    endSample = length(mainWindow.LoadedAudio);
end

selection = mainWindow.LoadedAudio(startSample:endSample, :);
sound(selection, mainWindow.SampleRate);
end

function updateVolume(mainWindow, volume)
% Update master volume (affects playback)
mainWindow.VolumeSlider.Value = volume;
end

%% VIEW CONTROLS
function zoomIn(mainWindow)
if ~isempty(mainWindow.LoadedAudio)
    currentLimits = xlim(mainWindow.WaveformAxes);
    center = mean(currentLimits);
    range = diff(currentLimits) * 0.8 / 2;
    xlim(mainWindow.WaveformAxes, [center - range, center + range]);
end
end

function zoomOut(mainWindow)
if ~isempty(mainWindow.LoadedAudio)
    currentLimits = xlim(mainWindow.WaveformAxes);
    center = mean(currentLimits);
    range = diff(currentLimits) * 1.25 / 2;
    xlim(mainWindow.WaveformAxes, [center - range, center + range]);
end
end

function fitToWindow(mainWindow)
if ~isempty(mainWindow.LoadedAudio)
    duration = length(mainWindow.LoadedAudio) / mainWindow.SampleRate;
    xlim(mainWindow.WaveformAxes, [0, duration]);
end
end

%% TAB NAVIGATION
function switchToLibraryTab(mainWindow)
mainWindow.TabGroup.SelectedTab = mainWindow.TabGroup.Children(8);
end

function switchToSettingsTab(mainWindow)
mainWindow.TabGroup.SelectedTab = mainWindow.TabGroup.Children(9);
end

%% EDIT TAB FUNCTIONS
function initializeEditor(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

mainWindow.AudioEditor = AudioEditor(mainWindow.LoadedAudio, mainWindow.SampleRate);
updateEditHistory(mainWindow);
end

function updateSelectionInfo(mainWindow)
startTime = mainWindow.SelectionStartField.Value;
endTime = mainWindow.SelectionEndField.Value;

if endTime > startTime
    duration = endTime - startTime;
    mainWindow.SelectionDurationLabel.Text = sprintf('Duration: %.3fs', duration);
end
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
mainWindow.StatusText.Text = 'Audio trimmed successfully';
end

function cutAudio(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

startTime = mainWindow.SelectionStartField.Value;
endTime = mainWindow.SelectionEndField.Value;

if startTime >= endTime
    uialert(mainWindow.Figure, 'Invalid selection', 'Error');
    return;
end

mainWindow.AudioEditor.setSelection(startTime, endTime);
mainWindow.AudioEditor.cut();

mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
mainWindow.StatusText.Text = 'Selection cut to clipboard';
end

function copyAudio(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

startTime = mainWindow.SelectionStartField.Value;
endTime = mainWindow.SelectionEndField.Value;

if startTime >= endTime
    uialert(mainWindow.Figure, 'Invalid selection', 'Error');
    return;
end

mainWindow.AudioEditor.setSelection(startTime, endTime);
mainWindow.AudioEditor.copy();
mainWindow.StatusText.Text = 'Selection copied to clipboard';
end

function pasteAudio(mainWindow)
if isempty(mainWindow.AudioEditor) || isempty(mainWindow.AudioEditor.Clipboard)
    uialert(mainWindow.Figure, 'Clipboard is empty', 'Error');
    return;
end

position = mainWindow.PastePositionField.Value;
mainWindow.AudioEditor.paste(position);

mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
mainWindow.StatusText.Text = 'Audio pasted successfully';
end

function applyFadeInToSelection(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

duration = mainWindow.FadeInDurationField.Value;
curveType = mainWindow.FadeInCurveDropdown.Value;

mainWindow.AudioEditor.fadeIn(duration, curveType);
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
mainWindow.StatusText.Text = 'Fade in applied';
end

function applyFadeOutToSelection(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

duration = mainWindow.FadeOutDurationField.Value;
curveType = mainWindow.FadeOutCurveDropdown.Value;

mainWindow.AudioEditor.fadeOut(duration, curveType);
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
mainWindow.StatusText.Text = 'Fade out applied';
end

function showNormalizeDialog(mainWindow)
dialog = uifigure('Name', 'Normalize Audio', 'Position', [100, 100, 350, 250]);
grid = uigridlayout(dialog, [5, 2]);
grid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};
grid.Padding = [10, 10, 10, 10];

uilabel(grid, 'Text', 'Normalization Method:', 'FontWeight', 'bold');
uilabel(grid, 'Text', '');

methodButtons = uibuttongroup(grid);
methodButtons.Layout.Column = [1, 2];
peakBtn = uiradiobutton(methodButtons, 'Text', 'Peak (simple)', 'Position', [10, 50, 150, 22], 'Value', true);
rmsBtn = uiradiobutton(methodButtons, 'Text', 'RMS (loudness)', 'Position', [10, 25, 150, 22]);
lufsBtn = uiradiobutton(methodButtons, 'Text', 'LUFS (broadcast standard)', 'Position', [10, 0, 200, 22]);

uilabel(grid, 'Text', 'Target Level (dB):');
targetField = uispinner(grid, 'Value', -3, 'Limits', [-60, 0]);

uilabel(grid, 'Text', '');
uilabel(grid, 'Text', 'Peak: -3dB, RMS: -12dB, LUFS: -16dB recommended', 'FontSize', 9);

btnPanel = uipanel(grid);
btnPanel.Layout.Column = [1, 2];
btnGrid = uigridlayout(btnPanel, [1, 2]);
btnGrid.ColumnWidth = {'1x', '1x'};

uibutton(btnGrid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(btnGrid, 'Text', 'Apply', 'ButtonPushedFcn', @(src, event) applyNormalize(mainWindow, methodButtons, targetField.Value, dialog));
end

function applyNormalize(mainWindow, methodGroup, target, dialog)
if isempty(mainWindow.LoadedAudio)
    close(dialog);
    return;
end

if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

% Determine method
selectedBtn = methodGroup.SelectedObject;
if contains(selectedBtn.Text, 'Peak')
    method = 'peak';
elseif contains(selectedBtn.Text, 'RMS')
    method = 'rms';
else
    method = 'lufs';
end

mainWindow.AudioEditor.normalize(method, target);
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
close(dialog);
mainWindow.StatusText.Text = sprintf('Audio normalized to %ddB (%s)', target, method);
end

function reverseAudio(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

mainWindow.AudioEditor.reverse();
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
mainWindow.StatusText.Text = 'Audio reversed';
end

function removeDCOffset(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

mainWindow.AudioEditor.removeOffset();
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
mainWindow.StatusText.Text = 'DC offset removed';
end

function undoEdit(mainWindow)
if isempty(mainWindow.AudioEditor)
    return;
end

mainWindow.AudioEditor.undo();
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
mainWindow.StatusText.Text = 'Undo applied';
end

function redoEdit(mainWindow)
if isempty(mainWindow.AudioEditor)
    return;
end

mainWindow.AudioEditor.redo();
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
updateEditHistory(mainWindow);
mainWindow.StatusText.Text = 'Redo applied';
end

function updateEditHistory(mainWindow)
if isempty(mainWindow.AudioEditor)
    mainWindow.UndoButton.Enable = 'off';
    mainWindow.RedoButton.Enable = 'off';
    mainWindow.UndoMenuItem.Enable = 'off';
    mainWindow.RedoMenuItem.Enable = 'off';
    mainWindow.HistoryLabel.Text = 'No edit history';
    return;
end

% Update undo/redo button states
if mainWindow.AudioEditor.HistoryIndex > 1
    mainWindow.UndoButton.Enable = 'on';
    mainWindow.UndoMenuItem.Enable = 'on';
else
    mainWindow.UndoButton.Enable = 'off';
    mainWindow.UndoMenuItem.Enable = 'off';
end

if mainWindow.AudioEditor.HistoryIndex < length(mainWindow.AudioEditor.History)
    mainWindow.RedoButton.Enable = 'on';
    mainWindow.RedoMenuItem.Enable = 'on';
else
    mainWindow.RedoButton.Enable = 'off';
    mainWindow.RedoMenuItem.Enable = 'off';
end

% Update history label
historyText = sprintf('History: %d/%d', mainWindow.AudioEditor.HistoryIndex, length(mainWindow.AudioEditor.History));
mainWindow.HistoryLabel.Text = historyText;
end

function clearEditHistory(mainWindow)
mainWindow.AudioEditor = [];
updateEditHistory(mainWindow);
mainWindow.StatusText.Text = 'Edit history cleared';
end

function previewFadeIn(mainWindow)
uialert(mainWindow.Figure, 'Fade preview: Apply fade to hear result', 'Info');
end

function previewFadeOut(mainWindow)
uialert(mainWindow.Figure, 'Fade preview: Apply fade to hear result', 'Info');
end

function showRemoveSilenceDialog(mainWindow)
uialert(mainWindow.Figure, 'Remove silence feature: Set threshold and minimum duration, then apply', 'Coming Soon');
end

function showGainDialog(mainWindow)
dialog = uifigure('Name', 'Change Gain', 'Position', [100, 100, 300, 150]);
grid = uigridlayout(dialog, [3, 2]);
grid.RowHeight = {'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Gain (dB):');
gainField = uispinner(grid, 'Value', 0, 'Limits', [-60, 20], 'Step', 0.5);

uilabel(grid, 'Text', '');
uilabel(grid, 'Text', 'Positive = louder, Negative = quieter', 'FontSize', 9);

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Apply', 'ButtonPushedFcn', @(src, event) applyGain(mainWindow, gainField.Value, dialog));
end

function applyGain(mainWindow, gainDB, dialog)
if isempty(mainWindow.LoadedAudio)
    close(dialog);
    return;
end

gainLinear = db2mag(gainDB);
mainWindow.LoadedAudio = mainWindow.LoadedAudio * gainLinear;
updateWaveformDisplay(mainWindow);
close(dialog);
mainWindow.StatusText.Text = sprintf('Gain adjusted: %+.1f dB', gainDB);
end

function insertSilence(mainWindow)
uialert(mainWindow.Figure, 'Insert silence: Select position and duration', 'Coming Soon');
end

function generateTone(mainWindow)
uialert(mainWindow.Figure, 'Generate tone: Select frequency and duration', 'Coming Soon');
end

function generateNoise(mainWindow)
uialert(mainWindow.Figure, 'Generate noise: Select type and duration', 'Coming Soon');
end

%% EFFECTS TAB FUNCTIONS
function addEffectToChain(mainWindow)
effectType = mainWindow.AddEffectDropdown.Value;

% Add to chain
mainWindow.EffectChain{end+1} = struct('Type', effectType, 'Params', struct(), 'Enabled', true);

% Update list
updateEffectChainList(mainWindow);
mainWindow.StatusText.Text = sprintf('%s added to chain', effectType);
end

function updateEffectChainList(mainWindow)
if isempty(mainWindow.EffectChain)
    mainWindow.EffectChainListBox.Items = {'(Empty - Add effects above)'};
    return;
end

items = cell(length(mainWindow.EffectChain), 1);
for i = 1:length(mainWindow.EffectChain)
    effect = mainWindow.EffectChain{i};
    status = '';
    if ~effect.Enabled
        status = ' [BYPASSED]';
    end
    items{i} = sprintf('%d. %s%s', i, effect.Type, status);
end

mainWindow.EffectChainListBox.Items = items;
end

function selectEffect(mainWindow, selectedValue)
if contains(selectedValue, 'Empty')
    return;
end

% Parse effect number
tokens = regexp(selectedValue, '^(\d+)\.', 'tokens');
if isempty(tokens)
    return;
end

effectIdx = str2double(tokens{1}{1});
mainWindow.SelectedEffectIdx = effectIdx;

% Show effect parameters
showEffectParameters(mainWindow, effectIdx);
end

function showEffectParameters(mainWindow, effectIdx)
if effectIdx > length(mainWindow.EffectChain)
    return;
end

effect = mainWindow.EffectChain{effectIdx};

% Clear current controls
delete(mainWindow.EffectControlGrid.Children);

% Rebuild grid
mainWindow.EffectControlGrid = uigridlayout(mainWindow.EffectControlGrid.Parent, [6, 4]);
mainWindow.EffectControlGrid.RowHeight = repmat({'fit'}, 1, 6);
mainWindow.EffectControlGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit'};
mainWindow.EffectControlGrid.Padding = [10, 10, 10, 10];

% Title
titleLabel = uilabel(mainWindow.EffectControlGrid, ...
    'Text', sprintf('Effect %d: %s', effectIdx, effect.Type), ...
    'FontWeight', 'bold', 'FontSize', 12);
titleLabel.Layout.Column = [1, 4];

% Add common parameters based on effect type
switch effect.Type
    case 'Reverb'
        addReverbControls(mainWindow, effectIdx);
    case 'ConvolutionReverb'
        addConvolutionReverbControls(mainWindow, effectIdx);
    case 'Delay'
        addDelayControls(mainWindow, effectIdx);
    case 'EQ'
        addEQControls(mainWindow, effectIdx);
    case 'Compression'
        addCompressionControls(mainWindow, effectIdx);
    otherwise
        uilabel(mainWindow.EffectControlGrid, 'Text', 'Effect parameters coming soon');
end

% Bypass and remove buttons
uilabel(mainWindow.EffectControlGrid, 'Text', '');
uilabel(mainWindow.EffectControlGrid, 'Text', '');
uibutton(mainWindow.EffectControlGrid, 'Text', 'Bypass Effect', ...
    'ButtonPushedFcn', @(src, event) toggleEffectBypass(mainWindow, effectIdx));
uibutton(mainWindow.EffectControlGrid, 'Text', 'Remove from Chain', ...
    'ButtonPushedFcn', @(src, event) removeEffect(mainWindow, effectIdx));
end

function addReverbControls(mainWindow, effectIdx)
grid = mainWindow.EffectControlGrid;

uilabel(grid, 'Text', 'Room Size:');
roomSlider = uislider(grid, 'Value', 0.5, 'Limits', [0, 1]);
roomSlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '0.5');

uilabel(grid, 'Text', 'Decay Time (s):');
decaySlider = uislider(grid, 'Value', 2.0, 'Limits', [0.1, 10]);
decaySlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '2.0');

uilabel(grid, 'Text', 'Mix (Wet):');
mixSlider = uislider(grid, 'Value', 0.3, 'Limits', [0, 1]);
mixSlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '30%');
end

function addConvolutionReverbControls(mainWindow, effectIdx)
grid = mainWindow.EffectControlGrid;

uilabel(grid, 'Text', 'IR Space:');
irDropdown = uidropdown(grid, ...
    'Items', {'small_room', 'medium_room', 'concert_hall', 'chamber', 'plate', 'spring', 'ambience'}, ...
    'Value', 'medium_room');
irDropdown.Layout.Column = [2, 3];
uibutton(grid, 'Text', 'Load Custom IR', 'ButtonPushedFcn', @(src, event) loadCustomIR(mainWindow, effectIdx));

uilabel(grid, 'Text', 'Wet Amount:');
wetSlider = uislider(grid, 'Value', 0.3, 'Limits', [0, 1]);
wetSlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '30%');
end

function addDelayControls(mainWindow, effectIdx)
grid = mainWindow.EffectControlGrid;

uilabel(grid, 'Text', 'Delay Time (s):');
delaySlider = uislider(grid, 'Value', 0.25, 'Limits', [0.01, 2]);
delaySlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '0.25');

uilabel(grid, 'Text', 'Feedback:');
feedbackSlider = uislider(grid, 'Value', 0.3, 'Limits', [0, 0.95]);
feedbackSlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '0.3');
end

function addEQControls(mainWindow, effectIdx)
grid = mainWindow.EffectControlGrid;

uilabel(grid, 'Text', 'Low Gain (dB):');
lowSlider = uislider(grid, 'Value', 0, 'Limits', [-12, 12]);
lowSlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '0');

uilabel(grid, 'Text', 'Mid Gain (dB):');
midSlider = uislider(grid, 'Value', 0, 'Limits', [-12, 12]);
midSlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '0');

uilabel(grid, 'Text', 'High Gain (dB):');
highSlider = uislider(grid, 'Value', 0, 'Limits', [-12, 12]);
highSlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '0');
end

function addCompressionControls(mainWindow, effectIdx)
grid = mainWindow.EffectControlGrid;

uilabel(grid, 'Text', 'Threshold (dB):');
threshSlider = uislider(grid, 'Value', -12, 'Limits', [-60, 0]);
threshSlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '-12');

uilabel(grid, 'Text', 'Ratio:');
ratioSlider = uislider(grid, 'Value', 4, 'Limits', [1, 20]);
ratioSlider.Layout.Column = [2, 3];
uilabel(grid, 'Text', '4:1');
end

function toggleEffectBypass(mainWindow, effectIdx)
mainWindow.EffectChain{effectIdx}.Enabled = ~mainWindow.EffectChain{effectIdx}.Enabled;
updateEffectChainList(mainWindow);
mainWindow.StatusText.Text = sprintf('Effect %d bypass toggled', effectIdx);
end

function removeEffect(mainWindow, effectIdx)
mainWindow.EffectChain(effectIdx) = [];
updateEffectChainList(mainWindow);
mainWindow.StatusText.Text = sprintf('Effect removed from chain');
end

function clearEffectChain(mainWindow)
mainWindow.EffectChain = {};
updateEffectChainList(mainWindow);
mainWindow.StatusText.Text = 'Effect chain cleared';
end

function applyEffectChain(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

if isempty(mainWindow.EffectChain)
    uialert(mainWindow.Figure, 'Effect chain is empty', 'Info');
    return;
end

try
    processed = applyEffectChainToAudio(mainWindow, mainWindow.LoadedAudio);
    mainWindow.LoadedAudio = processed;
    updateWaveformDisplay(mainWindow);
    mainWindow.StatusText.Text = sprintf('Applied %d effects', length(mainWindow.EffectChain));
catch ME
    uialert(mainWindow.Figure, sprintf('Error applying effects: %s', ME.message), 'Error');
end
end

function processed = applyEffectChainToAudio(mainWindow, audio)
processed = audio;

for i = 1:length(mainWindow.EffectChain)
    effect = mainWindow.EffectChain{i};

    if ~effect.Enabled
        continue;
    end

    % Apply effect using AudioEffects
    try
        processed = AudioEffects(processed, effect.Type, ...
            'SampleRate', mainWindow.SampleRate);
    catch ME
        warning('Effect %d failed: %s', i, ME.message);
    end
end
end

function saveEffectPreset(mainWindow)
uialert(mainWindow.Figure, 'Save preset: Name your effect chain and save', 'Coming Soon');
end

function loadEffectPreset(mainWindow)
uialert(mainWindow.Figure, 'Load preset: Choose from saved presets', 'Coming Soon');
end

function loadCustomIR(mainWindow, effectIdx)
[file, path] = uigetfile({'*.wav', 'WAV Files'}, 'Load Impulse Response');
if file ~= 0
    mainWindow.EffectChain{effectIdx}.Params.IRFile = fullfile(path, file);
    mainWindow.StatusText.Text = sprintf('Custom IR loaded: %s', file);
end
end

function quickNormalize(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

if isempty(mainWindow.AudioEditor)
    initializeEditor(mainWindow);
end

mainWindow.AudioEditor.normalize('lufs', -16);
mainWindow.LoadedAudio = mainWindow.AudioEditor.getAudio();
updateWaveformDisplay(mainWindow);
mainWindow.StatusText.Text = 'Quick normalize: -16 LUFS';
end

function quickReverb(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

processed = AudioEffects(mainWindow.LoadedAudio, 'ConvolutionReverb', ...
    'IRSpace', 'medium_room', 'WetAmount', 0.3, 'SampleRate', mainWindow.SampleRate);

mainWindow.LoadedAudio = processed;
updateWaveformDisplay(mainWindow);
mainWindow.StatusText.Text = 'Quick reverb applied';
end

%% MIXER TAB FUNCTIONS (ENHANCED)
function loadTrackAudio(mainWindow, trackIndex)
[file, path] = uigetfile({'*.wav;*.mp3;*.flac', 'Audio Files'}, 'Select Audio File');
if file == 0
    return;
end

try
    [audioData, fs] = AudioLoader(fullfile(path, file));
    mainWindow.Mixer.loadTrack(trackIndex, audioData, fs);
    updateTimelineDisplay(mainWindow);
    mainWindow.StatusText.Text = sprintf('Track %d loaded: %s', trackIndex, file);
catch ME
    uialert(mainWindow.Figure, ['Error loading track: ' ME.message], 'Error');
end
end

function setTrackOffset(mainWindow, trackIndex, offset)
mainWindow.Mixer.setTrackOffset(trackIndex, offset);
updateTimelineDisplay(mainWindow);
end

function toggleTrackSolo(mainWindow, trackIndex, button)
currentState = mainWindow.Mixer.Tracks(trackIndex).Solo;
mainWindow.Mixer.setTrackSolo(trackIndex, ~currentState);

if ~currentState
    button.BackgroundColor = [1, 0.8, 0];
else
    button.BackgroundColor = [0.96, 0.96, 0.96];
end
end

function toggleTrackMute(mainWindow, trackIndex, button)
currentState = mainWindow.Mixer.Tracks(trackIndex).Mute;
mainWindow.Mixer.setTrackMute(trackIndex, ~currentState);

if ~currentState
    button.BackgroundColor = [1, 0.4, 0.4];
else
    button.BackgroundColor = [0.96, 0.96, 0.96];
end
end

function showTrackEffects(mainWindow, trackIndex)
uialert(mainWindow.Figure, sprintf('Track %d effects: Add effects to this track', trackIndex), 'Coming Soon');
end

function showFadeInDialog(mainWindow, trackIndex)
dialog = uifigure('Name', sprintf('Track %d Fade In', trackIndex), 'Position', [100, 100, 300, 200]);
grid = uigridlayout(dialog, [4, 2]);
grid.RowHeight = {'fit', 'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Duration (s):');
durationField = uispinner(grid, 'Value', 0.5, 'Limits', [0, 10], 'Step', 0.1);

uilabel(grid, 'Text', 'Curve Type:');
curveDropdown = uidropdown(grid, 'Items', {'linear', 'exponential', 'logarithmic', 'scurve'}, 'Value', 'scurve');

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Apply', 'ButtonPushedFcn', @(src, event) applyTrackFadeIn(mainWindow, trackIndex, durationField.Value, curveDropdown.Value, dialog));
end

function applyTrackFadeIn(mainWindow, trackIndex, duration, curveType, dialog)
mainWindow.Mixer.setTrackFadeIn(trackIndex, duration, curveType);
close(dialog);
mainWindow.StatusText.Text = sprintf('Fade in applied to Track %d', trackIndex);
end

function showFadeOutDialog(mainWindow, trackIndex)
dialog = uifigure('Name', sprintf('Track %d Fade Out', trackIndex), 'Position', [100, 100, 300, 200]);
grid = uigridlayout(dialog, [4, 2]);
grid.RowHeight = {'fit', 'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Duration (s):');
durationField = uispinner(grid, 'Value', 1.0, 'Limits', [0, 10], 'Step', 0.1);

uilabel(grid, 'Text', 'Curve Type:');
curveDropdown = uidropdown(grid, 'Items', {'linear', 'exponential', 'logarithmic', 'scurve'}, 'Value', 'exponential');

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Apply', 'ButtonPushedFcn', @(src, event) applyTrackFadeOut(mainWindow, trackIndex, durationField.Value, curveDropdown.Value, dialog));
end

function applyTrackFadeOut(mainWindow, trackIndex, duration, curveType, dialog)
mainWindow.Mixer.setTrackFadeOut(trackIndex, duration, curveType);
close(dialog);
mainWindow.StatusText.Text = sprintf('Fade out applied to Track %d', trackIndex);
end

function addMarkerDialog(mainWindow)
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
mainWindow.StatusText.Text = sprintf('Marker added: %s at %.1fs', label, time);
end

function alignTracks(mainWindow, method)
if strcmp(method, 'Manual')
    return;
end

methodMap = containers.Map(...
    {'Align to Start', 'Align to Peak', 'Align to End'}, ...
    {'start', 'peak', 'end'});

if isKey(methodMap, method)
    mainWindow.Mixer.alignTracks(methodMap(method));
    updateTimelineDisplay(mainWindow);
    mainWindow.StatusText.Text = sprintf('Tracks aligned: %s', method);
end
end

function zoomTimeline(mainWindow, factor)
if isfield(mainWindow, 'MixerTimelineAxes')
    currentXLim = xlim(mainWindow.MixerTimelineAxes);
    center = mean(currentXLim);
    range = diff(currentXLim) * factor / 2;
    xlim(mainWindow.MixerTimelineAxes, [center - range, center + range]);
end
end

function updateTimelineDisplay(mainWindow)
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
mainWindow.MixerTimelineAxes.XLabel.String = 'Time (s)';
mainWindow.MixerTimelineAxes.YLabel.String = 'Track';
mainWindow.MixerTimelineAxes.Title.String = 'Multi-track Timeline';
end

function processMix(mainWindow)
updateTimelineDisplay(mainWindow);

try
    mixedAudio = mainWindow.Mixer.processMix();
    mainWindow.LoadedAudio = mixedAudio;
    mainWindow.CurrentFile = 'Mixed Audio';
    mainWindow.SampleRate = mainWindow.Mixer.SampleRate;
    updateWaveformDisplay(mainWindow);
    mainWindow.StatusText.Text = 'Mix processed successfully';
catch ME
    uialert(mainWindow.Figure, ['Error processing mix: ' ME.message], 'Error');
end
end

function clearAllTracks(mainWindow)
for i = 1:mainWindow.Mixer.NumTracks
    mainWindow.Mixer.Tracks(i).AudioData = [];
    mainWindow.Mixer.Tracks(i).IsLoaded = false;
    mainWindow.Mixer.Tracks(i).Offset = 0;
end
updateTimelineDisplay(mainWindow);
mainWindow.StatusText.Text = 'All tracks cleared';
end

function exportMix(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No mix to export. Process mix first.', 'Warning');
    return;
end

[file, path] = uiputfile({'*.wav', 'WAV File'}, 'Export Mixed Audio');
if file == 0
    return;
end

try
    AudioExporter(mainWindow.LoadedAudio, fullfile(path, file), ...
        'SampleRate', mainWindow.SampleRate, 'BitDepth', 24);
    mainWindow.StatusText.Text = 'Mix exported successfully';
catch ME
    uialert(mainWindow.Figure, ['Error exporting: ' ME.message], 'Error');
end
end

function exportStems(mainWindow)
uialert(mainWindow.Figure, 'Export stems: Save each track individually', 'Coming Soon');
end

%% This file continues in the main MainWindow.m
% Append all remaining callback functions for Production, Research,
% Analysis, Library, and Settings tabs...

% Due to length, this is split across multiple files for organization.
% The complete implementation includes ~2000 more lines of callbacks.
