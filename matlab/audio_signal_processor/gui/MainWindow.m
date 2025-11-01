function mainWindow = MainWindow()
%MAINWINDOW Professional Audio Signal Processor - Complete GUI
%
%   MAINWINDOW = MAINWINDOW() creates the main application window with
%   all GUI components and functionality for professional audio processing.
%
%   Features:
%   ---------
%   - Waveform viewing and selection
%   - Professional audio editing (trim, cut, fade, normalize)
%   - Complete effects chain (11 effects including convolution reverb)
%   - Advanced multi-track mixer with time offsets and automation
%   - Music production tools (autotune, key/tempo detection, harmonizer)
%   - Research-grade analysis (wavelets, feature extraction, anti-aliasing)
%   - Comprehensive sample library management
%   - User preferences and settings
%
%   Tabs:
%   -----
%   1. Waveform - View and navigate audio
%   2. Edit - Audio editing with undo/redo
%   3. Effects - Effect chain management
%   4. Mixer - Multi-track mixing with timeline
%   5. Production - Music production tools
%   6. Analysis - Real-time audio analysis
%   7. Research - Advanced analysis tools
%   8. Library - Sample browser and management
%   9. Settings - Application preferences
%
%   Example:
%   --------
%   mainWindow = MainWindow();
%   mainWindow.show();

% Create main figure
mainWindow = struct();
mainWindow.Figure = uifigure('Name', 'Audio Signal Processor - Professional Edition', ...
    'Position', [50, 50, 1400, 900]);

% Initialize backend components
mainWindow.LibraryManager = SoundLibraryManager();
mainWindow.Mixer = MixerCoreEnhanced(8, 44100);  % ENHANCED MIXER
mainWindow.EffectsLibrary = InstrumentEffectsLibrary();
mainWindow.AudioEditor = [];  % Created on demand
mainWindow.MusicTools = MusicProductionTools();
mainWindow.WaveletProc = WaveletProcessor();
mainWindow.AdvancedAudio = AdvancedAudioProcessor();
mainWindow.AntiAliasing = AntiAliasingTools();

% Initialize state
mainWindow.IsPlaying = false;
mainWindow.CurrentFile = '';
mainWindow.LoadedAudio = [];
mainWindow.SampleRate = 44100;
mainWindow.EffectChain = {};  % Array of effects
mainWindow.Clipboard = [];  % For cut/copy/paste

% Create main layout
mainWindow.MainGrid = uigridlayout(mainWindow.Figure, [3, 1]);
mainWindow.MainGrid.RowHeight = {'fit', '1x', 'fit'};
mainWindow.MainGrid.ColumnWidth = {'1x'};
mainWindow.MainGrid.Padding = [5, 5, 5, 5];
mainWindow.MainGrid.RowSpacing = 5;
mainWindow.MainGrid.ColumnSpacing = 5;

% Create GUI components
createMenuBar(mainWindow);
createStatusBar(mainWindow);
createTabGroup(mainWindow);

% Add methods
mainWindow.show = @() show(mainWindow);
mainWindow.hide = @() hide(mainWindow);
mainWindow.close = @() close(mainWindow);
mainWindow.loadAudio = @(filename) loadAudio(mainWindow, filename);
mainWindow.play = @() play(mainWindow);
mainWindow.pause = @() pause(mainWindow);
mainWindow.stop = @() stop(mainWindow);

% Set close request function
mainWindow.Figure.CloseRequestFcn = @(src, event) delete(src);

% Show welcome message
uialert(mainWindow.Figure, ...
    sprintf(['Welcome to Audio Signal Processor - Professional Edition!\n\n', ...
    'All backend features are now accessible through the GUI.\n\n', ...
    'Start by loading audio (File → Load Audio or use Library tab).']),...
    'Welcome', 'Icon', 'info');
end

%% MENU BAR
function createMenuBar(mainWindow)
% File menu
fileMenu = uimenu(mainWindow.Figure, 'Text', 'File');
uimenu(fileMenu, 'Text', 'Load Audio...', 'MenuSelectedFcn', @(src, event) loadAudioDialog(mainWindow), ...
    'Accelerator', 'O');
uimenu(fileMenu, 'Text', 'Load from Library...', 'MenuSelectedFcn', @(src, event) switchToLibraryTab(mainWindow));
uimenu(fileMenu, 'Separator', 'on');
uimenu(fileMenu, 'Text', 'Export Audio...', 'MenuSelectedFcn', @(src, event) exportAudioDialog(mainWindow), ...
    'Accelerator', 'S');
uimenu(fileMenu, 'Text', 'Export with Effects...', 'MenuSelectedFcn', @(src, event) exportWithEffects(mainWindow));
uimenu(fileMenu, 'Separator', 'on');
uimenu(fileMenu, 'Text', 'Exit', 'MenuSelectedFcn', @(src, event) close(mainWindow));

% Edit menu
editMenu = uimenu(mainWindow.Figure, 'Text', 'Edit');
mainWindow.UndoMenuItem = uimenu(editMenu, 'Text', 'Undo', 'MenuSelectedFcn', @(src, event) undoEdit(mainWindow), ...
    'Enable', 'off', 'Accelerator', 'Z');
mainWindow.RedoMenuItem = uimenu(editMenu, 'Text', 'Redo', 'MenuSelectedFcn', @(src, event) redoEdit(mainWindow), ...
    'Enable', 'off', 'Accelerator', 'Y');
uimenu(editMenu, 'Separator', 'on');
uimenu(editMenu, 'Text', 'Select All', 'MenuSelectedFcn', @(src, event) selectAllAudio(mainWindow), 'Accelerator', 'A');
uimenu(editMenu, 'Text', 'Cut', 'MenuSelectedFcn', @(src, event) cutAudio(mainWindow), 'Accelerator', 'X');
uimenu(editMenu, 'Text', 'Copy', 'MenuSelectedFcn', @(src, event) copyAudio(mainWindow), 'Accelerator', 'C');
uimenu(editMenu, 'Text', 'Paste', 'MenuSelectedFcn', @(src, event) pasteAudio(mainWindow), 'Accelerator', 'V');
uimenu(editMenu, 'Separator', 'on');
uimenu(editMenu, 'Text', 'Preferences...', 'MenuSelectedFcn', @(src, event) switchToSettingsTab(mainWindow));

% View menu
viewMenu = uimenu(mainWindow.Figure, 'Text', 'View');
uimenu(viewMenu, 'Text', 'Zoom In', 'MenuSelectedFcn', @(src, event) zoomIn(mainWindow), 'Accelerator', '=');
uimenu(viewMenu, 'Text', 'Zoom Out', 'MenuSelectedFcn', @(src, event) zoomOut(mainWindow), 'Accelerator', '-');
uimenu(viewMenu, 'Text', 'Fit to Window', 'MenuSelectedFcn', @(src, event) fitToWindow(mainWindow), 'Accelerator', '0');

% Effects menu
effectsMenu = uimenu(mainWindow.Figure, 'Text', 'Effects');
uimenu(effectsMenu, 'Text', 'Apply Effect Chain', 'MenuSelectedFcn', @(src, event) applyEffectChain(mainWindow), 'Accelerator', 'E');
uimenu(effectsMenu, 'Text', 'Clear Effect Chain', 'MenuSelectedFcn', @(src, event) clearEffectChain(mainWindow));
uimenu(effectsMenu, 'Separator', 'on');
uimenu(effectsMenu, 'Text', 'Quick Normalize', 'MenuSelectedFcn', @(src, event) quickNormalize(mainWindow), 'Accelerator', 'N');
uimenu(effectsMenu, 'Text', 'Quick Reverb', 'MenuSelectedFcn', @(src, event) quickReverb(mainWindow), 'Accelerator', 'R');

% Tools menu
toolsMenu = uimenu(mainWindow.Figure, 'Text', 'Tools');
uimenu(toolsMenu, 'Text', 'Autotune...', 'MenuSelectedFcn', @(src, event) showAutotuneDialog(mainWindow));
uimenu(toolsMenu, 'Text', 'Detect Key', 'MenuSelectedFcn', @(src, event) detectKeyQuick(mainWindow));
uimenu(toolsMenu, 'Text', 'Detect Tempo', 'MenuSelectedFcn', @(src, event) detectTempoQuick(mainWindow));
uimenu(toolsMenu, 'Separator', 'on');
uimenu(toolsMenu, 'Text', 'Batch Process...', 'MenuSelectedFcn', @(src, event) showBatchProcessor(mainWindow));

% Help menu
helpMenu = uimenu(mainWindow.Figure, 'Text', 'Help');
uimenu(helpMenu, 'Text', 'Quick Start Guide', 'MenuSelectedFcn', @(src, event) showQuickStart(mainWindow));
uimenu(helpMenu, 'Text', 'Keyboard Shortcuts', 'MenuSelectedFcn', @(src, event) showShortcuts(mainWindow));
uimenu(helpMenu, 'Separator', 'on');
uimenu(helpMenu, 'Text', 'About', 'MenuSelectedFcn', @(src, event) showAbout(mainWindow));
end

%% STATUS BAR
function createStatusBar(mainWindow)
statusBar = uipanel(mainWindow.MainGrid);
statusBar.Layout.Row = 3;
statusBar.Layout.Column = 1;

statusGrid = uigridlayout(statusBar, [1, 4]);
statusGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit'};
statusGrid.Padding = [5, 5, 5, 5];

% Transport controls
transportPanel = uipanel(statusGrid);
transportPanel.Layout.Column = 1;
transportGrid = uigridlayout(transportPanel, [1, 3]);
transportGrid.ColumnWidth = {'fit', 'fit', 'fit'};
transportGrid.Padding = [2, 2, 2, 2];

mainWindow.PlayButton = uibutton(transportGrid, 'Text', '▶', ...
    'ButtonPushedFcn', @(src, event) play(mainWindow), 'Tooltip', 'Play (Space)');
mainWindow.PauseButton = uibutton(transportGrid, 'Text', '⏸', ...
    'ButtonPushedFcn', @(src, event) pause(mainWindow), 'Tooltip', 'Pause');
mainWindow.StopButton = uibutton(transportGrid, 'Text', '⏹', ...
    'ButtonPushedFcn', @(src, event) stop(mainWindow), 'Tooltip', 'Stop');

% Status text
mainWindow.StatusText = uilabel(statusGrid, 'Text', 'Ready - Load audio to begin');
mainWindow.StatusText.Layout.Column = 2;

% Time display
mainWindow.TimeDisplay = uilabel(statusGrid, 'Text', '00:00 / 00:00');
mainWindow.TimeDisplay.Layout.Column = 3;

% Volume control
volumePanel = uipanel(statusGrid);
volumePanel.Layout.Column = 4;
volumeGrid = uigridlayout(volumePanel, [1, 2]);
volumeGrid.ColumnWidth = {'fit', 100};
volumeGrid.Padding = [2, 2, 2, 2];

uilabel(volumeGrid, 'Text', 'Vol:');
mainWindow.VolumeSlider = uislider(volumeGrid, 'Value', 0.7, ...
    'Limits', [0, 1], ...
    'ValueChangedFcn', @(src, event) updateVolume(mainWindow, src.Value));
end

%% TAB GROUP
function createTabGroup(mainWindow)
mainWindow.TabGroup = uitabgroup(mainWindow.MainGrid);
mainWindow.TabGroup.Layout.Row = 2;
mainWindow.TabGroup.Layout.Column = 1;

% Create all 9 tabs
waveformTab = uitab(mainWindow.TabGroup, 'Title', '📊 Waveform');
createWaveformPanel(mainWindow, waveformTab);

editTab = uitab(mainWindow.TabGroup, 'Title', '✂️ Edit');
createEditPanel(mainWindow, editTab);

effectsTab = uitab(mainWindow.TabGroup, 'Title', '🎛️ Effects');
createEffectsPanel(mainWindow, effectsTab);

mixerTab = uitab(mainWindow.TabGroup, 'Title', '🎚️ Mixer');
createMixerPanel(mainWindow, mixerTab);

productionTab = uitab(mainWindow.TabGroup, 'Title', '🎵 Production');
createProductionPanel(mainWindow, productionTab);

analysisTab = uitab(mainWindow.TabGroup, 'Title', '📈 Analysis');
createAnalysisPanel(mainWindow, analysisTab);

researchTab = uitab(mainWindow.TabGroup, 'Title', '🔬 Research');
createResearchPanel(mainWindow, researchTab);

libraryTab = uitab(mainWindow.TabGroup, 'Title', '📚 Library');
createLibraryPanel(mainWindow, libraryTab);

settingsTab = uitab(mainWindow.TabGroup, 'Title', '⚙️ Settings');
createSettingsPanel(mainWindow, settingsTab);
end

%% TAB 1: WAVEFORM
function createWaveformPanel(mainWindow, parent)
waveformGrid = uigridlayout(parent, [2, 1]);
waveformGrid.RowHeight = {'1x', 'fit'};
waveformGrid.Padding = [5, 5, 5, 5];

% Waveform display area
waveformPanel = uipanel(waveformGrid);
waveformPanel.Layout.Row = 1;

axesGrid = uigridlayout(waveformPanel, [1, 1]);
axesGrid.Padding = [10, 10, 10, 10];

mainWindow.WaveformAxes = uiaxes(axesGrid);
mainWindow.WaveformAxes.XLabel.String = 'Time (s)';
mainWindow.WaveformAxes.YLabel.String = 'Amplitude';
mainWindow.WaveformAxes.Title.String = 'Audio Waveform';
grid(mainWindow.WaveformAxes, 'on');

% Waveform controls
controlsPanel = uipanel(waveformGrid);
controlsPanel.Layout.Row = 2;

controlsGrid = uigridlayout(controlsPanel, [1, 6]);
controlsGrid.ColumnWidth = {'fit', 'fit', 'fit', 'fit', '1x', 'fit'};
controlsGrid.Padding = [5, 5, 5, 5];

uibutton(controlsGrid, 'Text', 'Load Audio', ...
    'ButtonPushedFcn', @(src, event) loadAudioDialog(mainWindow), ...
    'Tooltip', 'Load audio file (Ctrl+O)');

uibutton(controlsGrid, 'Text', 'Zoom In', ...
    'ButtonPushedFcn', @(src, event) zoomIn(mainWindow), ...
    'Tooltip', 'Zoom in (Ctrl+=)');

uibutton(controlsGrid, 'Text', 'Zoom Out', ...
    'ButtonPushedFcn', @(src, event) zoomOut(mainWindow), ...
    'Tooltip', 'Zoom out (Ctrl+-)');

uibutton(controlsGrid, 'Text', 'Fit View', ...
    'ButtonPushedFcn', @(src, event) fitToWindow(mainWindow), ...
    'Tooltip', 'Fit to window (Ctrl+0)');

% File info
mainWindow.FileInfoLabel = uilabel(controlsGrid, 'Text', 'No audio loaded');

uibutton(controlsGrid, 'Text', 'Play Selected', ...
    'ButtonPushedFcn', @(src, event) playSelection(mainWindow));
end

%% TAB 2: EDIT
function createEditPanel(mainWindow, parent)
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
mainWindow.SelectionStartField = uispinner(selectionGrid, 'Value', 0, 'Limits', [0, 10000], ...
    'ValueDisplayFormat', '%.3fs', 'Tooltip', 'Selection start time', ...
    'ValueChangedFcn', @(src, event) updateSelectionInfo(mainWindow));
mainWindow.SelectionEndField = uispinner(selectionGrid, 'Value', 0, 'Limits', [0, 10000], ...
    'ValueDisplayFormat', '%.3fs', 'Tooltip', 'Selection end time', ...
    'ValueChangedFcn', @(src, event) updateSelectionInfo(mainWindow));
mainWindow.SelectionDurationLabel = uilabel(selectionGrid, 'Text', 'Duration: 0.000s');

uilabel(selectionGrid, 'Text', 'Actions:');
uibutton(selectionGrid, 'Text', 'Trim', 'ButtonPushedFcn', @(src, event) trimAudio(mainWindow), ...
    'Tooltip', 'Keep selection, delete rest');
uibutton(selectionGrid, 'Text', 'Cut', 'ButtonPushedFcn', @(src, event) cutAudio(mainWindow), ...
    'Tooltip', 'Cut selection to clipboard (Ctrl+X)');
uibutton(selectionGrid, 'Text', 'Copy', 'ButtonPushedFcn', @(src, event) copyAudio(mainWindow), ...
    'Tooltip', 'Copy selection to clipboard (Ctrl+C)');

uilabel(selectionGrid, 'Text', '');
mainWindow.PastePositionField = uispinner(selectionGrid, 'Value', 0, 'Limits', [0, 10000], ...
    'ValueDisplayFormat', '%.3fs', 'Tooltip', 'Paste position');
uibutton(selectionGrid, 'Text', 'Paste', 'ButtonPushedFcn', @(src, event) pasteAudio(mainWindow), ...
    'Tooltip', 'Paste at position (Ctrl+V)');
uibutton(selectionGrid, 'Text', 'Select All', 'ButtonPushedFcn', @(src, event) selectAllAudio(mainWindow), ...
    'Tooltip', 'Select all audio (Ctrl+A)');

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

uibutton(processingGrid, 'Text', 'Normalize', 'ButtonPushedFcn', @(src, event) showNormalizeDialog(mainWindow), ...
    'Tooltip', 'Normalize audio (peak, RMS, or LUFS)');
uibutton(processingGrid, 'Text', 'Remove Silence', 'ButtonPushedFcn', @(src, event) showRemoveSilenceDialog(mainWindow), ...
    'Tooltip', 'Remove silent sections');
uibutton(processingGrid, 'Text', 'Reverse', 'ButtonPushedFcn', @(src, event) reverseAudio(mainWindow), ...
    'Tooltip', 'Reverse audio');
uibutton(processingGrid, 'Text', 'Remove DC Offset', 'ButtonPushedFcn', @(src, event) removeDCOffset(mainWindow), ...
    'Tooltip', 'Remove DC bias');
uibutton(processingGrid, 'Text', 'Change Gain', 'ButtonPushedFcn', @(src, event) showGainDialog(mainWindow), ...
    'Tooltip', 'Adjust volume level');

uibutton(processingGrid, 'Text', 'Insert Silence', 'ButtonPushedFcn', @(src, event) insertSilence(mainWindow));
uibutton(processingGrid, 'Text', 'Generate Tone', 'ButtonPushedFcn', @(src, event) generateTone(mainWindow));
uibutton(processingGrid, 'Text', 'Generate Noise', 'ButtonPushedFcn', @(src, event) generateNoise(mainWindow));
uilabel(processingGrid, 'Text', '');
uilabel(processingGrid, 'Text', '');

% History
historyPanel = uipanel(editGrid, 'Title', 'Undo/Redo History');
historyPanel.Layout.Row = 4;
historyGrid = uigridlayout(historyPanel, [1, 4]);
historyGrid.ColumnWidth = {'fit', 'fit', '1x', 'fit'};
historyGrid.Padding = [5, 5, 5, 5];

mainWindow.UndoButton = uibutton(historyGrid, 'Text', '◀ Undo', ...
    'ButtonPushedFcn', @(src, event) undoEdit(mainWindow), ...
    'Enable', 'off', 'Tooltip', 'Undo last edit (Ctrl+Z)');
mainWindow.RedoButton = uibutton(historyGrid, 'Text', 'Redo ▶', ...
    'ButtonPushedFcn', @(src, event) redoEdit(mainWindow), ...
    'Enable', 'off', 'Tooltip', 'Redo last undo (Ctrl+Y)');
mainWindow.HistoryLabel = uilabel(historyGrid, 'Text', 'No edit history');
uibutton(historyGrid, 'Text', 'Clear History', 'ButtonPushedFcn', @(src, event) clearEditHistory(mainWindow));
end

%% TAB 3: EFFECTS
function createEffectsPanel(mainWindow, parent)
effectsGrid = uigridlayout(parent, [3, 1]);
effectsGrid.RowHeight = {'fit', '1x', 'fit'};
effectsGrid.Padding = [10, 10, 10, 10];
effectsGrid.RowSpacing = 10;

% Effect Chain Header
chainHeaderPanel = uipanel(effectsGrid, 'Title', 'Effect Chain');
chainHeaderPanel.Layout.Row = 1;
chainHeaderGrid = uigridlayout(chainHeaderPanel, [1, 6]);
chainHeaderGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit', 'fit', 'fit'};
chainHeaderGrid.Padding = [5, 5, 5, 5];

uilabel(chainHeaderGrid, 'Text', 'Add Effect:');
mainWindow.AddEffectDropdown = uidropdown(chainHeaderGrid, ...
    'Items', {'Reverb', 'ConvolutionReverb', 'Delay', 'EQ', 'Compression', 'Limiting', 'Distortion', 'Chorus', 'Flanger', 'PitchShift', 'TimeStretch'}, ...
    'Value', 'Reverb');
uibutton(chainHeaderGrid, 'Text', '+ Add', 'ButtonPushedFcn', @(src, event) addEffectToChain(mainWindow));
uibutton(chainHeaderGrid, 'Text', 'Clear All', 'ButtonPushedFcn', @(src, event) clearEffectChain(mainWindow));
uibutton(chainHeaderGrid, 'Text', 'Save Preset', 'ButtonPushedFcn', @(src, event) saveEffectPreset(mainWindow));
uibutton(chainHeaderGrid, 'Text', 'Load Preset', 'ButtonPushedFcn', @(src, event) loadEffectPreset(mainWindow));

% Effect Chain List
chainListPanel = uipanel(effectsGrid, 'Title', 'Current Effects');
chainListPanel.Layout.Row = 2;
chainListGrid = uigridlayout(chainListPanel, [1, 1]);
chainListGrid.Padding = [5, 5, 5, 5];

mainWindow.EffectChainListBox = uilistbox(chainListGrid, ...
    'Items', {'(Empty - Add effects above)'}, ...
    'ValueChangedFcn', @(src, event) selectEffect(mainWindow, src.Value));

% Effect Controls
effectControlPanel = uipanel(effectsGrid, 'Title', 'Effect Parameters');
effectControlPanel.Layout.Row = 3;
mainWindow.EffectControlGrid = uigridlayout(effectControlPanel, [5, 4]);
mainWindow.EffectControlGrid.RowHeight = repmat({'fit'}, 1, 5);
mainWindow.EffectControlGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit'};
mainWindow.EffectControlGrid.Padding = [10, 10, 10, 10];

% Placeholder text
uilabel(mainWindow.EffectControlGrid, 'Text', 'Select an effect from the chain to edit its parameters.', ...
    'HorizontalAlignment', 'center');
end

%% TAB 4: MIXER (ENHANCED)
function createMixerPanel(mainWindow, parent)
mixerGrid = uigridlayout(parent, [4, 1]);
mixerGrid.RowHeight = {'fit', 'fit', '1x', 'fit'};
mixerGrid.Padding = [5, 5, 5, 5];

% Timeline View
timelinePanel = uipanel(mixerGrid, 'Title', 'Timeline');
timelinePanel.Layout.Row = 1;
timelineAxesGrid = uigridlayout(timelinePanel, [1, 1]);
timelineAxesGrid.Padding = [5, 5, 5, 5];

mainWindow.MixerTimelineAxes = uiaxes(timelineAxesGrid);
mainWindow.MixerTimelineAxes.XLabel.String = 'Time (s)';
mainWindow.MixerTimelineAxes.YLabel.String = 'Track';
mainWindow.MixerTimelineAxes.Title.String = 'Multi-track Timeline';
grid(mainWindow.MixerTimelineAxes, 'on');

% Timeline Controls
timelineControlPanel = uipanel(mixerGrid);
timelineControlPanel.Layout.Row = 2;
timelineControlGrid = uigridlayout(timelineControlPanel, [1, 6]);
timelineControlGrid.ColumnWidth = {'fit', 'fit', 'fit', 'fit', '1x', 'fit'};
timelineControlGrid.Padding = [5, 5, 5, 5];

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
uibutton(timelineControlGrid, 'Text', 'Update Timeline', ...
    'ButtonPushedFcn', @(src, event) updateTimelineDisplay(mainWindow));

% Tracks Panel
tracksPanel = uipanel(mixerGrid, 'Title', 'Tracks');
tracksPanel.Layout.Row = 3;

tracksScroll = uigridlayout(tracksPanel, [1, 8]);
tracksScroll.ColumnWidth = repmat({100}, 1, 8);
tracksScroll.Padding = [5, 5, 5, 5];
tracksScroll.ColumnSpacing = 10;

% Create 8 track strips
mainWindow.TrackStrips = cell(8, 1);
for i = 1:8
    trackStrip = uipanel(tracksScroll);
    trackStrip.Title = sprintf('Track %d', i);

    trackGrid = uigridlayout(trackStrip, [10, 1]);
    trackGrid.RowHeight = {'fit', 'fit', '1x', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit'};
    trackGrid.ColumnWidth = {90};
    trackGrid.Padding = [5, 5, 5, 5];
    trackGrid.RowSpacing = 3;

    % Track name
    uilabel(trackGrid, 'Text', sprintf('Track %d', i), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');

    % Load button
    uibutton(trackGrid, 'Text', 'Load', ...
        'ButtonPushedFcn', @(src, event) loadTrackAudio(mainWindow, i));

    % Volume fader
    volumeSlider = uislider(trackGrid, 'Orientation', 'vertical', ...
        'Value', 0.8, 'Limits', [0, 1], ...
        'ValueChangedFcn', @(src, event) setTrackVolume(mainWindow.Mixer, i, src.Value));

    % Volume label
    uilabel(trackGrid, 'Text', 'Volume', 'HorizontalAlignment', 'center', 'FontSize', 8);

    % Pan knob
    panKnob = uiknob(trackGrid, 'Value', 0, 'Limits', [-1, 1], ...
        'ValueChangedFcn', @(src, event) setTrackPan(mainWindow.Mixer, i, src.Value));

    % Solo/Mute buttons
    soloButton = uibutton(trackGrid, 'Text', 'S', ...
        'ButtonPushedFcn', @(src, event) toggleTrackSolo(mainWindow, i, src), ...
        'Tooltip', 'Solo');
    muteButton = uibutton(trackGrid, 'Text', 'M', ...
        'ButtonPushedFcn', @(src, event) toggleTrackMute(mainWindow, i, src), ...
        'Tooltip', 'Mute');

    % Effects button
    uibutton(trackGrid, 'Text', 'FX', ...
        'ButtonPushedFcn', @(src, event) showTrackEffects(mainWindow, i), ...
        'Tooltip', 'Add effects to track');

    % Offset control
    offsetSpinner = uispinner(trackGrid, 'Value', 0, 'Limits', [0, 300], 'Step', 0.1, ...
        'ValueDisplayFormat', '%.1fs', ...
        'ValueChangedFcn', @(src, event) setTrackOffset(mainWindow, i, src.Value), ...
        'Tooltip', 'Time offset (seconds)');

    % Fade buttons
    fadePanel = uipanel(trackGrid);
    fadeGrid = uigridlayout(fadePanel, [1, 2]);
    fadeGrid.ColumnWidth = {'1x', '1x'};
    fadeGrid.Padding = [1, 1, 1, 1];

    uibutton(fadeGrid, 'Text', 'FI', 'Tooltip', 'Fade In', 'FontSize', 8, ...
        'ButtonPushedFcn', @(src, event) showFadeInDialog(mainWindow, i));
    uibutton(fadeGrid, 'Text', 'FO', 'Tooltip', 'Fade Out', 'FontSize', 8, ...
        'ButtonPushedFcn', @(src, event) showFadeOutDialog(mainWindow, i));

    mainWindow.TrackStrips{i} = struct('Panel', trackStrip, 'Volume', volumeSlider, ...
        'Pan', panKnob, 'Offset', offsetSpinner, 'Solo', soloButton, 'Mute', muteButton);
end

% Master Section
masterPanel = uipanel(mixerGrid, 'Title', 'Master');
masterPanel.Layout.Row = 4;

masterGrid = uigridlayout(masterPanel, [1, 6]);
masterGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit', 'fit', 'fit'};
masterGrid.Padding = [5, 5, 5, 5];

uilabel(masterGrid, 'Text', 'Master Volume:', 'FontWeight', 'bold');
mainWindow.MasterVolumeSlider = uislider(masterGrid, 'Value', 0.8, 'Limits', [0, 1], ...
    'ValueChangedFcn', @(src, event) updateVolume(mainWindow, src.Value));

uibutton(masterGrid, 'Text', 'Process Mix', ...
    'ButtonPushedFcn', @(src, event) processMix(mainWindow), ...
    'Tooltip', 'Process all tracks into one audio file');
uibutton(masterGrid, 'Text', 'Clear All', ...
    'ButtonPushedFcn', @(src, event) clearAllTracks(mainWindow));
uibutton(masterGrid, 'Text', 'Export Mix', ...
    'ButtonPushedFcn', @(src, event) exportMix(mainWindow));
uibutton(masterGrid, 'Text', 'Export Stems', ...
    'ButtonPushedFcn', @(src, event) exportStems(mainWindow));
end

%% TAB 5: PRODUCTION
function createProductionPanel(mainWindow, parent)
productionGrid = uigridlayout(parent, [4, 1]);
productionGrid.RowHeight = {'fit', 'fit', 'fit', '1x'};
productionGrid.Padding = [10, 10, 10, 10];
productionGrid.RowSpacing = 10;

% Pitch Correction (Autotune)
autotunePanel = uipanel(productionGrid, 'Title', 'Pitch Correction (Autotune)');
autotunePanel.Layout.Row = 1;
autotuneGrid = uigridlayout(autotunePanel, [3, 4]);
autotuneGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit'};
autotuneGrid.Padding = [5, 5, 5, 5];

uilabel(autotuneGrid, 'Text', 'Key:');
mainWindow.AutotuneKeyDropdown = uidropdown(autotuneGrid, ...
    'Items', {'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'}, ...
    'Value', 'C');
uilabel(autotuneGrid, 'Text', 'Scale:');
mainWindow.AutotuneScaleDropdown = uidropdown(autotuneGrid, ...
    'Items', {'major', 'minor', 'harmonic_minor', 'melodic_minor', 'dorian', 'phrygian', 'lydian', 'mixolydian'}, ...
    'Value', 'major');

uilabel(autotuneGrid, 'Text', 'Strength:');
mainWindow.AutotuneStrengthSlider = uislider(autotuneGrid, 'Value', 0.8, 'Limits', [0, 1], ...
    'Tooltip', '0 = Natural, 1 = Robotic');
uilabel(autotuneGrid, 'Text', 'Speed (ms):');
mainWindow.AutotuneSpeedSpinner = uispinner(autotuneGrid, 'Value', 10, 'Limits', [1, 100], ...
    'Tooltip', 'Pitch correction speed');

mainWindow.AutotuneFormantCheckbox = uicheckbox(autotuneGrid, 'Text', 'Preserve Formants', 'Value', true);
uibutton(autotuneGrid, 'Text', 'Apply Autotune', ...
    'ButtonPushedFcn', @(src, event) applyAutotune(mainWindow), ...
    'Tooltip', 'Apply pitch correction to loaded audio');
uibutton(autotuneGrid, 'Text', 'Preview', ...
    'ButtonPushedFcn', @(src, event) previewAutotune(mainWindow));
uilabel(autotuneGrid, 'Text', '');

% Musical Analysis
analysisPanel = uipanel(productionGrid, 'Title', 'Musical Analysis');
analysisPanel.Layout.Row = 2;
analysisGrid = uigridlayout(analysisPanel, [3, 3]);
analysisGrid.ColumnWidth = {'fit', '1x', 'fit'};
analysisGrid.Padding = [5, 5, 5, 5];

uilabel(analysisGrid, 'Text', 'Detected Key:');
mainWindow.DetectedKeyLabel = uilabel(analysisGrid, 'Text', '(Not analyzed)');
uibutton(analysisGrid, 'Text', 'Analyze Key', ...
    'ButtonPushedFcn', @(src, event) detectKeyQuick(mainWindow));

uilabel(analysisGrid, 'Text', 'Detected Tempo:');
mainWindow.DetectedTempoLabel = uilabel(analysisGrid, 'Text', '(Not analyzed)');
uibutton(analysisGrid, 'Text', 'Analyze Tempo', ...
    'ButtonPushedFcn', @(src, event) detectTempoQuick(mainWindow));

uilabel(analysisGrid, 'Text', 'Chord Detection:');
uilabel(analysisGrid, 'Text', '');
uibutton(analysisGrid, 'Text', 'Detect Chords', ...
    'ButtonPushedFcn', @(src, event) detectChordsDetailed(mainWindow));

% Rhythm & Timing
rhythmPanel = uipanel(productionGrid, 'Title', 'Rhythm & Timing');
rhythmPanel.Layout.Row = 3;
rhythmGrid = uigridlayout(rhythmPanel, [2, 4]);
rhythmGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit'};
rhythmGrid.Padding = [5, 5, 5, 5];

uilabel(rhythmGrid, 'Text', 'Generate Click:');
mainWindow.ClickBPMSpinner = uispinner(rhythmGrid, 'Value', 120, 'Limits', [40, 300], 'Tooltip', 'BPM');
mainWindow.ClickBarsSpinner = uispinner(rhythmGrid, 'Value', 16, 'Limits', [1, 128], 'Tooltip', 'Number of bars');
uibutton(rhythmGrid, 'Text', 'Generate', ...
    'ButtonPushedFcn', @(src, event) generateClickTrack(mainWindow));

uilabel(rhythmGrid, 'Text', 'Quantize Audio:');
mainWindow.QuantizeBPMSpinner = uispinner(rhythmGrid, 'Value', 120, 'Limits', [40, 300]);
mainWindow.QuantizeStrengthSlider = uislider(rhythmGrid, 'Value', 0.5, 'Limits', [0, 1], 'Tooltip', 'Quantize strength');
uibutton(rhythmGrid, 'Text', 'Quantize', ...
    'ButtonPushedFcn', @(src, event) quantizeAudio(mainWindow));

% Creative Tools
creativePanel = uipanel(productionGrid, 'Title', 'Creative Tools');
creativePanel.Layout.Row = 4;
creativeGrid = uigridlayout(creativePanel, [3, 3]);
creativeGrid.ColumnWidth = {'1x', '1x', '1x'};
creativeGrid.Padding = [5, 5, 5, 5];

uibutton(creativeGrid, 'Text', 'Harmonizer', ...
    'ButtonPushedFcn', @(src, event) showHarmonizerDialog(mainWindow), ...
    'Tooltip', 'Generate harmonies');
uibutton(creativeGrid, 'Text', 'Vocoder', ...
    'ButtonPushedFcn', @(src, event) showVocoderDialog(mainWindow), ...
    'Tooltip', 'Apply vocoder effect');
uibutton(creativeGrid, 'Text', 'Audio→MIDI', ...
    'ButtonPushedFcn', @(src, event) audioToMIDI(mainWindow), ...
    'Tooltip', 'Convert audio to MIDI');

uibutton(creativeGrid, 'Text', 'Pitch Shift', ...
    'ButtonPushedFcn', @(src, event) showPitchShiftDialog(mainWindow));
uibutton(creativeGrid, 'Text', 'Time Stretch', ...
    'ButtonPushedFcn', @(src, event) showTimeStretchDialog(mainWindow));
uilabel(creativeGrid, 'Text', '');
end

%% TAB 6: ANALYSIS (Keep existing, minor updates)
function createAnalysisPanel(mainWindow, parent)
analysisGrid = uigridlayout(parent, [3, 2]);
analysisGrid.RowHeight = {'1x', '1x', 'fit'};
analysisGrid.ColumnWidth = {'1x', '1x'};
analysisGrid.Padding = [10, 10, 10, 10];
analysisGrid.RowSpacing = 10;
analysisGrid.ColumnSpacing = 10;

% Spectrogram panel
spectrogramPanel = uipanel(analysisGrid, 'Title', 'Spectrogram');
spectrogramPanel.Layout.Row = 1;
spectrogramPanel.Layout.Column = 1;

mainWindow.SpectrogramAxes = uiaxes(spectrogramPanel);
mainWindow.SpectrogramAxes.XLabel.String = 'Time (s)';
mainWindow.SpectrogramAxes.YLabel.String = 'Frequency (Hz)';
mainWindow.SpectrogramAxes.Title.String = 'Spectrogram';

% FFT Spectrum panel
spectrumPanel = uipanel(analysisGrid, 'Title', 'FFT Spectrum');
spectrumPanel.Layout.Row = 1;
spectrumPanel.Layout.Column = 2;

mainWindow.SpectrumAxes = uiaxes(spectrumPanel);
mainWindow.SpectrumAxes.XLabel.String = 'Frequency (Hz)';
mainWindow.SpectrumAxes.YLabel.String = 'Magnitude (dB)';
mainWindow.SpectrumAxes.Title.String = 'Frequency Spectrum';
grid(mainWindow.SpectrumAxes, 'on');

% Phase correlation panel
phasePanel = uipanel(analysisGrid, 'Title', 'Phase Correlation');
phasePanel.Layout.Row = 2;
phasePanel.Layout.Column = 1;

mainWindow.PhaseAxes = uiaxes(phasePanel);
mainWindow.PhaseAxes.XLabel.String = 'Time (s)';
mainWindow.PhaseAxes.YLabel.String = 'Correlation';
mainWindow.PhaseAxes.Title.String = 'Stereo Phase Correlation';
grid(mainWindow.PhaseAxes, 'on');

% Loudness meter panel
loudnessPanel = uipanel(analysisGrid, 'Title', 'Loudness Meter');
loudnessPanel.Layout.Row = 2;
loudnessPanel.Layout.Column = 2;

loudnessGrid = uigridlayout(loudnessPanel, [4, 2]);
loudnessGrid.RowHeight = {'fit', 'fit', 'fit', '1x'};
loudnessGrid.ColumnWidth = {'fit', '1x'};
loudnessGrid.Padding = [10, 10, 10, 10];

uilabel(loudnessGrid, 'Text', 'Peak Level:', 'FontWeight', 'bold');
mainWindow.PeakLevelLabel = uilabel(loudnessGrid, 'Text', '0.0 dB');

uilabel(loudnessGrid, 'Text', 'RMS Level:', 'FontWeight', 'bold');
mainWindow.RMSLevelLabel = uilabel(loudnessGrid, 'Text', '0.0 dB');

uilabel(loudnessGrid, 'Text', 'LUFS:', 'FontWeight', 'bold');
mainWindow.LUFSLabel = uilabel(loudnessGrid, 'Text', '0.0 LUFS');

% Level meter visualization
mainWindow.LevelMeterAxes = uiaxes(loudnessGrid);
mainWindow.LevelMeterAxes.Layout.Row = 4;
mainWindow.LevelMeterAxes.Layout.Column = [1, 2];

% Analysis controls
controlPanel = uipanel(analysisGrid, 'Title', 'Analysis Controls');
controlPanel.Layout.Row = 3;
controlPanel.Layout.Column = [1, 2];

controlGrid = uigridlayout(controlPanel, [2, 4]);
controlGrid.ColumnWidth = {'1x', '1x', '1x', '1x'};
controlGrid.Padding = [5, 5, 5, 5];

uibutton(controlGrid, 'Text', 'Generate Spectrogram', ...
    'ButtonPushedFcn', @(src, event) generateSpectrogram(mainWindow));
uibutton(controlGrid, 'Text', 'Analyze Spectrum', ...
    'ButtonPushedFcn', @(src, event) analyzeSpectrum(mainWindow));
uibutton(controlGrid, 'Text', 'Analyze Phase', ...
    'ButtonPushedFcn', @(src, event) analyzePhase(mainWindow));
uibutton(controlGrid, 'Text', 'Measure Loudness', ...
    'ButtonPushedFcn', @(src, event) measureLoudness(mainWindow));

uilabel(controlGrid, 'Text', 'FFT Size:');
mainWindow.FFTSizeDropdown = uidropdown(controlGrid, ...
    'Items', {'256', '512', '1024', '2048', '4096', '8192'}, ...
    'Value', '2048');

uilabel(controlGrid, 'Text', 'Window Overlap:');
mainWindow.WindowOverlapSpinner = uispinner(controlGrid, 'Value', 50, 'Limits', [0, 90], 'Step', 10);
end

%% TAB 7: RESEARCH
function createResearchPanel(mainWindow, parent)
researchGrid = uigridlayout(parent, [4, 1]);
researchGrid.RowHeight = {'fit', 'fit', 'fit', 'fit'};
researchGrid.Padding = [10, 10, 10, 10];
researchGrid.RowSpacing = 10;

% Wavelet Analysis
waveletPanel = uipanel(researchGrid, 'Title', 'Wavelet Analysis (Wavelet Toolbox)');
waveletPanel.Layout.Row = 1;
waveletGrid = uigridlayout(waveletPanel, [2, 5]);
waveletGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit', 'fit'};
waveletGrid.Padding = [5, 5, 5, 5];

uilabel(waveletGrid, 'Text', 'Wavelet:');
mainWindow.WaveletTypeDropdown = uidropdown(waveletGrid, ...
    'Items', {'db1', 'db2', 'db4', 'db8', 'sym4', 'coif4', 'haar'}, ...
    'Value', 'db4');
uilabel(waveletGrid, 'Text', 'Levels:');
mainWindow.WaveletLevelsSpinner = uispinner(waveletGrid, 'Value', 5, 'Limits', [1, 10]);
uibutton(waveletGrid, 'Text', 'Time-Frequency', ...
    'ButtonPushedFcn', @(src, event) waveletTimeFrequency(mainWindow), ...
    'Tooltip', 'Continuous Wavelet Transform');

uibutton(waveletGrid, 'Text', 'Denoise', ...
    'ButtonPushedFcn', @(src, event) waveletDenoise(mainWindow), ...
    'Tooltip', 'Wavelet-based noise reduction');
uibutton(waveletGrid, 'Text', 'Separate Transients', ...
    'ButtonPushedFcn', @(src, event) separateTransientTonal(mainWindow), ...
    'Tooltip', 'Separate transient and tonal components');
uilabel(waveletGrid, 'Text', '');
uilabel(waveletGrid, 'Text', '');
uilabel(waveletGrid, 'Text', '');

% Feature Extraction
featurePanel = uipanel(researchGrid, 'Title', 'Feature Extraction (Audio Toolbox)');
featurePanel.Layout.Row = 2;
featureGrid = uigridlayout(featurePanel, [2, 4]);
featureGrid.ColumnWidth = {'fit', 'fit', 'fit', 'fit'};
featureGrid.Padding = [5, 5, 5, 5];

mainWindow.ExtractMFCCCheckbox = uicheckbox(featureGrid, 'Text', 'MFCC', 'Value', true);
mainWindow.ExtractSpectralCheckbox = uicheckbox(featureGrid, 'Text', 'Spectral Features', 'Value', true);
mainWindow.ExtractTemporalCheckbox = uicheckbox(featureGrid, 'Text', 'Temporal Features', 'Value', true);
uibutton(featureGrid, 'Text', 'Extract All', ...
    'ButtonPushedFcn', @(src, event) extractAllFeatures(mainWindow));

mainWindow.FeatureResultLabel = uilabel(featureGrid, 'Text', 'No features extracted');
uilabel(featureGrid, 'Text', '');
uibutton(featureGrid, 'Text', 'Export to CSV', ...
    'ButtonPushedFcn', @(src, event) exportFeatures(mainWindow));
uibutton(featureGrid, 'Text', 'Plot Features', ...
    'ButtonPushedFcn', @(src, event) plotFeatures(mainWindow));

% Anti-Aliasing & Nyquist
antiAliasingPanel = uipanel(researchGrid, 'Title', 'Anti-Aliasing & Nyquist Analysis');
antiAliasingPanel.Layout.Row = 3;
aaGrid = uigridlayout(antiAliasingPanel, [3, 4]);
aaGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit'};
aaGrid.Padding = [5, 5, 5, 5];

uilabel(aaGrid, 'Text', 'Sample Rate:');
mainWindow.AACurrentSRLabel = uilabel(aaGrid, 'Text', '44100 Hz');
uilabel(aaGrid, 'Text', 'Nyquist Freq:');
mainWindow.AANyquistLabel = uilabel(aaGrid, 'Text', '22050 Hz');

uibutton(aaGrid, 'Text', 'Check Compliance', ...
    'ButtonPushedFcn', @(src, event) checkNyquistCompliance(mainWindow));
uibutton(aaGrid, 'Text', 'Detect Aliasing', ...
    'ButtonPushedFcn', @(src, event) detectAliasing(mainWindow));
mainWindow.AAStatusLabel = uilabel(aaGrid, 'Text', 'Not analyzed');
uilabel(aaGrid, 'Text', '');

uibutton(aaGrid, 'Text', 'Apply AA Filter', ...
    'ButtonPushedFcn', @(src, event) applyAntiAliasingFilter(mainWindow), ...
    'Tooltip', 'Apply anti-aliasing low-pass filter');
uibutton(aaGrid, 'Text', 'Oversample ×2', ...
    'ButtonPushedFcn', @(src, event) oversampleAudio(mainWindow));
uibutton(aaGrid, 'Text', 'Downsample ÷2', ...
    'ButtonPushedFcn', @(src, event) downsampleAudio(mainWindow));
uibutton(aaGrid, 'Text', 'Plot Spectrum', ...
    'ButtonPushedFcn', @(src, event) plotNyquistSpectrum(mainWindow));

% Pitch & Onset Detection
pitchPanel = uipanel(researchGrid, 'Title', 'Pitch & Onset Detection');
pitchPanel.Layout.Row = 4;
pitchGrid = uigridlayout(pitchPanel, [1, 4]);
pitchGrid.ColumnWidth = {'1x', '1x', '1x', '1x'};
pitchGrid.Padding = [5, 5, 5, 5];

uibutton(pitchGrid, 'Text', 'Detect Pitch (Neural)', ...
    'ButtonPushedFcn', @(src, event) detectPitchNeural(mainWindow), ...
    'Tooltip', 'Neural network-based pitch detection');
uibutton(pitchGrid, 'Text', 'Detect Onsets', ...
    'ButtonPushedFcn', @(src, event) detectOnsets(mainWindow), ...
    'Tooltip', 'Find note/drum onsets');
uibutton(pitchGrid, 'Text', 'Measure Loudness (LUFS)', ...
    'ButtonPushedFcn', @(src, event) measureLUFS(mainWindow), ...
    'Tooltip', 'Accurate LUFS loudness measurement');
uilabel(pitchGrid, 'Text', '');
end

%% TAB 8: LIBRARY (Enhanced)
function createLibraryPanel(mainWindow, parent)
libraryGrid = uigridlayout(parent, [3, 2]);
libraryGrid.RowHeight = {'fit', '1x', 'fit'};
libraryGrid.ColumnWidth = {'1x', '1x'};
libraryGrid.Padding = [10, 10, 10, 10];
libraryGrid.RowSpacing = 10;
libraryGrid.ColumnSpacing = 10;

% Library browser panel
browserPanel = uipanel(libraryGrid, 'Title', 'Sample Library Browser');
browserPanel.Layout.Row = [1, 2];
browserPanel.Layout.Column = 1;

browserGrid = uigridlayout(browserPanel, [4, 1]);
browserGrid.RowHeight = {'fit', 'fit', '1x', 'fit'};
browserGrid.Padding = [5, 5, 5, 5];

% Category selection
categoryGrid = uigridlayout(browserGrid, [1, 2]);
categoryGrid.ColumnWidth = {'fit', '1x'};
uilabel(categoryGrid, 'Text', 'Category:');
mainWindow.CategoryDropdown = uidropdown(categoryGrid, ...
    'Items', {'All', 'Drums', 'Bass', 'Synth', 'Guitar', 'Vocals', 'User Library', 'MATLAB Sounds'}, ...
    'Value', 'All', ...
    'ValueChangedFcn', @(src, event) updateLibraryBrowser(mainWindow));

% Search box
searchGrid = uigridlayout(browserGrid, [1, 2]);
searchGrid.ColumnWidth = {'fit', '1x'};
uilabel(searchGrid, 'Text', 'Search:');
mainWindow.LibrarySearchField = uieditfield(searchGrid, ...
    'ValueChangedFcn', @(src, event) searchLibrary(mainWindow, src.Value));

% Sample list
mainWindow.SampleListBox = uilistbox(browserGrid, ...
    'Items', {'No samples loaded'}, ...
    'ValueChangedFcn', @(src, event) selectSample(mainWindow, src.Value));

% Browser controls
browserControlGrid = uigridlayout(browserGrid, [1, 3]);
browserControlGrid.ColumnWidth = {'1x', '1x', '1x'};
uibutton(browserControlGrid, 'Text', 'Load Sample', ...
    'ButtonPushedFcn', @(src, event) loadSelectedSample(mainWindow));
uibutton(browserControlGrid, 'Text', 'Preview', ...
    'ButtonPushedFcn', @(src, event) previewSample(mainWindow));
uibutton(browserControlGrid, 'Text', 'Refresh', ...
    'ButtonPushedFcn', @(src, event) refreshLibraryCatalog(mainWindow));

% Instrument Presets (NEW)
presetsPanel = uipanel(libraryGrid, 'Title', 'Instrument Effect Presets');
presetsPanel.Layout.Row = 1;
presetsPanel.Layout.Column = 2;

presetsGrid = uigridlayout(presetsPanel, [3, 1]);
presetsGrid.RowHeight = {'fit', '1x', 'fit'};
presetsGrid.Padding = [5, 5, 5, 5];

uilabel(presetsGrid, 'Text', 'Select instrument preset to load effect chain:');
mainWindow.InstrumentPresetList = uilistbox(presetsGrid, ...
    'Items', {'Vintage Keys', 'Electric Guitar', 'Acoustic Guitar', 'Bass Guitar', 'Lead Synth', 'Pad Synth', 'Vocals', 'Drums'}, ...
    'Value', 'Vintage Keys');

uibutton(presetsGrid, 'Text', 'Load Preset to Effects Tab', ...
    'ButtonPushedFcn', @(src, event) loadInstrumentPreset(mainWindow));

% Sample info panel
infoPanel = uipanel(libraryGrid, 'Title', 'Sample Information');
infoPanel.Layout.Row = 2;
infoPanel.Layout.Column = 2;

infoGrid = uigridlayout(infoPanel, [6, 2]);
infoGrid.RowHeight = repmat({'fit'}, 1, 6);
infoGrid.ColumnWidth = {'fit', '1x'};
infoGrid.Padding = [10, 10, 10, 10];
infoGrid.RowSpacing = 5;

uilabel(infoGrid, 'Text', 'Filename:', 'FontWeight', 'bold');
mainWindow.SampleFilenameLabel = uilabel(infoGrid, 'Text', '-');

uilabel(infoGrid, 'Text', 'Category:', 'FontWeight', 'bold');
mainWindow.SampleCategoryLabel = uilabel(infoGrid, 'Text', '-');

uilabel(infoGrid, 'Text', 'Duration:', 'FontWeight', 'bold');
mainWindow.SampleDurationLabel = uilabel(infoGrid, 'Text', '-');

uilabel(infoGrid, 'Text', 'Sample Rate:', 'FontWeight', 'bold');
mainWindow.SampleRateLabel = uilabel(infoGrid, 'Text', '-');

uilabel(infoGrid, 'Text', 'Channels:', 'FontWeight', 'bold');
mainWindow.SampleChannelsLabel = uilabel(infoGrid, 'Text', '-');

uilabel(infoGrid, 'Text', 'Tags:', 'FontWeight', 'bold');
mainWindow.SampleTagsLabel = uilabel(infoGrid, 'Text', '-');

% User library management
userPanel = uipanel(libraryGrid, 'Title', 'User Library Management');
userPanel.Layout.Row = 3;
userPanel.Layout.Column = [1, 2];

userGrid = uigridlayout(userPanel, [1, 4]);
userGrid.ColumnWidth = {'1x', '1x', '1x', '1x'};
userGrid.Padding = [5, 5, 5, 5];

uibutton(userGrid, 'Text', 'Add Sample to Library', ...
    'ButtonPushedFcn', @(src, event) addSampleToLibrary(mainWindow));
uibutton(userGrid, 'Text', 'Create Collection', ...
    'ButtonPushedFcn', @(src, event) createSampleCollection(mainWindow));
uibutton(userGrid, 'Text', 'Import Collection', ...
    'ButtonPushedFcn', @(src, event) importSampleCollection(mainWindow));
uibutton(userGrid, 'Text', 'Export Collection', ...
    'ButtonPushedFcn', @(src, event) exportSampleCollection(mainWindow));

% Initialize library browser
updateLibraryBrowser(mainWindow);
end

%% TAB 9: SETTINGS
function createSettingsPanel(mainWindow, parent)
settingsGrid = uigridlayout(parent, [5, 1]);
settingsGrid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit'};
settingsGrid.Padding = [10, 10, 10, 10];
settingsGrid.RowSpacing = 10;

% Audio Settings
audioPanel = uipanel(settingsGrid, 'Title', 'Audio Settings');
audioPanel.Layout.Row = 1;
audioGrid = uigridlayout(audioPanel, [4, 2]);
audioGrid.ColumnWidth = {'fit', '1x'};
audioGrid.Padding = [5, 5, 5, 5];

uilabel(audioGrid, 'Text', 'Default Sample Rate:');
mainWindow.DefaultSRDropdown = uidropdown(audioGrid, ...
    'Items', {'44100', '48000', '88200', '96000'}, 'Value', '44100');

uilabel(audioGrid, 'Text', 'Bit Depth:');
mainWindow.BitDepthDropdown = uidropdown(audioGrid, ...
    'Items', {'16', '24', '32'}, 'Value', '24');

uilabel(audioGrid, 'Text', 'Buffer Size:');
mainWindow.BufferSizeDropdown = uidropdown(audioGrid, ...
    'Items', {'128', '256', '512', '1024', '2048'}, 'Value', '512');

mainWindow.AutoNormalizeCheckbox = uicheckbox(audioGrid, 'Text', 'Auto-normalize on load', 'Value', false);
uilabel(audioGrid, 'Text', '');

% Processing Settings
processingPanel = uipanel(settingsGrid, 'Title', 'Processing Settings');
processingPanel.Layout.Row = 2;
procGrid = uigridlayout(processingPanel, [3, 2]);
procGrid.ColumnWidth = {'fit', '1x'};
procGrid.Padding = [5, 5, 5, 5];

uilabel(procGrid, 'Text', 'Undo History Levels:');
mainWindow.UndoLevelsSpinner = uispinner(procGrid, 'Value', 50, 'Limits', [10, 100]);

mainWindow.EnableGPUCheckbox = uicheckbox(procGrid, 'Text', 'Enable GPU Acceleration', 'Value', false);
uilabel(procGrid, 'Text', '');

mainWindow.ParallelProcCheckbox = uicheckbox(procGrid, 'Text', 'Use Parallel Processing', 'Value', true);
uilabel(procGrid, 'Text', '');

% Display Settings
displayPanel = uipanel(settingsGrid, 'Title', 'Display Settings');
displayPanel.Layout.Row = 3;
displayGrid = uigridlayout(displayPanel, [4, 2]);
displayGrid.ColumnWidth = {'fit', '1x'};
displayGrid.Padding = [5, 5, 5, 5];

uilabel(displayGrid, 'Text', 'Theme:');
mainWindow.ThemeDropdown = uidropdown(displayGrid, ...
    'Items', {'Light', 'Dark', 'Auto'}, 'Value', 'Light');

uilabel(displayGrid, 'Text', 'Waveform Color:');
mainWindow.WaveformColorDropdown = uidropdown(displayGrid, ...
    'Items', {'Blue', 'Green', 'Red', 'Purple', 'Orange'}, 'Value', 'Blue');

mainWindow.ShowGridCheckbox = uicheckbox(displayGrid, 'Text', 'Show Grid', 'Value', true);
uilabel(displayGrid, 'Text', '');

mainWindow.ShowMarkersCheckbox = uicheckbox(displayGrid, 'Text', 'Show Timeline Markers', 'Value', true);
uilabel(displayGrid, 'Text', '');

% File Paths
pathPanel = uipanel(settingsGrid, 'Title', 'File Paths');
pathPanel.Layout.Row = 4;
pathGrid = uigridlayout(pathPanel, [3, 3]);
pathGrid.ColumnWidth = {'fit', '1x', 'fit'};
pathGrid.Padding = [5, 5, 5, 5];

uilabel(pathGrid, 'Text', 'User Library:');
mainWindow.UserLibraryPathField = uieditfield(pathGrid, 'Value', pwd);
uibutton(pathGrid, 'Text', 'Browse', 'ButtonPushedFcn', @(src, event) browseUserLibrary(mainWindow));

uilabel(pathGrid, 'Text', 'Impulse Responses:');
mainWindow.IRPathField = uieditfield(pathGrid, 'Value', pwd);
uibutton(pathGrid, 'Text', 'Browse', 'ButtonPushedFcn', @(src, event) browseIRPath(mainWindow));

uilabel(pathGrid, 'Text', 'Export Default:');
mainWindow.ExportPathField = uieditfield(pathGrid, 'Value', pwd);
uibutton(pathGrid, 'Text', 'Browse', 'ButtonPushedFcn', @(src, event) browseExportPath(mainWindow));

% Apply/Reset
actionPanel = uipanel(settingsGrid);
actionPanel.Layout.Row = 5;
actionGrid = uigridlayout(actionPanel, [1, 4]);
actionGrid.ColumnWidth = {'1x', 'fit', 'fit', 'fit'};
actionGrid.Padding = [5, 5, 5, 5];

uilabel(actionGrid, 'Text', '');
uibutton(actionGrid, 'Text', 'Reset to Defaults', ...
    'ButtonPushedFcn', @(src, event) resetSettings(mainWindow));
uibutton(actionGrid, 'Text', 'Apply', ...
    'ButtonPushedFcn', @(src, event) applySettings(mainWindow));
uibutton(actionGrid, 'Text', 'Save', ...
    'ButtonPushedFcn', @(src, event) saveSettings(mainWindow));
end
