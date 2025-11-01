function mainWindow = MainWindow()
%MAINWINDOW Main application window for Audio Signal Processor
%
%   MAINWINDOW = MAINWINDOW() creates the main application window with
%   all GUI components and functionality.
%
%   Properties:
%   ----------
%   Figure - Main figure handle
%   TabGroup - Tab group for different panels
%   StatusBar - Status bar with playback controls
%   WaveformDisplay - Waveform visualization area
%   TransportControls - Playback controls
%   LibraryManager - Sound library manager
%   Mixer - Multi-track mixer
%   EffectsLibrary - Effects library
%
%   Methods:
%   --------
%   show() - Show the main window
%   hide() - Hide the main window
%   close() - Close the application
%   loadAudio() - Load audio file
%   play() - Start playback
%   pause() - Pause playback
%   stop() - Stop playback
%
%   Example:
%   --------
%   % Create and show main window
%   mainWindow = MainWindow();
%   mainWindow.show();
%
%   See also: FilterPanel, MixerPanel, AnalysisPanel, LibraryBrowserPanel

% Create main figure
mainWindow = struct();
mainWindow.Figure = uifigure('Name', 'Audio Signal Processor', ...
    'Position', [100, 100, 1200, 800]);

% Initialize components
mainWindow.LibraryManager = SoundLibraryManager();
mainWindow.Mixer = MixerCore(8, 44100);
mainWindow.EffectsLibrary = InstrumentEffectsLibrary();

% Create main layout (must be first as it creates MainGrid)
mainWindow.MainGrid = uigridlayout(mainWindow.Figure, [3, 1]);
mainWindow.MainGrid.RowHeight = {'fit', '1x', 'fit'};
mainWindow.MainGrid.ColumnWidth = {'1x'};
mainWindow.MainGrid.Padding = [5, 5, 5, 5];
mainWindow.MainGrid.RowSpacing = 5;
mainWindow.MainGrid.ColumnSpacing = 5;

% Create menu bar
createMenuBar(mainWindow);

% Create status bar
createStatusBar(mainWindow);

% Create tab group with panels
createTabGroup(mainWindow);

% Initialize application state
mainWindow.IsPlaying = false;
mainWindow.CurrentFile = '';
mainWindow.LoadedAudio = [];
mainWindow.SampleRate = 44100;

% Add methods
mainWindow.show = @() show(mainWindow);
mainWindow.hide = @() hide(mainWindow);
mainWindow.close = @() close(mainWindow);
mainWindow.loadAudio = @(filename) loadAudio(mainWindow, filename);
mainWindow.play = @() play(mainWindow);
mainWindow.pause = @() pause(mainWindow);
mainWindow.stop = @() stop(mainWindow);

% Set close request function - use simpler callback that just deletes the figure
mainWindow.Figure.CloseRequestFcn = @(src, event) delete(src);
end

function createMenuBar(mainWindow)
% Create menu bar

% File menu
fileMenu = uimenu(mainWindow.Figure, 'Text', 'File');
uimenu(fileMenu, 'Text', 'Load Audio...', 'MenuSelectedFcn', @(src, event) loadAudioDialog(mainWindow));
uimenu(fileMenu, 'Text', 'Load from Library...', 'MenuSelectedFcn', @(src, event) loadFromLibraryDialog(mainWindow));
uimenu(fileMenu, 'Separator', 'on');
uimenu(fileMenu, 'Text', 'Export Audio...', 'MenuSelectedFcn', @(src, event) exportAudioDialog(mainWindow));
uimenu(fileMenu, 'Separator', 'on');
uimenu(fileMenu, 'Text', 'Exit', 'MenuSelectedFcn', @(src, event) close(mainWindow));

% Edit menu
editMenu = uimenu(mainWindow.Figure, 'Text', 'Edit');
uimenu(editMenu, 'Text', 'Undo', 'Enable', 'off');
uimenu(editMenu, 'Text', 'Redo', 'Enable', 'off');
uimenu(editMenu, 'Separator', 'on');
uimenu(editMenu, 'Text', 'Preferences...', 'MenuSelectedFcn', @(src, event) showPreferences(mainWindow));

% View menu
viewMenu = uimenu(mainWindow.Figure, 'Text', 'View');
uimenu(viewMenu, 'Text', 'Zoom In', 'MenuSelectedFcn', @(src, event) zoomIn(mainWindow));
uimenu(viewMenu, 'Text', 'Zoom Out', 'MenuSelectedFcn', @(src, event) zoomOut(mainWindow));
uimenu(viewMenu, 'Text', 'Fit to Window', 'MenuSelectedFcn', @(src, event) fitToWindow(mainWindow));

% Tools menu
toolsMenu = uimenu(mainWindow.Figure, 'Text', 'Tools');
uimenu(toolsMenu, 'Text', 'Batch Process...', 'MenuSelectedFcn', @(src, event) showBatchProcessor(mainWindow));
uimenu(toolsMenu, 'Text', 'Audio Analysis...', 'MenuSelectedFcn', @(src, event) showAudioAnalysis(mainWindow));

% Help menu
helpMenu = uimenu(mainWindow.Figure, 'Text', 'Help');
uimenu(helpMenu, 'Text', 'User Guide', 'MenuSelectedFcn', @(src, event) showUserGuide(mainWindow));
uimenu(helpMenu, 'Text', 'About', 'MenuSelectedFcn', @(src, event) showAbout(mainWindow));
end

function createStatusBar(mainWindow)
% Create status bar with playback controls

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
    'ButtonPushedFcn', @(src, event) play(mainWindow));
mainWindow.PauseButton = uibutton(transportGrid, 'Text', '⏸', ...
    'ButtonPushedFcn', @(src, event) pause(mainWindow));
mainWindow.StopButton = uibutton(transportGrid, 'Text', '⏹', ...
    'ButtonPushedFcn', @(src, event) stop(mainWindow));

% Status text
mainWindow.StatusText = uilabel(statusGrid, 'Text', 'Ready');
mainWindow.StatusText.Layout.Column = 2;

% Time display
mainWindow.TimeDisplay = uilabel(statusGrid, 'Text', '00:00 / 00:00');
mainWindow.TimeDisplay.Layout.Column = 3;

% Volume control
volumePanel = uipanel(statusGrid);
volumePanel.Layout.Column = 4;

volumeGrid = uigridlayout(volumePanel, [1, 2]);
volumeGrid.ColumnWidth = {'fit', 'fit'};
volumeGrid.Padding = [2, 2, 2, 2];

uilabel(volumeGrid, 'Text', 'Vol:');
mainWindow.VolumeSlider = uislider(volumeGrid, 'Value', 0.7, ...
    'Limits', [0, 1], ...
    'ValueChangedFcn', @(src, event) updateVolume(mainWindow, src.Value));
end

function createTabGroup(mainWindow)
% Create tab group with different panels

mainWindow.TabGroup = uitabgroup(mainWindow.MainGrid);
mainWindow.TabGroup.Layout.Row = 2;
mainWindow.TabGroup.Layout.Column = 1;

% Waveform tab
waveformTab = uitab(mainWindow.TabGroup, 'Title', 'Waveform');
createWaveformPanel(mainWindow, waveformTab);

% Filters tab
filtersTab = uitab(mainWindow.TabGroup, 'Title', 'Filters');
createFiltersPanel(mainWindow, filtersTab);

% Mixer tab
mixerTab = uitab(mainWindow.TabGroup, 'Title', 'Mixer');
createMixerPanel(mainWindow, mixerTab);

% Analysis tab
analysisTab = uitab(mainWindow.TabGroup, 'Title', 'Analysis');
createAnalysisPanel(mainWindow, analysisTab);

% Library tab
libraryTab = uitab(mainWindow.TabGroup, 'Title', 'Library');
createLibraryPanel(mainWindow, libraryTab);
end

function createWaveformPanel(mainWindow, parent)
% Create waveform display panel

waveformGrid = uigridlayout(parent, [2, 1]);
waveformGrid.RowHeight = {'1x', 'fit'};
waveformGrid.Padding = [5, 5, 5, 5];

% Waveform display area
waveformPanel = uipanel(waveformGrid);
waveformPanel.Layout.Row = 1;

% Create grid layout inside panel for proper axes sizing
axesGrid = uigridlayout(waveformPanel, [1, 1]);
axesGrid.Padding = [10, 10, 10, 10];

% Create axes for waveform display
mainWindow.WaveformAxes = uiaxes(axesGrid);
mainWindow.WaveformAxes.XLabel.String = 'Time (s)';
mainWindow.WaveformAxes.YLabel.String = 'Amplitude';
mainWindow.WaveformAxes.Title.String = 'Audio Waveform';
grid(mainWindow.WaveformAxes, 'on');

% Waveform controls
controlsPanel = uipanel(waveformGrid);
controlsPanel.Layout.Row = 2;

controlsGrid = uigridlayout(controlsPanel, [1, 4]);
controlsGrid.ColumnWidth = {'fit', 'fit', 'fit', '1x'};
controlsGrid.Padding = [5, 5, 5, 5];

uibutton(controlsGrid, 'Text', 'Load Audio', ...
    'ButtonPushedFcn', @(src, event) loadAudioDialog(mainWindow));

uibutton(controlsGrid, 'Text', 'Zoom In', ...
    'ButtonPushedFcn', @(src, event) zoomIn(mainWindow));

uibutton(controlsGrid, 'Text', 'Zoom Out', ...
    'ButtonPushedFcn', @(src, event) zoomOut(mainWindow));

% Placeholder for additional controls
uilabel(controlsGrid, 'Text', '');
end

function createFiltersPanel(mainWindow, parent)
% Create filters panel with filter controls

filtersGrid = uigridlayout(parent, [4, 2]);
filtersGrid.RowHeight = {'fit', 'fit', 'fit', '2x'};
filtersGrid.ColumnWidth = {'1x', '1x'};
filtersGrid.Padding = [10, 10, 10, 10];
filtersGrid.RowSpacing = 8;
filtersGrid.ColumnSpacing = 10;

% Filter Type Selection - Compact horizontal layout
filterTypePanel = uipanel(filtersGrid, 'Title', 'Filter Type');
filterTypePanel.Layout.Row = 1;
filterTypePanel.Layout.Column = [1, 2];

filterTypeGrid = uigridlayout(filterTypePanel, [1, 7]);
filterTypeGrid.ColumnWidth = repmat({'fit'}, 1, 7);
filterTypeGrid.Padding = [5, 5, 5, 5];
filterTypeGrid.ColumnSpacing = 8;

mainWindow.FilterTypeGroup = uibuttongroup(filterTypeGrid);
mainWindow.FilterTypeGroup.Layout.Row = 1;
mainWindow.FilterTypeGroup.Layout.Column = [1, 7];
mainWindow.FilterTypeGroup.BorderType = 'none';

uiradiobutton(mainWindow.FilterTypeGroup, 'Text', 'Low Pass', 'Position', [5, 5, 85, 22], 'Value', true);
uiradiobutton(mainWindow.FilterTypeGroup, 'Text', 'High Pass', 'Position', [95, 5, 85, 22]);
uiradiobutton(mainWindow.FilterTypeGroup, 'Text', 'Band Pass', 'Position', [185, 5, 85, 22]);
uiradiobutton(mainWindow.FilterTypeGroup, 'Text', 'Band Stop', 'Position', [275, 5, 85, 22]);
uiradiobutton(mainWindow.FilterTypeGroup, 'Text', 'Butterworth', 'Position', [365, 5, 95, 22]);
uiradiobutton(mainWindow.FilterTypeGroup, 'Text', 'Moving Avg', 'Position', [465, 5, 95, 22]);
uiradiobutton(mainWindow.FilterTypeGroup, 'Text', 'Median', 'Position', [565, 5, 75, 22]);

% FFT Filter Parameters
fftPanel = uipanel(filtersGrid, 'Title', 'FFT Filter Parameters');
fftPanel.Layout.Row = 2;
fftPanel.Layout.Column = 1;

fftGrid = uigridlayout(fftPanel, [4, 2]);
fftGrid.ColumnWidth = {'fit', '1x'};
fftGrid.Padding = [5, 5, 5, 5];
fftGrid.RowSpacing = 5;

uilabel(fftGrid, 'Text', 'Cutoff Freq (Hz):');
mainWindow.CutoffFreqSpinner = uispinner(fftGrid, 'Value', 1000, 'Limits', [20, 20000]);

uilabel(fftGrid, 'Text', 'Transition BW (Hz):');
mainWindow.TransitionBWSpinner = uispinner(fftGrid, 'Value', 100, 'Limits', [10, 5000]);

uilabel(fftGrid, 'Text', 'Window Type:');
mainWindow.WindowTypeDropdown = uidropdown(fftGrid, ...
    'Items', {'Gaussian', 'Rectangular', 'Hamming', 'Hann', 'Blackman', 'Kaiser', 'Tukey', 'Bartlett'}, ...
    'Value', 'Gaussian');

uilabel(fftGrid, 'Text', 'Zero Phase:');
mainWindow.ZeroPhaseCheckbox = uicheckbox(fftGrid, 'Text', '', 'Value', true);

% Time-Domain Filter Parameters
timePanel = uipanel(filtersGrid, 'Title', 'Time-Domain Parameters');
timePanel.Layout.Row = 2;
timePanel.Layout.Column = 2;

timeGrid = uigridlayout(timePanel, [3, 2]);
timeGrid.ColumnWidth = {'fit', '1x'};
timeGrid.Padding = [5, 5, 5, 5];
timeGrid.RowSpacing = 5;

uilabel(timeGrid, 'Text', 'Filter Order:');
mainWindow.FilterOrderSpinner = uispinner(timeGrid, 'Value', 4, 'Limits', [1, 10]);

uilabel(timeGrid, 'Text', 'Window Size:');
mainWindow.WindowSizeSpinner = uispinner(timeGrid, 'Value', 5, 'Limits', [3, 101], 'Step', 2);

uilabel(timeGrid, 'Text', 'Passband Ripple:');
mainWindow.PassbandRippleSpinner = uispinner(timeGrid, 'Value', 1, 'Limits', [0.1, 10], 'Step', 0.1);

% Filter Controls
controlPanel = uipanel(filtersGrid, 'Title', 'Filter Controls');
controlPanel.Layout.Row = 3;
controlPanel.Layout.Column = [1, 2];

controlGrid = uigridlayout(controlPanel, [1, 3]);
controlGrid.ColumnWidth = {'1x', '1x', '1x'};
controlGrid.Padding = [5, 5, 5, 5];

uibutton(controlGrid, 'Text', 'Apply Filter', ...
    'ButtonPushedFcn', @(src, event) applyFilter(mainWindow));
uibutton(controlGrid, 'Text', 'Preview Response', ...
    'ButtonPushedFcn', @(src, event) previewFilterResponse(mainWindow));
uibutton(controlGrid, 'Text', 'Reset', ...
    'ButtonPushedFcn', @(src, event) resetFilter(mainWindow));

% Filter Response Display
responsePanel = uipanel(filtersGrid, 'Title', 'Filter Response');
responsePanel.Layout.Row = 4;
responsePanel.Layout.Column = [1, 2];

mainWindow.FilterResponseAxes = uiaxes(responsePanel);
mainWindow.FilterResponseAxes.XLabel.String = 'Frequency (Hz)';
mainWindow.FilterResponseAxes.YLabel.String = 'Magnitude (dB)';
mainWindow.FilterResponseAxes.Title.String = 'Frequency Response';
grid(mainWindow.FilterResponseAxes, 'on');
end

function createMixerPanel(mainWindow, parent)
% Create mixer panel with multi-track controls

mixerGrid = uigridlayout(parent, [2, 1]);
mixerGrid.RowHeight = {'1x', 'fit'};
mixerGrid.Padding = [5, 5, 5, 5];

% Tracks panel
tracksPanel = uipanel(mixerGrid, 'Title', 'Tracks');
tracksPanel.Layout.Row = 1;

tracksScroll = uigridlayout(tracksPanel, [1, 8]);
tracksScroll.ColumnWidth = repmat({'fit'}, 1, 8);
tracksScroll.Padding = [5, 5, 5, 5];
tracksScroll.ColumnSpacing = 10;

% Create 8 track strips
mainWindow.TrackStrips = cell(8, 1);
for i = 1:8
    trackStrip = uipanel(tracksScroll);
    trackStrip.Title = sprintf('Track %d', i);

    trackGrid = uigridlayout(trackStrip, [8, 1]);
    trackGrid.RowHeight = {'fit', 'fit', '1x', 'fit', 'fit', 'fit', 'fit', 'fit'};
    trackGrid.ColumnWidth = {80};
    trackGrid.Padding = [5, 5, 5, 5];
    trackGrid.RowSpacing = 5;

    % Track name
    trackName = uilabel(trackGrid, 'Text', sprintf('Track %d', i), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');

    % Load button
    uibutton(trackGrid, 'Text', 'Load', ...
        'ButtonPushedFcn', @(src, event) loadTrackAudio(mainWindow, i));

    % Volume fader
    volumeSlider = uislider(trackGrid, 'Orientation', 'vertical', ...
        'Value', 0.8, 'Limits', [0, 1], ...
        'ValueChangedFcn', @(src, event) setTrackVolume(mainWindow.Mixer, i, src.Value));

    % Volume label
    uilabel(trackGrid, 'Text', 'Volume', 'HorizontalAlignment', 'center');

    % Pan knob
    panKnob = uiknob(trackGrid, 'Value', 0, 'Limits', [-1, 1], ...
        'ValueChangedFcn', @(src, event) setTrackPan(mainWindow.Mixer, i, src.Value));

    % Solo button
    uibutton(trackGrid, 'Text', 'S', ...
        'ButtonPushedFcn', @(src, event) toggleTrackSolo(mainWindow, i, src));

    % Mute button
    uibutton(trackGrid, 'Text', 'M', ...
        'ButtonPushedFcn', @(src, event) toggleTrackMute(mainWindow, i, src));

    % Effects button
    uibutton(trackGrid, 'Text', 'FX', ...
        'ButtonPushedFcn', @(src, event) showTrackEffects(mainWindow, i));

    mainWindow.TrackStrips{i} = struct('Panel', trackStrip, 'Volume', volumeSlider, 'Pan', panKnob);
end

% Master section
masterPanel = uipanel(mixerGrid, 'Title', 'Master');
masterPanel.Layout.Row = 2;

masterGrid = uigridlayout(masterPanel, [1, 6]);
masterGrid.ColumnWidth = {'fit', '1x', 'fit', 'fit', 'fit', 'fit'};
masterGrid.Padding = [5, 5, 5, 5];

uilabel(masterGrid, 'Text', 'Master Volume:', 'FontWeight', 'bold');
mainWindow.MasterVolumeSlider = uislider(masterGrid, 'Value', 0.8, 'Limits', [0, 1], ...
    'ValueChangedFcn', @(src, event) updateVolume(mainWindow, src.Value));

uibutton(masterGrid, 'Text', 'Process Mix', ...
    'ButtonPushedFcn', @(src, event) processMix(mainWindow));
uibutton(masterGrid, 'Text', 'Clear All', ...
    'ButtonPushedFcn', @(src, event) clearAllTracks(mainWindow));
uibutton(masterGrid, 'Text', 'Export Mix', ...
    'ButtonPushedFcn', @(src, event) exportMix(mainWindow));
uibutton(masterGrid, 'Text', 'Export Stems', ...
    'ButtonPushedFcn', @(src, event) exportStems(mainWindow));
end

function createAnalysisPanel(mainWindow, parent)
% Create analysis panel with visualization tools

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

function createLibraryPanel(mainWindow, parent)
% Create library panel with sample browser

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

% MATLAB Sounds panel
matlabSoundsPanel = uipanel(libraryGrid, 'Title', 'MATLAB Built-in Sounds');
matlabSoundsPanel.Layout.Row = 1;
matlabSoundsPanel.Layout.Column = 2;

matlabGrid = uigridlayout(matlabSoundsPanel, [2, 1]);
matlabGrid.RowHeight = {'1x', 'fit'};
matlabGrid.Padding = [5, 5, 5, 5];

mainWindow.MATLABSoundsListBox = uilistbox(matlabGrid, ...
    'Items', fieldnames(mainWindow.LibraryManager.MATLABSounds));

uibutton(matlabGrid, 'Text', 'Load MATLAB Sound', ...
    'ButtonPushedFcn', @(src, event) loadMATLABSound(mainWindow));

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

function show(mainWindow)
% Show the main window

mainWindow.Figure.Visible = 'on';
end

function hide(mainWindow)
% Hide the main window

mainWindow.Figure.Visible = 'off';
end

function close(mainWindow)
% Close the application

delete(mainWindow.Figure);
end

function loadAudio(mainWindow, filename)
% Load audio file

try
    [audioData, sampleRate, info] = AudioLoader(filename);

    mainWindow.LoadedAudio = audioData;
    mainWindow.SampleRate = sampleRate;
    mainWindow.CurrentFile = filename;

    % Update waveform display
    updateWaveformDisplay(mainWindow);

    % Update status
    mainWindow.StatusText.Text = sprintf('Loaded: %s', filename);

catch ME
    uialert(mainWindow.Figure, sprintf('Error loading audio: %s', ME.message), 'Load Error');
end
end

function play(mainWindow)
% Start playback

if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Playback Error');
    return;
end

mainWindow.IsPlaying = true;
mainWindow.PlayButton.Enable = 'off';
mainWindow.PauseButton.Enable = 'on';
mainWindow.StopButton.Enable = 'on';

mainWindow.StatusText.Text = 'Playing...';
end

function pause(mainWindow)
% Pause playback

mainWindow.IsPlaying = false;
mainWindow.PlayButton.Enable = 'on';
mainWindow.PauseButton.Enable = 'off';

mainWindow.StatusText.Text = 'Paused';
end

function stop(mainWindow)
% Stop playback

mainWindow.IsPlaying = false;
mainWindow.PlayButton.Enable = 'on';
mainWindow.PauseButton.Enable = 'off';
mainWindow.StopButton.Enable = 'off';

mainWindow.StatusText.Text = 'Stopped';
end

function updateWaveformDisplay(mainWindow)
% Update waveform display

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
end

function updateVolume(mainWindow, volume)
% Update master volume

mainWindow.Mixer.MasterBus.Volume = volume;
end

% Dialog functions (placeholders)
function loadAudioDialog(mainWindow)
[filename, pathname] = uigetfile({'*.wav;*.mp3;*.flac;*.ogg;*.m4a', 'Audio Files'}, 'Load Audio');
if filename ~= 0
    loadAudio(mainWindow, fullfile(pathname, filename));
end
end

function loadFromLibraryDialog(mainWindow)
uialert(mainWindow.Figure, 'Library browser coming soon', 'Info');
end

function exportAudioDialog(mainWindow)
uialert(mainWindow.Figure, 'Export dialog coming soon', 'Info');
end

function showPreferences(mainWindow)
uialert(mainWindow.Figure, 'Preferences coming soon', 'Info');
end

function zoomIn(mainWindow)
if ~isempty(mainWindow.LoadedAudio)
    xlim(mainWindow.WaveformAxes, xlim(mainWindow.WaveformAxes) * 0.8);
end
end

function zoomOut(mainWindow)
if ~isempty(mainWindow.LoadedAudio)
    xlim(mainWindow.WaveformAxes, xlim(mainWindow.WaveformAxes) * 1.25);
end
end

function fitToWindow(mainWindow)
if ~isempty(mainWindow.LoadedAudio)
    audioData = mainWindow.LoadedAudio;
    sampleRate = mainWindow.SampleRate;
    time = (0:size(audioData, 1)-1) / sampleRate;
    xlim(mainWindow.WaveformAxes, [min(time), max(time)]);
end
end

function showBatchProcessor(mainWindow)
uialert(mainWindow.Figure, 'Batch processor coming soon', 'Info');
end

function showAudioAnalysis(mainWindow)
uialert(mainWindow.Figure, 'Audio analysis coming soon', 'Info');
end

function showUserGuide(mainWindow)
uialert(mainWindow.Figure, 'User guide coming soon', 'Info');
end

function showAbout(mainWindow)
uialert(mainWindow.Figure, 'Audio Signal Processor v1.0\nMATLAB Audio Processing Suite', 'About');
end

function closeApp(src, event, mainWindow)
% Handle application close request

try
    % Stop playback if active
    if isfield(mainWindow, 'IsPlaying') && mainWindow.IsPlaying
        stop(mainWindow);
    end
catch
    % Ignore errors during cleanup
end

% Delete the figure
delete(src);
end

% ========== Filter Panel Callbacks ==========

function applyFilter(mainWindow)
% Apply selected filter to loaded audio

if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'Please load audio first', 'No Audio');
    return;
end

try
    filterType = mainWindow.FilterTypeGroup.SelectedObject.Text;
    audioData = mainWindow.LoadedAudio;
    sampleRate = mainWindow.SampleRate;

    switch filterType
        case {'Low Pass', 'High Pass', 'Band Pass', 'Band Stop'}
            % FFT-based filter
            cutoffFreq = mainWindow.CutoffFreqSpinner.Value;
            transitionBW = mainWindow.TransitionBWSpinner.Value;
            windowType = mainWindow.WindowTypeDropdown.Value;
            zeroPhase = mainWindow.ZeroPhaseCheckbox.Value;

            filtered = FFTFilters(audioData, filterType, ...
                'CutoffFrequency', cutoffFreq, ...
                'TransitionBandwidth', transitionBW, ...
                'WindowType', windowType, ...
                'ZeroPhase', zeroPhase, ...
                'SampleRate', sampleRate);

        case 'Butterworth'
            % Time-domain Butterworth filter
            cutoffFreq = mainWindow.CutoffFreqSpinner.Value;
            filterOrder = mainWindow.FilterOrderSpinner.Value;

            filtered = AudioFilterEngine(audioData, 'Butterworth', ...
                'CutoffFrequency', cutoffFreq, ...
                'FilterOrder', filterOrder, ...
                'SampleRate', sampleRate);

        case 'Moving Avg'
            % Moving average filter
            windowSize = mainWindow.WindowSizeSpinner.Value;
            filtered = AudioFilterEngine(audioData, 'MovingAverage', ...
                'WindowSize', windowSize);

        case 'Median'
            % Median filter
            windowSize = mainWindow.WindowSizeSpinner.Value;
            filtered = AudioFilterEngine(audioData, 'Median', ...
                'WindowSize', windowSize);
    end

    mainWindow.LoadedAudio = filtered;
    updateWaveformDisplay(mainWindow);
    uialert(mainWindow.Figure, 'Filter applied successfully', 'Success');

catch ME
    uialert(mainWindow.Figure, ['Error applying filter: ' ME.message], 'Error');
end
end

function previewFilterResponse(mainWindow)
% Preview filter frequency response

try
    filterType = mainWindow.FilterTypeGroup.SelectedObject.Text;
    cutoffFreq = mainWindow.CutoffFreqSpinner.Value;
    sampleRate = mainWindow.SampleRate;

    % Generate filter response
    freqs = linspace(0, sampleRate/2, 1000);
    response = zeros(size(freqs));

    for i = 1:length(freqs)
        if strcmp(filterType, 'Low Pass')
            response(i) = freqs(i) < cutoffFreq;
        elseif strcmp(filterType, 'High Pass')
            response(i) = freqs(i) > cutoffFreq;
        end
    end

    % Plot on filter response axes
    plot(mainWindow.FilterResponseAxes, freqs, 20*log10(response + eps));
    title(mainWindow.FilterResponseAxes, sprintf('%s Filter Response', filterType));

catch ME
    uialert(mainWindow.Figure, ['Error previewing filter: ' ME.message], 'Error');
end
end

function resetFilter(mainWindow)
% Reset filter parameters to defaults

mainWindow.CutoffFreqSpinner.Value = 1000;
mainWindow.TransitionBWSpinner.Value = 100;
mainWindow.WindowTypeDropdown.Value = 'Gaussian';
mainWindow.ZeroPhaseCheckbox.Value = true;
mainWindow.FilterOrderSpinner.Value = 4;
mainWindow.WindowSizeSpinner.Value = 5;
end

% ========== Mixer Panel Callbacks ==========

function loadTrackAudio(mainWindow, trackIndex)
% Load audio file into specific track

[file, path] = uigetfile({'*.wav;*.mp3;*.flac', 'Audio Files'}, 'Select Audio File');
if file == 0
    return;
end

try
    [audioData, fs] = AudioLoader(fullfile(path, file));
    mainWindow.Mixer.loadTrack(trackIndex, audioData, fs);
    uialert(mainWindow.Figure, sprintf('Track %d loaded successfully', trackIndex), 'Success');
catch ME
    uialert(mainWindow.Figure, ['Error loading track: ' ME.message], 'Error');
end
end

function toggleTrackSolo(mainWindow, trackIndex, button)
% Toggle track solo state

currentState = mainWindow.Mixer.Tracks(trackIndex).Solo;
mainWindow.Mixer.setTrackSolo(trackIndex, ~currentState);

if ~currentState
    button.BackgroundColor = [1, 0.8, 0];
else
    button.BackgroundColor = [0.96, 0.96, 0.96];
end
end

function toggleTrackMute(mainWindow, trackIndex, button)
% Toggle track mute state

currentState = mainWindow.Mixer.Tracks(trackIndex).Mute;
mainWindow.Mixer.setTrackMute(trackIndex, ~currentState);

if ~currentState
    button.BackgroundColor = [1, 0.4, 0.4];
else
    button.BackgroundColor = [0.96, 0.96, 0.96];
end
end

function showTrackEffects(mainWindow, trackIndex)
% Show effects dialog for track

uialert(mainWindow.Figure, sprintf('Effects editor for Track %d coming soon', trackIndex), 'Info');
end

function processMix(mainWindow)
% Process and mix all tracks

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

function clearAllTracks(mainWindow)
% Clear all tracks

for i = 1:mainWindow.Mixer.NumTracks
    mainWindow.Mixer.Tracks(i).AudioData = [];
    mainWindow.Mixer.Tracks(i).IsLoaded = false;
end
uialert(mainWindow.Figure, 'All tracks cleared', 'Success');
end

function exportMix(mainWindow)
% Export mixed audio

if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio to export', 'Warning');
    return;
end

[file, path] = uiputfile({'*.wav', 'WAV File'}, 'Export Mixed Audio');
if file == 0
    return;
end

try
    AudioExporter(mainWindow.LoadedAudio, fullfile(path, file), ...
        'SampleRate', mainWindow.SampleRate, 'BitDepth', 24);
    uialert(mainWindow.Figure, 'Mix exported successfully', 'Success');
catch ME
    uialert(mainWindow.Figure, ['Error exporting: ' ME.message], 'Error');
end
end

function exportStems(mainWindow)
% Export individual track stems

uialert(mainWindow.Figure, 'Stem export coming soon', 'Info');
end

% ========== Analysis Panel Callbacks ==========

function generateSpectrogram(mainWindow)
% Generate and display spectrogram

if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'Please load audio first', 'No Audio');
    return;
end

try
    fftSize = str2double(mainWindow.FFTSizeDropdown.Value);
    overlap = mainWindow.WindowOverlapSpinner.Value / 100;

    [S, F, T] = SpectrogramGenerator(mainWindow.LoadedAudio, ...
        'SampleRate', mainWindow.SampleRate, ...
        'FFTSize', fftSize, ...
        'Overlap', overlap);

    imagesc(mainWindow.SpectrogramAxes, T, F, 10*log10(abs(S)));
    axis(mainWindow.SpectrogramAxes, 'xy');
    colormap(mainWindow.SpectrogramAxes, 'jet');
    colorbar(mainWindow.SpectrogramAxes);

catch ME
    uialert(mainWindow.Figure, ['Error generating spectrogram: ' ME.message], 'Error');
end
end

function analyzeSpectrum(mainWindow)
% Analyze and display FFT spectrum

if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'Please load audio first', 'No Audio');
    return;
end

try
    fftSize = str2double(mainWindow.FFTSizeDropdown.Value);
    [freqs, magnitudes] = FrequencyAnalyzer(mainWindow.LoadedAudio, ...
        'SampleRate', mainWindow.SampleRate, ...
        'FFTSize', fftSize);

    plot(mainWindow.SpectrumAxes, freqs, 20*log10(magnitudes));
    xlim(mainWindow.SpectrumAxes, [0, mainWindow.SampleRate/2]);

catch ME
    uialert(mainWindow.Figure, ['Error analyzing spectrum: ' ME.message], 'Error');
end
end

function analyzePhase(mainWindow)
% Analyze stereo phase correlation

if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'Please load audio first', 'No Audio');
    return;
end

if size(mainWindow.LoadedAudio, 2) < 2
    uialert(mainWindow.Figure, 'Phase analysis requires stereo audio', 'Warning');
    return;
end

try
    L = mainWindow.LoadedAudio(:, 1);
    R = mainWindow.LoadedAudio(:, 2);

    % Calculate correlation
    windowSize = round(0.1 * mainWindow.SampleRate); % 0.1 sec window, dynamic for sample rate
    numWindows = floor(length(L) / windowSize);
    correlation = zeros(numWindows, 1);
    time = (1:numWindows) * windowSize / mainWindow.SampleRate;

    for i = 1:numWindows
        idx = (i-1)*windowSize + (1:windowSize);
        correlation(i) = corr(L(idx), R(idx));
    end

    plot(mainWindow.PhaseAxes, time, correlation);
    ylim(mainWindow.PhaseAxes, [-1, 1]);

catch ME
    uialert(mainWindow.Figure, ['Error analyzing phase: ' ME.message], 'Error');
end
end

function measureLoudness(mainWindow)
% Measure audio loudness metrics

if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'Please load audio first', 'No Audio');
    return;
end

try
    audioData = mainWindow.LoadedAudio;

    % Peak level
    peakLevel = 20 * log10(max(abs(audioData(:))));
    mainWindow.PeakLevelLabel.Text = sprintf('%.2f dB', peakLevel);

    % RMS level
    rmsLevel = 20 * log10(rms(audioData(:)));
    mainWindow.RMSLevelLabel.Text = sprintf('%.2f dB', rmsLevel);

    % Approximate LUFS (simplified calculation)
    lufs = rmsLevel - 0.691; % Rough approximation - does not implement ITU-R BS.1770 K-weighting or gating. Use this only as an estimate.
    mainWindow.LUFSLabel.Text = sprintf('%.2f LUFS', lufs);

    % Display level meter
    bar(mainWindow.LevelMeterAxes, [peakLevel, rmsLevel, lufs]);
    set(mainWindow.LevelMeterAxes, 'XTickLabel', {'Peak', 'RMS', 'LUFS'});
    ylabel(mainWindow.LevelMeterAxes, 'Level (dB)');

catch ME
    uialert(mainWindow.Figure, ['Error measuring loudness: ' ME.message], 'Error');
end
end

% ========== Library Panel Callbacks ==========

function updateLibraryBrowser(mainWindow)
% Update library browser based on category selection

try
    category = mainWindow.CategoryDropdown.Value;

    if strcmp(category, 'MATLAB Sounds')
        items = fieldnames(mainWindow.LibraryManager.MATLABSounds);
    elseif strcmp(category, 'All')
        items = {'Refresh catalog to see samples'};
    else
        items = {sprintf('Samples in %s category', category)};
    end

    mainWindow.SampleListBox.Items = items;
catch ME
    warning('Error updating library browser: %s', ME.message);
end
end

function searchLibrary(mainWindow, query)
% Search library for samples matching query

if isempty(query)
    updateLibraryBrowser(mainWindow);
    return;
end

try
    results = mainWindow.LibraryManager.searchSamples(query);

    if results.Count > 0
        items = cell(results.Count, 1);
        for i = 1:results.Count
            match = results.Matches{i};
            items{i} = sprintf('%s - %s', match.Category, match.Filename);
        end
        mainWindow.SampleListBox.Items = items;
    else
        mainWindow.SampleListBox.Items = {'No matches found'};
    end
catch ME
    uialert(mainWindow.Figure, ['Search error: ' ME.message], 'Error');
end
end

function selectSample(mainWindow, selectedValue)
% Display information about selected sample

mainWindow.SampleFilenameLabel.Text = selectedValue;
mainWindow.SampleCategoryLabel.Text = mainWindow.CategoryDropdown.Value;
end

function loadSelectedSample(mainWindow)
% Load selected sample into main window

selected = mainWindow.SampleListBox.Value;
if isempty(selected) || strcmp(selected, 'No samples loaded')
    return;
end

try
    category = mainWindow.CategoryDropdown.Value;
    [audioData, fs, info] = mainWindow.LibraryManager.loadSample(category, selected);

    mainWindow.LoadedAudio = audioData;
    mainWindow.SampleRate = fs;
    mainWindow.CurrentFile = selected;
    updateWaveformDisplay(mainWindow);

    uialert(mainWindow.Figure, 'Sample loaded successfully', 'Success');
catch ME
    uialert(mainWindow.Figure, ['Error loading sample: ' ME.message], 'Error');
end
end

function previewSample(mainWindow)
% Preview selected sample

uialert(mainWindow.Figure, 'Sample preview coming soon', 'Info');
end

function refreshLibraryCatalog(mainWindow)
% Refresh the library catalog

try
    mainWindow.LibraryManager.updateCatalog();
    updateLibraryBrowser(mainWindow);
    uialert(mainWindow.Figure, 'Library catalog refreshed', 'Success');
catch ME
    uialert(mainWindow.Figure, ['Error refreshing catalog: ' ME.message], 'Error');
end
end

function loadMATLABSound(mainWindow)
% Load MATLAB built-in sound

selected = mainWindow.MATLABSoundsListBox.Value;
if isempty(selected)
    return;
end

try
    [audioData, fs, info] = mainWindow.LibraryManager.loadMATLABSound(selected);

    mainWindow.LoadedAudio = audioData;
    mainWindow.SampleRate = fs;
    mainWindow.CurrentFile = selected;
    updateWaveformDisplay(mainWindow);

    uialert(mainWindow.Figure, sprintf('MATLAB sound "%s" loaded', selected), 'Success');
catch ME
    uialert(mainWindow.Figure, ['Error loading MATLAB sound: ' ME.message], 'Error');
end
end

function addSampleToLibrary(mainWindow)
% Add new sample to user library

uialert(mainWindow.Figure, 'Add sample feature coming soon', 'Info');
end

function createSampleCollection(mainWindow)
% Create new sample collection

uialert(mainWindow.Figure, 'Create collection feature coming soon', 'Info');
end

function importSampleCollection(mainWindow)
% Import sample collection

uialert(mainWindow.Figure, 'Import collection feature coming soon', 'Info');
end

function exportSampleCollection(mainWindow)
% Export sample collection

uialert(mainWindow.Figure, 'Export collection feature coming soon', 'Info');
end
