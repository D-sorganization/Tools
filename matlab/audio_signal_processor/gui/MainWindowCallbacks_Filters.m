% MAINWINDOWCALLBACKS_FILTERS - Filters tab callback functions
% These functions were in the original MainWindow.m and are preserved here

%% FILTERS TAB (from original - now needs to be added to new MainWindow)

% Add this function to createTabGroup in MainWindow.m:
% filtersTab = uitab(mainWindow.TabGroup, 'Title', '🔧 Filters');
% createFiltersPanel(mainWindow, filtersTab);

function createFiltersPanel(mainWindow, parent)
% Create filters panel with filter controls

filtersGrid = uigridlayout(parent, [4, 2]);
filtersGrid.RowHeight = {'fit', 'fit', 'fit', '2x'};
filtersGrid.ColumnWidth = {'1x', '1x'};
filtersGrid.Padding = [10, 10, 10, 10];
filtersGrid.RowSpacing = 8;
filtersGrid.ColumnSpacing = 10;

% Filter Type Selection
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

%% FILTER PANEL CALLBACKS

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
    mainWindow.StatusText.Text = sprintf('%s filter applied successfully', filterType);

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

    % Simple ideal response for preview
    switch filterType
        case 'Low Pass'
            response = freqs < cutoffFreq;
        case 'High Pass'
            response = freqs > cutoffFreq;
        case 'Band Pass'
            response = (freqs > cutoffFreq * 0.5) & (freqs < cutoffFreq * 1.5);
        case 'Band Stop'
            response = ~((freqs > cutoffFreq * 0.5) & (freqs < cutoffFreq * 1.5));
        case 'Butterworth'
            % Butterworth response
            n = mainWindow.FilterOrderSpinner.Value;
            response = 1 ./ sqrt(1 + (freqs / cutoffFreq).^(2*n));
        otherwise
            response = ones(size(freqs));
    end

    % Plot on filter response axes
    plot(mainWindow.FilterResponseAxes, freqs, 20*log10(response + eps));
    title(mainWindow.FilterResponseAxes, sprintf('%s Filter Response', filterType));
    xlabel(mainWindow.FilterResponseAxes, 'Frequency (Hz)');
    ylabel(mainWindow.FilterResponseAxes, 'Magnitude (dB)');
    grid(mainWindow.FilterResponseAxes, 'on');
    ylim(mainWindow.FilterResponseAxes, [-60, 5]);

    mainWindow.StatusText.Text = 'Filter response previewed';

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
mainWindow.PassbandRippleSpinner.Value = 1;

% Clear filter response plot
cla(mainWindow.FilterResponseAxes);
mainWindow.FilterResponseAxes.Title.String = 'Frequency Response';
grid(mainWindow.FilterResponseAxes, 'on');

mainWindow.StatusText.Text = 'Filter parameters reset';
end

%% END OF FILTER CALLBACKS
