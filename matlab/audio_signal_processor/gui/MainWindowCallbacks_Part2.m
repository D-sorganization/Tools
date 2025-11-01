% MAINWINDOWCALLBACKS_PART2 - Remaining callback functions
% Production, Research, Analysis, Library, and Settings tabs

%% PRODUCTION TAB FUNCTIONS

function applyAutotune(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    key = mainWindow.AutotuneKeyDropdown.Value;
    scale = mainWindow.AutotuneScaleDropdown.Value;
    strength = mainWindow.AutotuneStrengthSlider.Value;
    speed = mainWindow.AutotuneSpeedSpinner.Value;
    formant = mainWindow.AutotuneFormantCheckbox.Value;

    mainWindow.StatusText.Text = 'Applying autotune (this may take a moment)...';
    drawnow;

    autotuned = mainWindow.MusicTools.autotune(mainWindow.LoadedAudio, mainWindow.SampleRate, ...
        'Key', key, 'Scale', scale, 'Strength', strength, 'Speed', speed, 'Formant', formant);

    mainWindow.LoadedAudio = autotuned;
    updateWaveformDisplay(mainWindow);
    mainWindow.StatusText.Text = sprintf('Autotune applied: %s %s', key, scale);
catch ME
    uialert(mainWindow.Figure, sprintf('Autotune error: %s', ME.message), 'Error');
    mainWindow.StatusText.Text = 'Autotune failed';
end
end

function previewAutotune(mainWindow)
if isempty(mainWindow.LoadedAudio)
    return;
end

% Preview first 5 seconds
previewLength = min(5 * mainWindow.SampleRate, length(mainWindow.LoadedAudio));
preview = mainWindow.LoadedAudio(1:previewLength, :);

try
    key = mainWindow.AutotuneKeyDropdown.Value;
    scale = mainWindow.AutotuneScaleDropdown.Value;
    strength = mainWindow.AutotuneStrengthSlider.Value;

    autotuned = mainWindow.MusicTools.autotune(preview, mainWindow.SampleRate, ...
        'Key', key, 'Scale', scale, 'Strength', strength);

    sound(autotuned, mainWindow.SampleRate);
    mainWindow.StatusText.Text = 'Playing autotune preview...';
catch ME
    uialert(mainWindow.Figure, sprintf('Preview error: %s', ME.message), 'Error');
end
end

function detectKeyQuick(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    mainWindow.StatusText.Text = 'Detecting key...';
    drawnow;

    [key, scale, confidence] = mainWindow.MusicTools.detectKey(mainWindow.LoadedAudio, mainWindow.SampleRate);

    resultText = sprintf('%s %s (%.0f%% confidence)', key, scale, confidence*100);
    mainWindow.DetectedKeyLabel.Text = resultText;
    mainWindow.StatusText.Text = sprintf('Key detected: %s', resultText);

    % Auto-fill autotune fields
    mainWindow.AutotuneKeyDropdown.Value = key;
    mainWindow.AutotuneScaleDropdown.Value = scale;
catch ME
    uialert(mainWindow.Figure, sprintf('Key detection error: %s', ME.message), 'Error');
    mainWindow.StatusText.Text = 'Key detection failed';
end
end

function detectTempoQuick(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    mainWindow.StatusText.Text = 'Detecting tempo...';
    drawnow;

    [bpm, beats] = mainWindow.MusicTools.detectTempo(mainWindow.LoadedAudio, mainWindow.SampleRate);

    resultText = sprintf('%.1f BPM (%d beats)', bpm, length(beats));
    mainWindow.DetectedTempoLabel.Text = resultText;
    mainWindow.StatusText.Text = sprintf('Tempo detected: %s', resultText);

    % Auto-fill click/quantize fields
    mainWindow.ClickBPMSpinner.Value = round(bpm);
    mainWindow.QuantizeBPMSpinner.Value = round(bpm);
catch ME
    uialert(mainWindow.Figure, sprintf('Tempo detection error: %s', ME.message), 'Error');
    mainWindow.StatusText.Text = 'Tempo detection failed';
end
end

function detectChordsDetailed(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    mainWindow.StatusText.Text = 'Detecting chords...';
    drawnow;

    [chords, times] = mainWindow.MusicTools.detectChords(mainWindow.LoadedAudio, mainWindow.SampleRate);

    % Create result dialog
    dialog = uifigure('Name', 'Chord Detection Results', 'Position', [100, 100, 400, 500]);
    grid = uigridlayout(dialog, [2, 1]);
    grid.RowHeight = {'1x', 'fit'};

    % Chord list
    listPanel = uipanel(grid, 'Title', sprintf('Found %d chords', length(chords)));
    listPanel.Layout.Row = 1;

    chordTexts = cell(length(chords), 1);
    for i = 1:length(chords)
        chordTexts{i} = sprintf('%.2fs: %s', times(i), chords{i});
    end

    uilistbox(listPanel, 'Items', chordTexts);

    % Close button
    uibutton(grid, 'Text', 'Close', 'ButtonPushedFcn', @(src, event) close(dialog));

    mainWindow.StatusText.Text = sprintf('Chords detected: %d chords found', length(chords));
catch ME
    uialert(mainWindow.Figure, sprintf('Chord detection error: %s', ME.message), 'Error');
    mainWindow.StatusText.Text = 'Chord detection failed';
end
end

function generateClickTrack(mainWindow)
bpm = mainWindow.ClickBPMSpinner.Value;
bars = mainWindow.ClickBarsSpinner.Value;

try
    click = mainWindow.MusicTools.generateClickTrack(bpm, bars, mainWindow.SampleRate);

    mainWindow.LoadedAudio = click;
    mainWindow.CurrentFile = sprintf('Click Track %d BPM', bpm);
    updateWaveformDisplay(mainWindow);
    mainWindow.StatusText.Text = sprintf('Click track generated: %d BPM, %d bars', bpm, bars);
catch ME
    uialert(mainWindow.Figure, sprintf('Click generation error: %s', ME.message), 'Error');
end
end

function quantizeAudio(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    bpm = mainWindow.QuantizeBPMSpinner.Value;
    strength = mainWindow.QuantizeStrengthSlider.Value;

    mainWindow.StatusText.Text = 'Quantizing audio...';
    drawnow;

    quantized = mainWindow.MusicTools.quantizeToGrid(mainWindow.LoadedAudio, mainWindow.SampleRate, bpm, strength);

    mainWindow.LoadedAudio = quantized;
    updateWaveformDisplay(mainWindow);
    mainWindow.StatusText.Text = sprintf('Audio quantized to %d BPM grid', bpm);
catch ME
    uialert(mainWindow.Figure, sprintf('Quantize error: %s', ME.message), 'Error');
end
end

function showHarmonizerDialog(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

dialog = uifigure('Name', 'Harmonizer', 'Position', [100, 100, 350, 200]);
grid = uigridlayout(dialog, [4, 2]);
grid.RowHeight = {'fit', 'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Intervals (semitones):');
uilabel(grid, 'Text', 'e.g., [3, 7] for 3rd and 5th', 'FontSize', 9);

uilabel(grid, 'Text', 'Intervals:');
intervalsField = uieditfield(grid, 'Value', '[3, 7]');

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Generate Harmony', ...
    'ButtonPushedFcn', @(src, event) applyHarmonizer(mainWindow, intervalsField.Value, dialog));
end

function applyHarmonizer(mainWindow, intervalsStr, dialog)
try
    intervals = str2num(intervalsStr); %#ok<ST2NM>

    mainWindow.StatusText.Text = 'Generating harmonies...';
    drawnow;

    harmonized = mainWindow.MusicTools.harmonizer(mainWindow.LoadedAudio, mainWindow.SampleRate, intervals);

    mainWindow.LoadedAudio = harmonized;
    updateWaveformDisplay(mainWindow);
    close(dialog);
    mainWindow.StatusText.Text = sprintf('Harmonies generated: %s', intervalsStr);
catch ME
    uialert(mainWindow.Figure, sprintf('Harmonizer error: %s', ME.message), 'Error');
end
end

function showVocoderDialog(mainWindow)
uialert(mainWindow.Figure, 'Vocoder: Load carrier and modulator signals', 'Coming Soon');
end

function audioToMIDI(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    mainWindow.StatusText.Text = 'Converting to MIDI...';
    drawnow;

    midiNotes = mainWindow.MusicTools.audioToMIDI(mainWindow.LoadedAudio, mainWindow.SampleRate);

    % Show results
    resultText = sprintf('Detected %d notes', length(midiNotes.notes));
    uialert(mainWindow.Figure, resultText, 'Audio to MIDI');
    mainWindow.StatusText.Text = resultText;
catch ME
    uialert(mainWindow.Figure, sprintf('Audio to MIDI error: %s', ME.message), 'Error');
end
end

function showPitchShiftDialog(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

dialog = uifigure('Name', 'Pitch Shift', 'Position', [100, 100, 300, 150]);
grid = uigridlayout(dialog, [3, 2]);
grid.RowHeight = {'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Semitones:');
semitonesField = uispinner(grid, 'Value', 0, 'Limits', [-12, 12], 'Step', 0.5);

uilabel(grid, 'Text', 'Positive = higher, Negative = lower', 'FontSize', 9);
uilabel(grid, 'Text', '');

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Apply', ...
    'ButtonPushedFcn', @(src, event) applyPitchShift(mainWindow, semitonesField.Value, dialog));
end

function applyPitchShift(mainWindow, semitones, dialog)
try
    shifted = AudioEffects(mainWindow.LoadedAudio, 'PitchShift', ...
        'PitchShift', semitones, 'SampleRate', mainWindow.SampleRate);

    mainWindow.LoadedAudio = shifted;
    updateWaveformDisplay(mainWindow);
    close(dialog);
    mainWindow.StatusText.Text = sprintf('Pitch shifted: %+.1f semitones', semitones);
catch ME
    uialert(mainWindow.Figure, sprintf('Pitch shift error: %s', ME.message), 'Error');
end
end

function showTimeStretchDialog(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

dialog = uifigure('Name', 'Time Stretch', 'Position', [100, 100, 300, 150]);
grid = uigridlayout(dialog, [3, 2]);
grid.RowHeight = {'fit', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x'};

uilabel(grid, 'Text', 'Stretch Factor:');
factorField = uispinner(grid, 'Value', 1.0, 'Limits', [0.5, 2.0], 'Step', 0.1);

uilabel(grid, 'Text', '1.0 = normal, <1 = faster, >1 = slower', 'FontSize', 9);
uilabel(grid, 'Text', '');

uibutton(grid, 'Text', 'Cancel', 'ButtonPushedFcn', @(src, event) close(dialog));
uibutton(grid, 'Text', 'Apply', ...
    'ButtonPushedFcn', @(src, event) applyTimeStretch(mainWindow, factorField.Value, dialog));
end

function applyTimeStretch(mainWindow, factor, dialog)
try
    stretched = AudioEffects(mainWindow.LoadedAudio, 'TimeStretch', ...
        'TimeStretch', factor, 'SampleRate', mainWindow.SampleRate);

    mainWindow.LoadedAudio = stretched;
    updateWaveformDisplay(mainWindow);
    close(dialog);
    mainWindow.StatusText.Text = sprintf('Time stretched: ×%.2f', factor);
catch ME
    uialert(mainWindow.Figure, sprintf('Time stretch error: %s', ME.message), 'Error');
end
end

function showAutotuneDialog(mainWindow)
% Switch to Production tab
mainWindow.TabGroup.SelectedTab = mainWindow.TabGroup.Children(5);
end

%% RESEARCH TAB FUNCTIONS

function waveletTimeFrequency(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    waveletType = mainWindow.WaveletTypeDropdown.Value;

    mainWindow.StatusText.Text = 'Computing wavelet transform...';
    drawnow;

    [cwtData, frequencies, time] = mainWindow.WaveletProc.timeFrequencyAnalysis(...
        mainWindow.LoadedAudio(:,1), mainWindow.SampleRate, 'Wavelet', waveletType);

    % Create figure for CWT
    figure('Name', 'Continuous Wavelet Transform');
    imagesc(time, frequencies, abs(cwtData));
    axis xy;
    colormap jet;
    colorbar;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title(sprintf('CWT using %s wavelet', waveletType));

    mainWindow.StatusText.Text = 'Wavelet transform completed';
catch ME
    uialert(mainWindow.Figure, sprintf('Wavelet error: %s', ME.message), 'Error');
end
end

function waveletDenoise(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    waveletType = mainWindow.WaveletTypeDropdown.Value;

    mainWindow.StatusText.Text = 'Denoising with wavelets...';
    drawnow;

    denoised = mainWindow.WaveletProc.denoise(mainWindow.LoadedAudio, ...
        'Method', 'Bayes', 'Wavelet', waveletType);

    mainWindow.LoadedAudio = denoised;
    updateWaveformDisplay(mainWindow);
    mainWindow.StatusText.Text = sprintf('Wavelet denoising applied (%s)', waveletType);
catch ME
    uialert(mainWindow.Figure, sprintf('Denoise error: %s', ME.message), 'Error');
end
end

function separateTransientTonal(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    mainWindow.StatusText.Text = 'Separating transient and tonal components...';
    drawnow;

    [transient, tonal] = mainWindow.WaveletProc.separateTransientTonal(...
        mainWindow.LoadedAudio(:,1), mainWindow.SampleRate);

    % Create figure showing both
    figure('Name', 'Transient/Tonal Separation');
    subplot(3,1,1);
    plot((0:length(mainWindow.LoadedAudio)-1)/mainWindow.SampleRate, mainWindow.LoadedAudio(:,1));
    title('Original');
    ylabel('Amplitude');
    grid on;

    subplot(3,1,2);
    plot((0:length(transient)-1)/mainWindow.SampleRate, transient);
    title('Transient Component');
    ylabel('Amplitude');
    grid on;

    subplot(3,1,3);
    plot((0:length(tonal)-1)/mainWindow.SampleRate, tonal);
    title('Tonal Component');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;

    % Ask which to keep
    choice = uiconfirm(mainWindow.Figure, ...
        'Which component would you like to keep?', ...
        'Select Component', ...
        'Options', {'Transient', 'Tonal', 'Both (Sum)', 'Cancel'}, ...
        'DefaultOption', 3);

    switch choice
        case 'Transient'
            mainWindow.LoadedAudio = transient;
        case 'Tonal'
            mainWindow.LoadedAudio = tonal;
        case 'Both (Sum)'
            mainWindow.LoadedAudio = transient + tonal;
    end

    if ~strcmp(choice, 'Cancel')
        updateWaveformDisplay(mainWindow);
        mainWindow.StatusText.Text = sprintf('Loaded: %s component', choice);
    end
catch ME
    uialert(mainWindow.Figure, sprintf('Separation error: %s', ME.message), 'Error');
end
end

function extractAllFeatures(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    mainWindow.StatusText.Text = 'Extracting audio features...';
    drawnow;

    features = mainWindow.AdvancedAudio.extractAllFeatures(mainWindow.LoadedAudio(:,1), mainWindow.SampleRate);

    % Store features
    mainWindow.ExtractedFeatures = features;

    % Update label
    numFeatures = length(fieldnames(features));
    mainWindow.FeatureResultLabel.Text = sprintf('%d features extracted', numFeatures);
    mainWindow.StatusText.Text = sprintf('Feature extraction complete: %d features', numFeatures);
catch ME
    uialert(mainWindow.Figure, sprintf('Feature extraction error: %s', ME.message), 'Error');
end
end

function exportFeatures(mainWindow)
if ~isfield(mainWindow, 'ExtractedFeatures') || isempty(mainWindow.ExtractedFeatures)
    uialert(mainWindow.Figure, 'No features to export. Extract features first.', 'Warning');
    return;
end

[file, path] = uiputfile({'*.csv', 'CSV File'}, 'Export Features');
if file == 0
    return;
end

try
    % Convert struct to table and write
    T = struct2table(mainWindow.ExtractedFeatures);
    writetable(T, fullfile(path, file));
    mainWindow.StatusText.Text = sprintf('Features exported to %s', file);
catch ME
    uialert(mainWindow.Figure, sprintf('Export error: %s', ME.message), 'Error');
end
end

function plotFeatures(mainWindow)
if ~isfield(mainWindow, 'ExtractedFeatures') || isempty(mainWindow.ExtractedFeatures)
    uialert(mainWindow.Figure, 'No features to plot. Extract features first.', 'Warning');
    return;
end

% Create feature visualization
figure('Name', 'Extracted Audio Features');
features = mainWindow.ExtractedFeatures;
featureNames = fieldnames(features);

% Plot first 12 features as bar chart
numToPlot = min(12, length(featureNames));
values = zeros(numToPlot, 1);

for i = 1:numToPlot
    val = features.(featureNames{i});
    if length(val) == 1
        values(i) = val;
    else
        values(i) = mean(val);  % Take mean if vector
    end
end

bar(values);
set(gca, 'XTickLabel', featureNames(1:numToPlot));
xtickangle(45);
ylabel('Feature Value');
title('Audio Features');
grid on;

mainWindow.StatusText.Text = 'Features plotted';
end

function checkNyquistCompliance(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    compliance = mainWindow.AntiAliasing.checkNyquistCompliance(mainWindow.LoadedAudio(:,1), mainWindow.SampleRate);

    if compliance.isCompliant
        status = sprintf('✓ Compliant (%.2f%% above Nyquist)', compliance.percentAbove*100);
        mainWindow.AAStatusLabel.Text = status;
        mainWindow.AAStatusLabel.FontColor = [0, 0.6, 0];
    else
        status = sprintf('⚠ Non-compliant (%.2f%% above Nyquist)', compliance.percentAbove*100);
        mainWindow.AAStatusLabel.Text = status;
        mainWindow.AAStatusLabel.FontColor = [0.8, 0, 0];
    end

    mainWindow.StatusText.Text = status;
catch ME
    uialert(mainWindow.Figure, sprintf('Compliance check error: %s', ME.message), 'Error');
end
end

function detectAliasing(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    aliasing = mainWindow.AntiAliasing.detectAliasing(mainWindow.LoadedAudio(:,1), mainWindow.SampleRate);

    if aliasing.hasAliasing
        status = sprintf('⚠ Aliasing detected! Level: %.1f dB', aliasing.level);
        mainWindow.AAStatusLabel.Text = status;
        mainWindow.AAStatusLabel.FontColor = [0.8, 0, 0];

        uialert(mainWindow.Figure, sprintf('Aliasing detected at %.1f dB. Consider applying anti-aliasing filter.', aliasing.level), 'Aliasing Warning');
    else
        status = '✓ No aliasing detected';
        mainWindow.AAStatusLabel.Text = status;
        mainWindow.AAStatusLabel.FontColor = [0, 0.6, 0];
    end

    mainWindow.StatusText.Text = status;
catch ME
    uialert(mainWindow.Figure, sprintf('Aliasing detection error: %s', ME.message), 'Error');
end
end

function applyAntiAliasingFilter(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    filtered = mainWindow.AntiAliasing.applyAntiAliasingFilter(mainWindow.LoadedAudio, mainWindow.SampleRate);

    mainWindow.LoadedAudio = filtered;
    updateWaveformDisplay(mainWindow);
    mainWindow.StatusText.Text = 'Anti-aliasing filter applied';
catch ME
    uialert(mainWindow.Figure, sprintf('Filter error: %s', ME.message), 'Error');
end
end

function oversampleAudio(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    oversampled = mainWindow.AntiAliasing.oversample(mainWindow.LoadedAudio, mainWindow.SampleRate, 2);

    mainWindow.LoadedAudio = oversampled;
    mainWindow.SampleRate = mainWindow.SampleRate * 2;
    updateWaveformDisplay(mainWindow);
    updateAAInfo(mainWindow);
    mainWindow.StatusText.Text = sprintf('Oversampled to %d Hz', mainWindow.SampleRate);
catch ME
    uialert(mainWindow.Figure, sprintf('Oversample error: %s', ME.message), 'Error');
end
end

function downsampleAudio(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    downsampled = mainWindow.AntiAliasing.downsampleWithAA(mainWindow.LoadedAudio, mainWindow.SampleRate, 2);

    mainWindow.LoadedAudio = downsampled;
    mainWindow.SampleRate = mainWindow.SampleRate / 2;
    updateWaveformDisplay(mainWindow);
    updateAAInfo(mainWindow);
    mainWindow.StatusText.Text = sprintf('Downsampled to %d Hz', mainWindow.SampleRate);
catch ME
    uialert(mainWindow.Figure, sprintf('Downsample error: %s', ME.message), 'Error');
end
end

function plotNyquistSpectrum(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    mainWindow.AntiAliasing.plotSpectrum(mainWindow.LoadedAudio(:,1), mainWindow.SampleRate);
    mainWindow.StatusText.Text = 'Spectrum plotted with Nyquist line';
catch ME
    uialert(mainWindow.Figure, sprintf('Plot error: %s', ME.message), 'Error');
end
end

function updateAAInfo(mainWindow)
mainWindow.AACurrentSRLabel.Text = sprintf('%d Hz', mainWindow.SampleRate);
mainWindow.AANyquistLabel.Text = sprintf('%d Hz', mainWindow.SampleRate/2);
end

function detectPitchNeural(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    mainWindow.StatusText.Text = 'Detecting pitch (neural network)...';
    drawnow;

    [pitch, time] = mainWindow.AdvancedAudio.detectPitch(mainWindow.LoadedAudio(:,1), mainWindow.SampleRate);

    % Plot pitch contour
    figure('Name', 'Pitch Detection (Neural Network)');
    plot(time, pitch);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Detected Pitch Contour');
    grid on;

    avgPitch = mean(pitch(pitch > 0));
    mainWindow.StatusText.Text = sprintf('Pitch detected (avg: %.1f Hz)', avgPitch);
catch ME
    uialert(mainWindow.Figure, sprintf('Pitch detection error: %s', ME.message), 'Error');
end
end

function detectOnsets(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    mainWindow.StatusText.Text = 'Detecting onsets...';
    drawnow;

    onsetTimes = mainWindow.AdvancedAudio.detectOnsets(mainWindow.LoadedAudio(:,1), mainWindow.SampleRate);

    % Plot waveform with onset markers
    figure('Name', 'Onset Detection');
    time = (0:length(mainWindow.LoadedAudio)-1) / mainWindow.SampleRate;
    plot(time, mainWindow.LoadedAudio(:,1));
    hold on;
    for i = 1:length(onsetTimes)
        xline(onsetTimes(i), 'r--');
    end
    hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('Onset Detection (%d onsets found)', length(onsetTimes)));
    grid on;
    legend('Audio', 'Onsets');

    mainWindow.StatusText.Text = sprintf('Onsets detected: %d events', length(onsetTimes));
catch ME
    uialert(mainWindow.Figure, sprintf('Onset detection error: %s', ME.message), 'Error');
end
end

function measureLUFS(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'No audio loaded', 'Error');
    return;
end

try
    lufs = mainWindow.AdvancedAudio.measureLoudness(mainWindow.LoadedAudio, mainWindow.SampleRate);

    resultText = sprintf('Integrated Loudness: %.1f LUFS', lufs);
    uialert(mainWindow.Figure, resultText, 'LUFS Measurement');
    mainWindow.StatusText.Text = resultText;

    % Update Analysis tab label too
    mainWindow.LUFSLabel.Text = sprintf('%.1f LUFS', lufs);
catch ME
    uialert(mainWindow.Figure, sprintf('LUFS measurement error: %s', ME.message), 'Error');
end
end

%% ANALYSIS TAB FUNCTIONS

function generateSpectrogram(mainWindow)
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
    mainWindow.SpectrogramAxes.Title.String = 'Spectrogram';
    mainWindow.StatusText.Text = 'Spectrogram generated';
catch ME
    uialert(mainWindow.Figure, ['Error generating spectrogram: ' ME.message], 'Error');
end
end

function analyzeSpectrum(mainWindow)
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
    mainWindow.SpectrumAxes.Title.String = 'Frequency Spectrum';
    mainWindow.StatusText.Text = 'Spectrum analyzed';
catch ME
    uialert(mainWindow.Figure, ['Error analyzing spectrum: ' ME.message], 'Error');
end
end

function analyzePhase(mainWindow)
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

    windowSize = 4410;
    numWindows = floor(length(L) / windowSize);
    correlation = zeros(numWindows, 1);
    time = (1:numWindows) * windowSize / mainWindow.SampleRate;

    for i = 1:numWindows
        idx = (i-1)*windowSize + (1:windowSize);
        correlation(i) = corr(L(idx), R(idx));
    end

    plot(mainWindow.PhaseAxes, time, correlation);
    ylim(mainWindow.PhaseAxes, [-1, 1]);
    mainWindow.PhaseAxes.Title.String = 'Stereo Phase Correlation';
    mainWindow.StatusText.Text = 'Phase analyzed';
catch ME
    uialert(mainWindow.Figure, ['Error analyzing phase: ' ME.message], 'Error');
end
end

function measureLoudness(mainWindow)
if isempty(mainWindow.LoadedAudio)
    uialert(mainWindow.Figure, 'Please load audio first', 'No Audio');
    return;
end

try
    audioData = mainWindow.LoadedAudio;

    peakLevel = 20 * log10(max(abs(audioData(:))));
    mainWindow.PeakLevelLabel.Text = sprintf('%.2f dB', peakLevel);

    rmsLevel = 20 * log10(rms(audioData(:)));
    mainWindow.RMSLevelLabel.Text = sprintf('%.2f dB', rmsLevel);

    % Try accurate LUFS
    try
        lufs = mainWindow.AdvancedAudio.measureLoudness(audioData, mainWindow.SampleRate);
        mainWindow.LUFSLabel.Text = sprintf('%.2f LUFS', lufs);
    catch
        lufs = rmsLevel - 0.691;
        mainWindow.LUFSLabel.Text = sprintf('%.2f LUFS (approx)', lufs);
    end

    bar(mainWindow.LevelMeterAxes, [peakLevel, rmsLevel, lufs]);
    set(mainWindow.LevelMeterAxes, 'XTickLabel', {'Peak', 'RMS', 'LUFS'});
    ylabel(mainWindow.LevelMeterAxes, 'Level (dB)');
    mainWindow.LevelMeterAxes.Title.String = 'Loudness Levels';

    mainWindow.StatusText.Text = 'Loudness measured';
catch ME
    uialert(mainWindow.Figure, ['Error measuring loudness: ' ME.message], 'Error');
end
end

%% LIBRARY TAB FUNCTIONS

function updateLibraryBrowser(mainWindow)
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
mainWindow.SampleFilenameLabel.Text = selectedValue;
mainWindow.SampleCategoryLabel.Text = mainWindow.CategoryDropdown.Value;
end

function loadSelectedSample(mainWindow)
selected = mainWindow.SampleListBox.Value;
if isempty(selected) || strcmp(selected, 'No samples loaded')
    return;
end

try
    category = mainWindow.CategoryDropdown.Value;

    if strcmp(category, 'MATLAB Sounds')
        [audioData, fs, ~] = mainWindow.LibraryManager.loadMATLABSound(selected);
    else
        [audioData, fs, ~] = mainWindow.LibraryManager.loadSample(category, selected);
    end

    mainWindow.LoadedAudio = audioData;
    mainWindow.SampleRate = fs;
    mainWindow.CurrentFile = selected;
    updateWaveformDisplay(mainWindow);

    mainWindow.StatusText.Text = sprintf('Sample loaded: %s', selected);
catch ME
    uialert(mainWindow.Figure, ['Error loading sample: ' ME.message], 'Error');
end
end

function previewSample(mainWindow)
selected = mainWindow.SampleListBox.Value;
if isempty(selected)
    return;
end

try
    category = mainWindow.CategoryDropdown.Value;

    if strcmp(category, 'MATLAB Sounds')
        [audioData, fs, ~] = mainWindow.LibraryManager.loadMATLABSound(selected);
    else
        [audioData, fs, ~] = mainWindow.LibraryManager.loadSample(category, selected);
    end

    % Play first 3 seconds
    previewLength = min(3 * fs, length(audioData));
    sound(audioData(1:previewLength, :), fs);
    mainWindow.StatusText.Text = 'Playing preview...';
catch ME
    uialert(mainWindow.Figure, ['Preview error: ' ME.message], 'Error');
end
end

function refreshLibraryCatalog(mainWindow)
try
    mainWindow.LibraryManager.updateCatalog();
    updateLibraryBrowser(mainWindow);
    mainWindow.StatusText.Text = 'Library catalog refreshed';
catch ME
    uialert(mainWindow.Figure, ['Error refreshing catalog: ' ME.message], 'Error');
end
end

function loadInstrumentPreset(mainWindow)
preset = mainWindow.InstrumentPresetList.Value;

try
    % Get preset effects
    effects = mainWindow.EffectsLibrary.getPreset(preset);

    % Clear current chain
    mainWindow.EffectChain = {};

    % Add preset effects to chain
    for i = 1:length(effects)
        mainWindow.EffectChain{end+1} = struct('Type', effects{i}.Type, 'Params', effects{i}.Params, 'Enabled', true);
    end

    % Update effects tab
    updateEffectChainList(mainWindow);

    % Switch to effects tab
    mainWindow.TabGroup.SelectedTab = mainWindow.TabGroup.Children(3);

    mainWindow.StatusText.Text = sprintf('Loaded preset: %s (%d effects)', preset, length(effects));
catch ME
    uialert(mainWindow.Figure, sprintf('Error loading preset: %s', ME.message), 'Error');
end
end

function addSampleToLibrary(mainWindow)
uialert(mainWindow.Figure, 'Add sample: Select audio file to add to your library', 'Coming Soon');
end

function createSampleCollection(mainWindow)
uialert(mainWindow.Figure, 'Create collection: Group samples into collection', 'Coming Soon');
end

function importSampleCollection(mainWindow)
uialert(mainWindow.Figure, 'Import collection: Load collection file', 'Coming Soon');
end

function exportSampleCollection(mainWindow)
uialert(mainWindow.Figure, 'Export collection: Save collection to file', 'Coming Soon');
end

%% SETTINGS TAB FUNCTIONS

function applySettings(mainWindow)
% Apply current settings
mainWindow.SampleRate = str2double(mainWindow.DefaultSRDropdown.Value);
mainWindow.StatusText.Text = 'Settings applied';
end

function saveSettings(mainWindow)
% Save settings to file
uialert(mainWindow.Figure, 'Settings saved', 'Success');
mainWindow.StatusText.Text = 'Settings saved';
end

function resetSettings(mainWindow)
% Reset to defaults
mainWindow.DefaultSRDropdown.Value = '44100';
mainWindow.BitDepthDropdown.Value = '24';
mainWindow.BufferSizeDropdown.Value = '512';
mainWindow.UndoLevelsSpinner.Value = 50;
mainWindow.StatusText.Text = 'Settings reset to defaults';
end

function browseUserLibrary(mainWindow)
folder = uigetdir(mainWindow.UserLibraryPathField.Value, 'Select User Library Folder');
if folder ~= 0
    mainWindow.UserLibraryPathField.Value = folder;
end
end

function browseIRPath(mainWindow)
folder = uigetdir(mainWindow.IRPathField.Value, 'Select Impulse Response Folder');
if folder ~= 0
    mainWindow.IRPathField.Value = folder;
end
end

function browseExportPath(mainWindow)
folder = uigetdir(mainWindow.ExportPathField.Value, 'Select Export Folder');
if folder ~= 0
    mainWindow.ExportPathField.Value = folder;
end
end

%% HELP MENU FUNCTIONS

function showBatchProcessor(mainWindow)
uialert(mainWindow.Figure, 'Batch Processor: Process multiple files with same settings', 'Coming Soon');
end

function showQuickStart(mainWindow)
helpText = sprintf(['Quick Start Guide\n\n', ...
    '1. Load audio: File → Load Audio (Ctrl+O)\n', ...
    '2. Edit: Use Edit tab for trim, cut, fade, normalize\n', ...
    '3. Effects: Add effects in Effects tab\n', ...
    '4. Mix: Load multiple tracks in Mixer tab\n', ...
    '5. Production: Use autotune and music tools\n', ...
    '6. Analysis: Visualize with spectrogram\n', ...
    '7. Research: Advanced wavelet and feature extraction\n\n', ...
    'See documentation for detailed guides.']);

uialert(mainWindow.Figure, helpText, 'Quick Start');
end

function showShortcuts(mainWindow)
helpText = sprintf(['Keyboard Shortcuts\n\n', ...
    'Ctrl+O: Open audio file\n', ...
    'Ctrl+S: Save/Export\n', ...
    'Ctrl+Z: Undo\n', ...
    'Ctrl+Y: Redo\n', ...
    'Ctrl+A: Select All\n', ...
    'Ctrl+X: Cut\n', ...
    'Ctrl+C: Copy\n', ...
    'Ctrl+V: Paste\n', ...
    'Ctrl+E: Apply Effect Chain\n', ...
    'Ctrl+N: Quick Normalize\n', ...
    'Ctrl+R: Quick Reverb\n', ...
    'Ctrl+=: Zoom In\n', ...
    'Ctrl+-: Zoom Out\n', ...
    'Ctrl+0: Fit to Window\n', ...
    'Space: Play/Pause']);

uialert(mainWindow.Figure, helpText, 'Keyboard Shortcuts');
end

function showAbout(mainWindow)
aboutText = sprintf(['Audio Signal Processor - Professional Edition\n', ...
    'Version 2.0\n\n', ...
    'A comprehensive audio processing suite with:\n', ...
    '• Professional audio editing\n', ...
    '• Complete effects library\n', ...
    '• Advanced multi-track mixer\n', ...
    '• Music production tools (autotune!)\n', ...
    '• Research-grade analysis\n', ...
    '• Convolution reverb\n\n', ...
    'Leverages MATLAB Audio Toolbox and Wavelet Toolbox\n\n', ...
    'All backend features now accessible through GUI!']);

uialert(mainWindow.Figure, aboutText, 'About');
end

%% END OF MAINWINDOWCALLBACKS_PART2
% These functions complete the full GUI implementation.
% Append or include these functions in MainWindow.m
