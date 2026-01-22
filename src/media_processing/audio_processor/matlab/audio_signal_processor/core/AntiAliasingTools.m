function tools = AntiAliasingTools()
%ANTIALIASINGTOOLS Comprehensive anti-aliasing analysis and prevention toolkit
%
%   TOOLS = ANTIALIASINGTOOLS() creates a toolkit for detecting, analyzing,
%   and preventing aliasing artifacts in digital audio processing.
%
%   Key Features:
%   ------------
%   - Aliasing detection and analysis
%   - Nyquist frequency identification and warnings
%   - Anti-aliasing filter design
%   - Oversampling and downsampling with proper filtering
%   - Spectral analysis for aliasing artifacts
%   - Sample rate conversion with explicit anti-aliasing control
%
%   Aliasing Detection Methods:
%   --------------------------
%   detectAliasing(audio, fs) - Detect aliasing artifacts
%   analyzeNyquistViolations(audio, fs) - Find frequencies above Nyquist
%   measureAliasingLevel(audio, fs) - Quantify aliasing severity
%   detectFoldback(audio, fs) - Detect frequency foldback
%
%   Anti-Aliasing Filter Design:
%   ---------------------------
%   designAntiAliasingFilter(fs, cutoffRatio) - Design AA filter
%   applyAntiAliasingFilter(audio, fs) - Apply AA before downsampling
%   designOversamplingFilter(fs, factor) - Design for oversampling
%
%   Sample Rate Conversion:
%   ----------------------
%   downsampleWithAA(audio, fs, factor) - Downsample with proper AA
%   upsample(audio, fs, factor) - Upsample with interpolation filter
%   resampleWithAA(audio, oldFs, newFs) - Resample with explicit AA control
%
%   Oversampling:
%   ------------
%   oversample(audio, fs, factor) - Oversample by integer factor
%   processOversampled(audio, fs, factor, processFn) - Process at higher rate
%   downsampleBack(audio, fs, factor) - Downsample back to original rate
%
%   Analysis:
%   --------
%   getNyquistFrequency(fs) - Get Nyquist frequency
%   checkNyquistCompliance(audio, fs) - Check if audio respects Nyquist
%   plotSpectrum(audio, fs) - Plot spectrum with Nyquist line
%   findAliasingArtifacts(audio, fs) - Locate aliased content
%
%   Utilities:
%   ---------
%   calculateRequiredSampleRate(maxFreq) - Calculate minimum fs
%   suggestOversamplingFactor(fs, targetFs) - Suggest oversampling
%   getAntiAliasingInfo() - Get toolkit information
%
%   Example Usage:
%   -------------
%   % Detect aliasing in audio
%   tools = AntiAliasingTools();
%   [audio, fs] = audioread('audio.wav');
%
%   aliasing = tools.detectAliasing(audio, fs);
%   fprintf('Aliasing detected: %s\n', aliasing.detected);
%   fprintf('Aliasing level: %.2f dB\n', aliasing.level);
%
%   % Downsample with proper anti-aliasing
%   downsampled = tools.downsampleWithAA(audio, fs, 2);  % Downsample by 2x
%
%   % Oversample for processing
%   oversampled = tools.oversample(audio, fs, 4);  % 4x oversampling
%   % ... process at higher rate ...
%   final = tools.downsampleBack(oversampled, fs*4, 4);
%
%   % Check Nyquist compliance
%   compliance = tools.checkNyquistCompliance(audio, fs);
%   if ~compliance.compliant
%       fprintf('Warning: Frequencies above Nyquist detected!\n');
%   end
%
%   See also: resample, decimate, interp, fir1

% Initialize tools structure
tools = struct();
tools.Version = '1.0';
tools.NyquistTheorem = 'fs >= 2 * f_max';

% Aliasing detection
tools.detectAliasing = @(audio, fs, varargin) detectAliasing(audio, fs, varargin{:});
tools.analyzeNyquistViolations = @(audio, fs, varargin) analyzeNyquistViolations(audio, fs, varargin{:});
tools.measureAliasingLevel = @(audio, fs, varargin) measureAliasingLevel(audio, fs, varargin{:});
tools.detectFoldback = @(audio, fs, varargin) detectFoldback(audio, fs, varargin{:});

% Anti-aliasing filter design
tools.designAntiAliasingFilter = @(fs, cutoffRatio, varargin) designAntiAliasingFilter(fs, cutoffRatio, varargin{:});
tools.applyAntiAliasingFilter = @(audio, fs, varargin) applyAntiAliasingFilter(audio, fs, varargin{:});
tools.designOversamplingFilter = @(fs, factor, varargin) designOversamplingFilter(fs, factor, varargin{:});

% Sample rate conversion
tools.downsampleWithAA = @(audio, fs, factor, varargin) downsampleWithAA(audio, fs, factor, varargin{:});
tools.upsample = @(audio, fs, factor, varargin) upsampleAudio(audio, fs, factor, varargin{:});
tools.resampleWithAA = @(audio, oldFs, newFs, varargin) resampleWithAA(audio, oldFs, newFs, varargin{:});

% Oversampling
tools.oversample = @(audio, fs, factor, varargin) oversample(audio, fs, factor, varargin{:});
tools.processOversampled = @(audio, fs, factor, processFn) processOversampled(audio, fs, factor, processFn);
tools.downsampleBack = @(audio, fs, factor, varargin) downsampleBack(audio, fs, factor, varargin{:});

% Analysis
tools.getNyquistFrequency = @(fs) getNyquistFrequency(fs);
tools.checkNyquistCompliance = @(audio, fs, varargin) checkNyquistCompliance(audio, fs, varargin{:});
tools.plotSpectrum = @(audio, fs, varargin) plotSpectrum(audio, fs, varargin{:});
tools.findAliasingArtifacts = @(audio, fs, varargin) findAliasingArtifacts(audio, fs, varargin{:});

% Utilities
tools.calculateRequiredSampleRate = @(maxFreq) calculateRequiredSampleRate(maxFreq);
tools.suggestOversamplingFactor = @(fs, targetFs) suggestOversamplingFactor(fs, targetFs);
tools.getAntiAliasingInfo = @() getAntiAliasingInfo();
end

%% Aliasing Detection Methods

function result = detectAliasing(audio, fs, varargin)
% Detect aliasing artifacts in audio
%
%   Options:
%   'Threshold' - Detection threshold in dB (default: -60)
%   'Method' - Detection method ('spectral', 'correlation', 'both')

p = inputParser;
addParameter(p, 'Threshold', -60, @isnumeric);
addParameter(p, 'Method', 'spectral', @ischar);
parse(p, varargin{:});

options = p.Results;

% Convert to mono if stereo
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

% Calculate Nyquist frequency
nyquistFreq = fs / 2;

% Perform FFT
N = length(audio);
fftAudio = fft(audio);
fftMag = abs(fftAudio(1:floor(N/2)+1));
freqs = (0:length(fftMag)-1) * fs / N;

% Calculate power spectrum
powerSpectrum = 20 * log10(fftMag + eps);

% Find energy above 0.8 * Nyquist (aliasing likely if significant)
highFreqIdx = freqs > 0.8 * nyquistFreq;
highFreqEnergy = sum(fftMag(highFreqIdx).^2);
totalEnergy = sum(fftMag.^2);
highFreqRatio = highFreqEnergy / totalEnergy;

% Detect if significant energy near Nyquist
aliasingDetected = highFreqRatio > 0.01;  % 1% of energy near Nyquist

% Measure aliasing level
if highFreqEnergy > 0
    aliasingLevel = 10 * log10(highFreqRatio);
else
    aliasingLevel = -Inf;
end

% Find specific frequencies that might be aliased
suspiciousFreqs = freqs(powerSpectrum > options.Threshold & freqs > 0.8 * nyquistFreq);

% Create result structure
result = struct();
result.detected = aliasingDetected;
result.level = aliasingLevel;
result.highFreqRatio = highFreqRatio;
result.suspiciousFrequencies = suspiciousFreqs;
result.nyquistFrequency = nyquistFreq;
result.maxFrequencyDetected = max(freqs(powerSpectrum > options.Threshold));
result.recommendation = getAliasingRecommendation(aliasingDetected, aliasingLevel);
end

function violations = analyzeNyquistViolations(audio, fs, varargin)
% Analyze content above Nyquist frequency

p = inputParser;
addParameter(p, 'Threshold', -60, @isnumeric);
parse(p, varargin{:});

nyquistFreq = fs / 2;

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

% FFT analysis
N = length(audio);
fftAudio = fft(audio);
fftMag = abs(fftAudio(1:floor(N/2)+1));
freqs = (0:length(fftMag)-1) * fs / N;
powerSpectrum = 20 * log10(fftMag + eps);

% Find violations (energy above Nyquist)
% Note: In properly sampled audio, there should be NOTHING above Nyquist
violationIdx = freqs > nyquistFreq & powerSpectrum > p.Results.Threshold;

violations = struct();
violations.violating = any(violationIdx);
violations.violatingFrequencies = freqs(violationIdx);
violations.violatingLevels = powerSpectrum(violationIdx);
violations.numViolations = sum(violationIdx);
violations.nyquistFrequency = nyquistFreq;

if violations.violating
    violations.warning = 'CRITICAL: Content detected above Nyquist frequency! This should not happen in properly sampled audio.';
else
    violations.warning = 'No Nyquist violations detected.';
end
end

function level = measureAliasingLevel(audio, fs, varargin)
% Quantify aliasing severity

result = detectAliasing(audio, fs, varargin{:});
level = result.level;
end

function foldback = detectFoldback(audio, fs, varargin)
% Detect frequency foldback (aliasing artifact)
%
% Foldback occurs when frequencies above Nyquist "fold back" below Nyquist

p = inputParser;
addParameter(p, 'WindowSize', 0.1, @isnumeric);  % seconds
parse(p, varargin{:});

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

windowSamples = round(p.Results.WindowSize * fs);
hopSamples = round(windowSamples / 2);

numFrames = floor((length(audio) - windowSamples) / hopSamples) + 1;
foldbackDetected = false(numFrames, 1);

nyquistFreq = fs / 2;

for i = 1:numFrames
    startIdx = (i-1) * hopSamples + 1;
    endIdx = startIdx + windowSamples - 1;
    frame = audio(startIdx:endIdx);

    % FFT of frame
    fftFrame = abs(fft(frame));
    fftFrame = fftFrame(1:floor(length(fftFrame)/2)+1);
    freqs = (0:length(fftFrame)-1) * fs / length(frame);

    % Look for suspicious patterns:
    % - Unexpected energy in very high frequencies
    % - Mirror images (foldback signature)
    highFreqEnergy = sum(fftFrame(freqs > 0.8 * nyquistFreq).^2);
    totalEnergy = sum(fftFrame.^2);

    if highFreqEnergy / totalEnergy > 0.05
        foldbackDetected(i) = true;
    end
end

timeAxis = (0:numFrames-1) * hopSamples / fs;

foldback = struct();
foldback.detected = any(foldbackDetected);
foldback.times = timeAxis(foldbackDetected);
foldback.percentage = sum(foldbackDetected) / numFrames * 100;
end

%% Anti-Aliasing Filter Design

function filterObj = designAntiAliasingFilter(fs, cutoffRatio, varargin)
% Design anti-aliasing lowpass filter
%
%   fs - Sample rate
%   cutoffRatio - Cutoff as ratio of Nyquist (default: 0.9)
%
%   Options:
%   'FilterOrder' - Filter order (default: 8)
%   'FilterType' - 'fir' or 'iir' (default: 'fir')
%   'Window' - Window function for FIR (default: 'kaiser')

p = inputParser;
addParameter(p, 'FilterOrder', 8, @isnumeric);
addParameter(p, 'FilterType', 'fir', @ischar);
addParameter(p, 'Window', 'kaiser', @ischar);
addParameter(p, 'Attenuation', 80, @isnumeric);  % dB
parse(p, varargin{:});

options = p.Results;

nyquistFreq = fs / 2;
cutoffFreq = cutoffRatio * nyquistFreq;

% Design filter
if strcmp(options.FilterType, 'fir')
    % FIR filter with specified window
    filterOrder = options.FilterOrder;

    % Design using fir1
    if strcmp(options.Window, 'kaiser')
        % Calculate Kaiser beta for desired attenuation
        beta = 0.1102 * (options.Attenuation - 8.7);
        win = kaiser(filterOrder + 1, beta);
        b = fir1(filterOrder, cutoffFreq / nyquistFreq, 'low', win);
    else
        b = fir1(filterOrder, cutoffFreq / nyquistFreq, 'low', options.Window);
    end

    filterObj = struct('type', 'fir', 'b', b, 'a', 1, ...
                      'fs', fs, 'cutoff', cutoffFreq);

elseif strcmp(options.FilterType, 'iir')
    % IIR filter (Butterworth)
    filterOrder = options.FilterOrder;
    [b, a] = butter(filterOrder, cutoffFreq / nyquistFreq, 'low');

    filterObj = struct('type', 'iir', 'b', b, 'a', a, ...
                      'fs', fs, 'cutoff', cutoffFreq);
else
    error('AntiAliasingTools:InvalidFilterType', 'FilterType must be ''fir'' or ''iir''');
end

% Add filter response information
[h, f] = freqz(filterObj.b, filterObj.a, 1024, fs);
filterObj.response = struct('h', h, 'f', f);
end

function filtered = applyAntiAliasingFilter(audio, fs, varargin)
% Apply anti-aliasing filter to audio

p = inputParser;
addParameter(p, 'CutoffRatio', 0.9, @isnumeric);
addParameter(p, 'FilterOrder', 8, @isnumeric);
addParameter(p, 'FilterType', 'fir', @ischar);
parse(p, varargin{:});

% Design filter
filterObj = designAntiAliasingFilter(fs, p.Results.CutoffRatio, ...
    'FilterOrder', p.Results.FilterOrder, 'FilterType', p.Results.FilterType);

% Apply filter
if strcmp(filterObj.type, 'fir')
    % Zero-phase filtering for FIR
    filtered = filtfilt(filterObj.b, 1, audio);
else
    % IIR filter
    filtered = filtfilt(filterObj.b, filterObj.a, audio);
end
end

function filterObj = designOversamplingFilter(fs, factor, varargin)
% Design filter for oversampling

% For oversampling by factor N, design lowpass at fs/2
oversampledFs = fs * factor;
nyquistOriginal = fs / 2;

% Design lowpass to prevent imaging
filterObj = designAntiAliasingFilter(oversampledFs, nyquistOriginal / (oversampledFs/2), varargin{:});
end

%% Sample Rate Conversion

function downsampled = downsampleWithAA(audio, fs, factor, varargin)
% Downsample with explicit anti-aliasing filter
%
%   factor - Downsampling factor (e.g., 2 = half the sample rate)

p = inputParser;
addParameter(p, 'FilterOrder', 8, @isnumeric);
addParameter(p, 'Method', 'decimate', @ischar);  % 'decimate' or 'manual'
parse(p, varargin{:});

if strcmp(p.Results.Method, 'decimate')
    % Use MATLAB's decimate (has built-in AA filter)
    downsampled = decimate(audio, factor, p.Results.FilterOrder, 'fir');
else
    % Manual: Apply AA filter then downsample
    % Design filter at new Nyquist
    newNyquist = (fs / factor) / 2;

    filtered = applyAntiAliasingFilter(audio, fs, ...
        'CutoffRatio', newNyquist / (fs/2), ...
        'FilterOrder', p.Results.FilterOrder);

    % Downsample
    downsampled = filtered(1:factor:end, :);
end
end

function upsampled = upsampleAudio(audio, fs, factor, varargin)
% Upsample with interpolation filter

p = inputParser;
addParameter(p, 'FilterOrder', 8, @isnumeric);
addParameter(p, 'Method', 'interp', @ischar);  % 'interp' or 'manual'
parse(p, varargin{:});

if strcmp(p.Results.Method, 'interp')
    % Use MATLAB's interp (has built-in interpolation filter)
    upsampled = interp(audio, factor, p.Results.FilterOrder, 'fir');
else
    % Manual upsampling
    [nSamples, nChannels] = size(audio);
    upsampled = zeros(nSamples * factor, nChannels);

    % Insert zeros
    upsampled(1:factor:end, :) = audio;

    % Design interpolation filter
    upsampledFs = fs * factor;
    filterObj = designAntiAliasingFilter(upsampledFs, 0.9, ...
        'FilterOrder', p.Results.FilterOrder);

    % Apply filter and scale
    upsampled = filtfilt(filterObj.b, 1, upsampled) * factor;
end
end

function resampled = resampleWithAA(audio, oldFs, newFs, varargin)
% Resample with explicit anti-aliasing control

p = inputParser;
addParameter(p, 'FilterOrder', 8, @isnumeric);
addParameter(p, 'Quality', 'high', @ischar);  % 'low', 'medium', 'high'
parse(p, varargin{:});

if newFs < oldFs
    % Downsampling - need anti-aliasing
    fprintf('Downsampling from %d Hz to %d Hz with anti-aliasing...\n', oldFs, newFs);

    % Apply AA filter at new Nyquist
    newNyquist = newFs / 2;
    filtered = applyAntiAliasingFilter(audio, oldFs, ...
        'CutoffRatio', newNyquist / (oldFs/2), ...
        'FilterOrder', p.Results.FilterOrder);

    % Resample
    resampled = resample(filtered, newFs, oldFs);

elseif newFs > oldFs
    % Upsampling - need interpolation filter
    fprintf('Upsampling from %d Hz to %d Hz with interpolation...\n', oldFs, newFs);

    resampled = resample(audio, newFs, oldFs);

else
    % Same rate
    resampled = audio;
end
end

%% Oversampling Methods

function oversampled = oversample(audio, fs, factor, varargin)
% Oversample audio by integer factor

p = inputParser;
addParameter(p, 'FilterOrder', 16, @isnumeric);
parse(p, varargin{:});

% Upsample
oversampled = upsampleAudio(audio, fs, factor, ...
    'FilterOrder', p.Results.FilterOrder);

fprintf('Oversampled from %d Hz to %d Hz (factor: %dx)\n', ...
    fs, fs * factor, factor);
end

function processed = processOversampled(audio, fs, factor, processFn)
% Process audio at oversampled rate then downsample
%
%   processFn - Function handle to processing function

% Oversample
oversampled = oversample(audio, fs, factor);

% Process at high rate
fprintf('Processing at %d Hz...\n', fs * factor);
processed = processFn(oversampled);

% Downsample back
processed = downsampleBack(processed, fs * factor, factor);
end

function downsampled = downsampleBack(audio, fs, factor, varargin)
% Downsample from oversampled rate back to original

downsampled = downsampleWithAA(audio, fs, factor, varargin{:});

fprintf('Downsampled from %d Hz to %d Hz\n', fs, fs / factor);
end

%% Analysis Methods

function nyquistFreq = getNyquistFrequency(fs)
% Get Nyquist frequency (fs/2)

nyquistFreq = fs / 2;
fprintf('Sample Rate: %d Hz\n', fs);
fprintf('Nyquist Frequency: %.2f Hz\n', nyquistFreq);
fprintf('Nyquist Theorem: All signal content must be below %.2f Hz\n', nyquistFreq);
end

function compliance = checkNyquistCompliance(audio, fs, varargin)
% Check if audio respects Nyquist theorem

p = inputParser;
addParameter(p, 'Threshold', -60, @isnumeric);
addParameter(p, 'Verbose', true, @islogical);
parse(p, varargin{:});

nyquistFreq = fs / 2;

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

% FFT analysis
N = length(audio);
fftAudio = fft(audio);
fftMag = abs(fftAudio(1:floor(N/2)+1));
freqs = (0:length(fftMag)-1) * fs / N;
powerSpectrum = 20 * log10(fftMag + eps);

% Find maximum frequency with significant energy
maxFreqIdx = find(powerSpectrum > p.Results.Threshold, 1, 'last');
if ~isempty(maxFreqIdx)
    maxFreq = freqs(maxFreqIdx);
else
    maxFreq = 0;
end

% Check compliance
compliant = maxFreq < nyquistFreq;

% Calculate headroom
headroom = nyquistFreq - maxFreq;
headroomPercent = (headroom / nyquistFreq) * 100;

compliance = struct();
compliance.compliant = compliant;
compliance.nyquistFrequency = nyquistFreq;
compliance.maxFrequency = maxFreq;
compliance.headroom = headroom;
compliance.headroomPercent = headroomPercent;

if p.Results.Verbose
    fprintf('\n=== Nyquist Compliance Check ===\n');
    fprintf('Sample Rate: %d Hz\n', fs);
    fprintf('Nyquist Frequency: %.2f Hz\n', nyquistFreq);
    fprintf('Maximum Frequency in Audio: %.2f Hz\n', maxFreq);
    fprintf('Headroom: %.2f Hz (%.1f%%)\n', headroom, headroomPercent);

    if compliant
        fprintf('Status: ✓ COMPLIANT - Audio respects Nyquist theorem\n');
        if headroomPercent < 10
            fprintf('Warning: Limited headroom. Consider higher sample rate.\n');
        end
    else
        fprintf('Status: ✗ NON-COMPLIANT - Content above Nyquist detected!\n');
        fprintf('ACTION REQUIRED: Increase sample rate or apply anti-aliasing filter\n');
    end
    fprintf('================================\n\n');
end
end

function plotSpectrum(audio, fs, varargin)
% Plot spectrum with Nyquist frequency line

p = inputParser;
addParameter(p, 'Title', 'Frequency Spectrum with Nyquist Line', @ischar);
addParameter(p, 'ShowAliasing', true, @islogical);
parse(p, varargin{:});

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

% Calculate spectrum
N = length(audio);
fftAudio = fft(audio);
fftMag = abs(fftAudio(1:floor(N/2)+1));
freqs = (0:length(fftMag)-1) * fs / N;
powerSpectrum = 20 * log10(fftMag + eps);

nyquistFreq = fs / 2;

% Plot
figure('Name', 'Anti-Aliasing Analysis');

% Main spectrum plot
subplot(2, 1, 1);
plot(freqs, powerSpectrum, 'b-', 'LineWidth', 1);
hold on;

% Nyquist line
yLimits = ylim;
plot([nyquistFreq, nyquistFreq], yLimits, 'r--', 'LineWidth', 2);

% Danger zone (above 0.8 * Nyquist)
dangerZone = 0.8 * nyquistFreq;
plot([dangerZone, dangerZone], yLimits, 'y--', 'LineWidth', 1.5);

% Shade aliasing risk area
aliasingZone = [dangerZone, nyquistFreq, nyquistFreq, dangerZone];
patch(aliasingZone, [yLimits(1), yLimits(1), yLimits(2), yLimits(2)], ...
    'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title(p.Results.Title);
legend('Audio Spectrum', 'Nyquist Frequency', 'Aliasing Risk Zone', 'Location', 'best');
grid on;
hold off;

% Zoomed view near Nyquist
subplot(2, 1, 2);
zoomStart = max(0, 0.5 * nyquistFreq);
zoomEnd = min(max(freqs), 1.5 * nyquistFreq);
zoomIdx = freqs >= zoomStart & freqs <= zoomEnd;

plot(freqs(zoomIdx), powerSpectrum(zoomIdx), 'b-', 'LineWidth', 1.5);
hold on;
plot([nyquistFreq, nyquistFreq], yLimits, 'r--', 'LineWidth', 2);
plot([dangerZone, dangerZone], yLimits, 'y--', 'LineWidth', 1.5);

xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Zoomed View Near Nyquist Frequency');
legend('Audio Spectrum', 'Nyquist Frequency', 'Aliasing Risk');
grid on;
hold off;

% Add text annotations
annotation('textbox', [0.15, 0.95, 0.7, 0.03], ...
    'String', sprintf('Sample Rate: %d Hz | Nyquist: %.2f Hz', fs, nyquistFreq), ...
    'EdgeColor', 'none', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

function artifacts = findAliasingArtifacts(audio, fs, varargin)
% Locate specific aliasing artifacts in time

p = inputParser;
addParameter(p, 'WindowSize', 0.1, @isnumeric);  % seconds
addParameter(p, 'Threshold', -60, @isnumeric);   % dB
parse(p, varargin{:});

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

windowSamples = round(p.Results.WindowSize * fs);
hopSamples = round(windowSamples / 2);
numFrames = floor((length(audio) - windowSamples) / hopSamples) + 1;

nyquistFreq = fs / 2;
artifacts = struct('times', [], 'severity', [], 'frequencies', {});

for i = 1:numFrames
    startIdx = (i-1) * hopSamples + 1;
    endIdx = startIdx + windowSamples - 1;
    frame = audio(startIdx:endIdx);

    % Analyze frame
    result = detectAliasing(frame, fs, 'Threshold', p.Results.Threshold);

    if result.detected
        time = startIdx / fs;
        artifacts.times(end+1) = time;
        artifacts.severity(end+1) = result.level;
        artifacts.frequencies{end+1} = result.suspiciousFrequencies;
    end
end

fprintf('Found %d aliasing artifacts\n', length(artifacts.times));
end

%% Utility Methods

function requiredFs = calculateRequiredSampleRate(maxFreq)
% Calculate minimum required sample rate for given maximum frequency
%
% According to Nyquist theorem: fs >= 2 * f_max
% In practice: fs >= 2.5 * f_max (for realistic filters)

theoreticalMin = 2 * maxFreq;
practicalMin = 2.5 * maxFreq;

% Round up to standard sample rates
standardRates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000];
requiredFs = min(standardRates(standardRates >= practicalMin));

fprintf('Maximum Frequency: %.2f Hz\n', maxFreq);
fprintf('Theoretical Minimum (Nyquist): %.2f Hz\n', theoreticalMin);
fprintf('Practical Minimum (with filter rolloff): %.2f Hz\n', practicalMin);
fprintf('Recommended Sample Rate: %d Hz\n', requiredFs);
end

function factor = suggestOversamplingFactor(fs, targetFs)
% Suggest oversampling factor

if targetFs <= fs
    factor = 1;
    fprintf('No oversampling needed (target <= current)\n');
    return;
end

% Find smallest integer factor
factor = ceil(targetFs / fs);

fprintf('Current: %d Hz, Target: %d Hz\n', fs, targetFs);
fprintf('Suggested oversampling factor: %dx (result: %d Hz)\n', ...
    factor, fs * factor);
end

function info = getAntiAliasingInfo()
% Get anti-aliasing toolkit information

info = struct();
info.NyquistTheorem = 'fs >= 2 * f_max';
info.NyquistDefinition = 'The Nyquist frequency is half the sample rate (fs/2)';
info.AliasingDefinition = 'Aliasing occurs when frequencies above Nyquist fold back into the audible range';
info.PreventionMethods = {
    'Anti-aliasing filter before downsampling', ...
    'Oversampling for nonlinear processing', ...
    'Proper sample rate selection (2.5x max frequency)', ...
    'Band-limiting before A/D conversion'
};

info.StandardSampleRates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000];
info.CommonMaxFrequencies = struct(...
    'Telephone', 3400, ...
    'AM Radio', 5000, ...
    'FM Radio', 15000, ...
    'CD Quality', 20000, ...
    'Human Hearing', 20000, ...
    'High Res Audio', 40000 ...
);

% Display info
fprintf('\n=== Anti-Aliasing Information ===\n');
fprintf('Nyquist Theorem: %s\n', info.NyquistTheorem);
fprintf('\nAliasing: %s\n', info.AliasingDefinition);
fprintf('\nPrevention Methods:\n');
for i = 1:length(info.PreventionMethods)
    fprintf('  %d. %s\n', i, info.PreventionMethods{i});
end
fprintf('\nStandard Sample Rates (Hz): ');
fprintf('%d ', info.StandardSampleRates);
fprintf('\n================================\n\n');
end

function recommendation = getAliasingRecommendation(detected, level)
% Get recommendation based on aliasing detection

if ~detected
    recommendation = 'No aliasing detected. Audio is clean.';
elseif level > -20
    recommendation = 'SEVERE ALIASING DETECTED. Immediate action required: increase sample rate or apply strong anti-aliasing filter.';
elseif level > -40
    recommendation = 'Moderate aliasing detected. Consider resampling or filtering.';
else
    recommendation = 'Minor aliasing detected. May be acceptable depending on application.';
end
end
