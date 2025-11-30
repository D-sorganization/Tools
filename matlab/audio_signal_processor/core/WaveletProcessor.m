function processor = WaveletProcessor()
%WAVELETPROCESSOR Wavelet-based audio processing leveraging Wavelet Toolbox
%
%   PROCESSOR = WAVELETPROCESSOR() creates a wavelet processor object that
%   provides advanced wavelet-based audio analysis and processing features
%   using MATLAB's Wavelet Toolbox.
%
%   Key Features:
%   ------------
%   - Wavelet denoising (superior to traditional noise gates)
%   - Continuous wavelet transform (CWT) for time-frequency analysis
%   - Wavelet synchrosqueezing for improved resolution
%   - Transient/tonal component separation
%   - Wavelet-based compression
%   - Multi-resolution analysis
%
%   Denoising Methods:
%   -----------------
%   denoise(audio, options) - Wavelet denoising with multiple methods
%   denoiseByLevel(audio, level, options) - Level-dependent denoising
%   adaptiveDenoising(audio, options) - Adaptive threshold selection
%
%   Analysis Methods:
%   ----------------
%   timeFrequencyAnalysis(audio, fs) - CWT-based time-frequency analysis
%   synchrosqueeze(audio, fs) - Wavelet synchrosqueezing transform
%   coherenceAnalysis(audio1, audio2, fs) - Wavelet coherence
%   multiscaleAnalysis(audio, fs) - Multi-resolution decomposition
%
%   Component Separation:
%   --------------------
%   separateTransientTonal(audio, fs) - Separate transients from tonal
%   separateHarmonicPercussive(audio, fs) - Harmonic/percussive separation
%   extractRhythm(audio, fs) - Extract rhythmic components
%
%   Compression:
%   -----------
%   compress(audio, level) - Wavelet-based compression
%   decompress(compressed) - Decompress wavelet-compressed audio
%
%   Visualization:
%   -------------
%   plotScalogram(cfs, f, t) - Plot wavelet scalogram
%   plotWaveletDecomposition(audio, fs) - Plot multi-level decomposition
%   plotCoherence(coherence, f, t) - Plot wavelet coherence
%
%   Example Usage:
%   -------------
%   % Create processor
%   wp = WaveletProcessor();
%
%   % Denoise audio
%   [audio, fs] = audioread('noisy_speech.wav');
%   cleanAudio = wp.denoise(audio, 'Wavelet', 'db4', 'Method', 'Bayes');
%
%   % Time-frequency analysis
%   [cfs, frequencies] = wp.timeFrequencyAnalysis(audio, fs);
%   wp.plotScalogram(cfs, frequencies, (0:length(audio)-1)/fs);
%
%   % Separate transients from tonal
%   [transients, tonal] = wp.separateTransientTonal(audio, fs);
%
%   % Compress audio
%   [compressed, ratio] = wp.compress(audio, 5);
%   decompressed = wp.decompress(compressed);
%
%   See also: cwt, wdenoise, wsst, wcoherence, modwt

% Initialize processor structure
processor = struct();
processor.Version = '1.0';
processor.HasWaveletToolbox = license('test', 'Wavelet_Toolbox');

if ~processor.HasWaveletToolbox
    warning('WaveletProcessor:NoToolbox', ...
        'Wavelet Toolbox not available. Some features will be limited.');
end

% Add denoising methods
processor.denoise = @(audio, varargin) denoise(audio, varargin{:});
processor.denoiseByLevel = @(audio, level, varargin) denoiseByLevel(audio, level, varargin{:});
processor.adaptiveDenoising = @(audio, varargin) adaptiveDenoising(audio, varargin{:});

% Add analysis methods
processor.timeFrequencyAnalysis = @(audio, fs, varargin) timeFrequencyAnalysis(audio, fs, varargin{:});
processor.synchrosqueeze = @(audio, fs, varargin) synchrosqueeze(audio, fs, varargin{:});
processor.coherenceAnalysis = @(audio1, audio2, fs, varargin) coherenceAnalysis(audio1, audio2, fs, varargin{:});
processor.multiscaleAnalysis = @(audio, fs, varargin) multiscaleAnalysis(audio, fs, varargin{:});

% Add component separation methods
processor.separateTransientTonal = @(audio, fs, varargin) separateTransientTonal(audio, fs, varargin{:});
processor.separateHarmonicPercussive = @(audio, fs, varargin) separateHarmonicPercussive(audio, fs, varargin{:});
processor.extractRhythm = @(audio, fs, varargin) extractRhythm(audio, fs, varargin{:});

% Add compression methods
processor.compress = @(audio, level, varargin) compressAudio(audio, level, varargin{:});
processor.decompress = @(compressed) decompressAudio(compressed);

% Add visualization methods
processor.plotScalogram = @(cfs, f, t, varargin) plotScalogram(cfs, f, t, varargin{:});
processor.plotWaveletDecomposition = @(audio, fs, varargin) plotWaveletDecomposition(audio, fs, varargin{:});
processor.plotCoherence = @(coherence, f, t, varargin) plotCoherence(coherence, f, t, varargin{:});

% Add utility methods
processor.listWavelets = @() listWavelets();
processor.getInfo = @() getInfo(processor);
end

%% Denoising Methods

function cleanedAudio = denoise(audio, varargin)
% Wavelet denoising with multiple methods
%
%   CLEANEDAUDIO = denoise(AUDIO, Name, Value)
%
%   Options:
%   -------
%   'Wavelet' - Wavelet family ('db4', 'coif3', 'sym4', etc.)
%   'Level' - Decomposition level (default: automatic)
%   'Method' - Denoising method ('Bayes', 'BlockJS', 'SURE', 'Minimax')
%   'Threshold' - Threshold rule ('Soft', 'Hard')
%   'NoiseEstimate' - Noise estimation method ('LevelIndependent', 'LevelDependent')

p = inputParser;
addParameter(p, 'Wavelet', 'db4', @ischar);
addParameter(p, 'Level', [], @(x) isempty(x) || isnumeric(x));
addParameter(p, 'Method', 'Bayes', @ischar);
addParameter(p, 'Threshold', 'Soft', @ischar);
addParameter(p, 'NoiseEstimate', 'LevelDependent', @ischar);
parse(p, varargin{:});

options = p.Results;

% Convert to mono if stereo (process each channel separately)
[nSamples, nChannels] = size(audio);
cleanedAudio = zeros(size(audio));

for ch = 1:nChannels
    channelData = audio(:, ch);

    try
        % Use wdenoise if available (Wavelet Toolbox)
        if exist('wdenoise', 'file') == 2
            cleanedAudio(:, ch) = wdenoise(channelData, options.Level, ...
                'Wavelet', options.Wavelet, ...
                'DenoisingMethod', options.Method, ...
                'ThresholdRule', options.Threshold, ...
                'NoiseEstimate', options.NoiseEstimate);
        else
            % Fallback: manual wavelet denoising
            cleanedAudio(:, ch) = manualWaveletDenoise(channelData, options);
        end
    catch ME
        warning('WaveletProcessor:DenoiseError', ...
            'Error denoising channel %d: %s', ch, ME.message);
        cleanedAudio(:, ch) = channelData;  % Return original on error
    end
end
end

function cleanedAudio = denoiseByLevel(audio, level, varargin)
% Level-dependent wavelet denoising

p = inputParser;
addParameter(p, 'Wavelet', 'db4', @ischar);
addParameter(p, 'Threshold', 'Soft', @ischar);
parse(p, varargin{:});

options = p.Results;

[nSamples, nChannels] = size(audio);
cleanedAudio = zeros(size(audio));

for ch = 1:nChannels
    try
        if exist('wdenoise', 'file') == 2
            cleanedAudio(:, ch) = wdenoise(audio(:, ch), level, ...
                'Wavelet', options.Wavelet, ...
                'ThresholdRule', options.Threshold, ...
                'NoiseEstimate', 'LevelDependent');
        else
            cleanedAudio(:, ch) = manualWaveletDenoise(audio(:, ch), options);
        end
    catch ME
        warning('WaveletProcessor:DenoiseError', 'Error: %s', ME.message);
        cleanedAudio(:, ch) = audio(:, ch);
    end
end
end

function cleanedAudio = adaptiveDenoising(audio, varargin)
% Adaptive threshold selection for denoising

p = inputParser;
addParameter(p, 'Wavelet', 'db4', @ischar);
addParameter(p, 'MinLevel', 1, @isnumeric);
addParameter(p, 'MaxLevel', 6, @isnumeric);
parse(p, varargin{:});

options = p.Results;

% Try different levels and select best SNR
bestSNR = -Inf;
bestAudio = audio;

for level = options.MinLevel:options.MaxLevel
    denoised = denoiseByLevel(audio, level, 'Wavelet', options.Wavelet);

    % Estimate SNR
    noise = audio - denoised;
    snr = 10 * log10(sum(denoised.^2) / sum(noise.^2));

    if mean(snr) > bestSNR
        bestSNR = mean(snr);
        bestAudio = denoised;
    end
end

cleanedAudio = bestAudio;
end

function cleanedAudio = manualWaveletDenoise(audio, options)
% Manual wavelet denoising implementation (fallback)

% Determine decomposition level
if isempty(options.Level)
    options.Level = wmaxlev(length(audio), options.Wavelet);
end

% Decompose
[C, L] = wavedec(audio, options.Level, options.Wavelet);

% Estimate noise standard deviation
sigma = median(abs(C(L(1):L(2)))) / 0.6745;

% Threshold calculation (VisuShrink)
thr = sigma * sqrt(2 * log(length(audio)));

% Apply thresholding
if strcmp(options.Threshold, 'Soft')
    C_thresh = wthresh(C, 's', thr);
else
    C_thresh = wthresh(C, 'h', thr);
end

% Reconstruct
cleanedAudio = waverec(C_thresh, L, options.Wavelet);

% Ensure same length
cleanedAudio = cleanedAudio(1:length(audio));
end

%% Analysis Methods

function [cfs, frequencies, t] = timeFrequencyAnalysis(audio, fs, varargin)
% Continuous wavelet transform for time-frequency analysis

p = inputParser;
addParameter(p, 'Wavelet', 'amor', @ischar);  % Analytic Morlet
addParameter(p, 'FrequencyLimits', [20, fs/2], @isnumeric);
addParameter(p, 'VoicesPerOctave', 10, @isnumeric);
parse(p, varargin{:});

options = p.Results;

% Convert to mono if stereo
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if exist('cwt', 'file') == 2
        % Use Wavelet Toolbox CWT
        [cfs, frequencies] = cwt(audio, options.Wavelet, fs, ...
            'FrequencyLimits', options.FrequencyLimits, ...
            'VoicesPerOctave', options.VoicesPerOctave);
    else
        % Fallback: simple spectrogram
        warning('WaveletProcessor:NoCWT', 'CWT not available, using spectrogram');
        [cfs, frequencies, t] = spectrogram(audio, 256, 200, 256, fs);
        return;
    end
catch ME
    warning('WaveletProcessor:CWTError', 'Error in CWT: %s', ME.message);
    [cfs, frequencies, t] = spectrogram(audio, 256, 200, 256, fs);
    return;
end

% Time vector
t = (0:length(audio)-1) / fs;
end

function [sst, frequencies, t] = synchrosqueeze(audio, fs, varargin)
% Wavelet synchrosqueezing transform for improved time-frequency resolution

p = inputParser;
addParameter(p, 'Wavelet', 'amor', @ischar);
addParameter(p, 'FrequencyLimits', [20, fs/2], @isnumeric);
parse(p, varargin{:});

options = p.Results;

% Convert to mono if stereo
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if exist('wsst', 'file') == 2
        [sst, frequencies] = wsst(audio, fs, options.Wavelet, ...
            'FrequencyLimits', options.FrequencyLimits);
    else
        % Fallback to regular CWT
        warning('WaveletProcessor:NoWSST', 'WSST not available, using CWT');
        [sst, frequencies] = cwt(audio, options.Wavelet, fs, ...
            'FrequencyLimits', options.FrequencyLimits);
    end
catch ME
    warning('WaveletProcessor:WSSTError', 'Error: %s', ME.message);
    [sst, frequencies] = deal([], []);
    return;
end

t = (0:length(audio)-1) / fs;
end

function [wcoh, wcs, f, coi] = coherenceAnalysis(audio1, audio2, fs, varargin)
% Wavelet coherence between two signals

p = inputParser;
addParameter(p, 'FrequencyLimits', [20, fs/2], @isnumeric);
parse(p, varargin{:});

options = p.Results;

% Convert to mono if stereo
if size(audio1, 2) > 1
    audio1 = mean(audio1, 2);
end
if size(audio2, 2) > 1
    audio2 = mean(audio2, 2);
end

% Ensure same length
minLen = min(length(audio1), length(audio2));
audio1 = audio1(1:minLen);
audio2 = audio2(1:minLen);

try
    if exist('wcoherence', 'file') == 2
        [wcoh, wcs, f, coi] = wcoherence(audio1, audio2, fs, ...
            'FrequencyLimits', options.FrequencyLimits);
    else
        warning('WaveletProcessor:NoCoherence', 'wcoherence not available');
        [wcoh, wcs, f, coi] = deal([], [], [], []);
    end
catch ME
    warning('WaveletProcessor:CoherenceError', 'Error: %s', ME.message);
    [wcoh, wcs, f, coi] = deal([], [], [], []);
end
end

function [approximation, details] = multiscaleAnalysis(audio, fs, varargin)
% Multi-resolution wavelet decomposition

p = inputParser;
addParameter(p, 'Wavelet', 'db4', @ischar);
addParameter(p, 'Level', 5, @isnumeric);
parse(p, varargin{:});

options = p.Results;

% Convert to mono if stereo
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if exist('modwt', 'file') == 2
        % Use maximal overlap discrete wavelet transform
        wt = modwt(audio, options.Wavelet, options.Level);

        % Extract approximation and details
        approximation = wt(1, :)';
        details = cell(options.Level, 1);
        for i = 1:options.Level
            details{i} = wt(i+1, :)';
        end
    else
        % Fallback: standard DWT
        [C, L] = wavedec(audio, options.Level, options.Wavelet);
        approximation = wrcoef('a', C, L, options.Wavelet, options.Level);

        details = cell(options.Level, 1);
        for i = 1:options.Level
            details{i} = wrcoef('d', C, L, options.Wavelet, i);
        end
    end
catch ME
    warning('WaveletProcessor:MultiscaleError', 'Error: %s', ME.message);
    approximation = audio;
    details = {};
end
end

%% Component Separation Methods

function [transients, tonal] = separateTransientTonal(audio, fs, varargin)
% Separate transient and tonal components using wavelets

p = inputParser;
addParameter(p, 'Wavelet', 'db4', @ischar);
addParameter(p, 'Level', 5, @isnumeric);
addParameter(p, 'TransientThreshold', 0.3, @isnumeric);
parse(p, varargin{:});

options = p.Results;

[nSamples, nChannels] = size(audio);
transients = zeros(size(audio));
tonal = zeros(size(audio));

for ch = 1:nChannels
    channelData = audio(:, ch);

    % Wavelet decomposition
    [approximation, details] = multiscaleAnalysis(channelData, fs, ...
        'Wavelet', options.Wavelet, 'Level', options.Level);

    % Transients are in high-frequency details
    transientComponent = zeros(size(channelData));
    for i = 1:min(3, length(details))  % Use first 3 detail levels
        transientComponent = transientComponent + details{i};
    end

    % Tonal is approximation + mid-frequency details
    tonalComponent = approximation;
    for i = 4:length(details)
        tonalComponent = tonalComponent + details{i};
    end

    transients(:, ch) = transientComponent;
    tonal(:, ch) = tonalComponent;
end
end

function [harmonic, percussive] = separateHarmonicPercussive(audio, fs, varargin)
% Harmonic/percussive source separation using wavelets

p = inputParser;
addParameter(p, 'Method', 'wavelet', @ischar);
parse(p, varargin{:});

% Use transient/tonal separation as approximation
[percussive, harmonic] = separateTransientTonal(audio, fs, varargin{:});
end

function [rhythm, nonRhythm] = extractRhythm(audio, fs, varargin)
% Extract rhythmic components

p = inputParser;
addParameter(p, 'Wavelet', 'db4', @ischar);
addParameter(p, 'RhythmBand', [100, 500], @isnumeric);  % Hz
parse(p, varargin{:});

options = p.Results;

% Use wavelet packet decomposition for better frequency resolution
[transients, ~] = separateTransientTonal(audio, fs, varargin{:});

% Filter transients to rhythm band
nyquist = fs / 2;
normalizedBand = options.RhythmBand / nyquist;

[b, a] = butter(4, normalizedBand, 'bandpass');
rhythm = filtfilt(b, a, transients);

nonRhythm = audio - rhythm;
end

%% Compression Methods

function [compressed, compressionRatio] = compressAudio(audio, level, varargin)
% Wavelet-based audio compression

p = inputParser;
addParameter(p, 'Wavelet', 'bior4.4', @ischar);  % Good for compression
addParameter(p, 'ThresholdFraction', 0.01, @isnumeric);
parse(p, varargin{:});

options = p.Results;

[nSamples, nChannels] = size(audio);
compressed = struct();
compressed.Wavelet = options.Wavelet;
compressed.Level = level;
compressed.OriginalLength = nSamples;
compressed.NumChannels = nChannels;
compressed.Channels = cell(nChannels, 1);

for ch = 1:nChannels
    % Wavelet decomposition
    [C, L] = wavedec(audio(:, ch), level, options.Wavelet);

    % Threshold small coefficients
    threshold = options.ThresholdFraction * max(abs(C));
    C_compressed = C;
    C_compressed(abs(C) < threshold) = 0;

    % Store only non-zero coefficients and their indices
    nonZeroIdx = find(C_compressed ~= 0);
    nonZeroValues = C_compressed(nonZeroIdx);

    compressed.Channels{ch} = struct('Indices', nonZeroIdx, ...
                                     'Values', nonZeroValues, ...
                                     'L', L);
end

% Calculate compression ratio
originalSize = nSamples * nChannels * 8;  % bytes (64-bit double)
compressedSize = 0;
for ch = 1:nChannels
    compressedSize = compressedSize + length(compressed.Channels{ch}.Values) * 8;
end

compressionRatio = originalSize / compressedSize;
end

function audio = decompressAudio(compressed)
% Decompress wavelet-compressed audio

nChannels = compressed.NumChannels;
audio = zeros(compressed.OriginalLength, nChannels);

for ch = 1:nChannels
    channelData = compressed.Channels{ch};

    % Reconstruct coefficient array
    C = zeros(1, sum(channelData.L));
    C(channelData.Indices) = channelData.Values;

    % Wavelet reconstruction
    reconstructed = waverec(C, channelData.L, compressed.Wavelet);

    % Ensure correct length
    audio(:, ch) = reconstructed(1:compressed.OriginalLength);
end
end

%% Visualization Methods

function plotScalogram(cfs, f, t, varargin)
% Plot wavelet scalogram

p = inputParser;
addParameter(p, 'Title', 'Wavelet Scalogram', @ischar);
addParameter(p, 'ColorMap', 'parula', @ischar);
parse(p, varargin{:});

options = p.Results;

figure;
imagesc(t, f, abs(cfs));
axis xy;
set(gca, 'YScale', 'log');
colormap(options.ColorMap);
colorbar;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title(options.Title);
end

function plotWaveletDecomposition(audio, fs, varargin)
% Plot multi-level wavelet decomposition

p = inputParser;
addParameter(p, 'Wavelet', 'db4', @ischar);
addParameter(p, 'Level', 5, @isnumeric);
parse(p, varargin{:});

options = p.Results;

[approximation, details] = multiscaleAnalysis(audio, fs, ...
    'Wavelet', options.Wavelet, 'Level', options.Level);

figure;
numPlots = options.Level + 2;

% Original signal
subplot(numPlots, 1, 1);
t = (0:length(audio)-1) / fs;
plot(t, audio);
title('Original Signal');
ylabel('Amplitude');
grid on;

% Detail levels
for i = 1:options.Level
    subplot(numPlots, 1, i+1);
    plot(t, details{i});
    title(sprintf('Detail Level %d', i));
    ylabel('Amplitude');
    grid on;
end

% Approximation
subplot(numPlots, 1, numPlots);
plot(t, approximation);
title('Approximation');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
end

function plotCoherence(wcoh, f, t, varargin)
% Plot wavelet coherence

p = inputParser;
addParameter(p, 'Title', 'Wavelet Coherence', @ischar);
parse(p, varargin{:});

options = p.Results;

figure;
imagesc(t, f, wcoh);
axis xy;
set(gca, 'YScale', 'log');
colormap('jet');
colorbar;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title(options.Title);
caxis([0 1]);
end

%% Utility Methods

function wavelets = listWavelets()
% List available wavelet families

wavelets = struct();
wavelets.Daubechies = {'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db8', 'db10'};
wavelets.Coiflets = {'coif1', 'coif2', 'coif3', 'coif4', 'coif5'};
wavelets.Symlets = {'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym8'};
wavelets.Biorthogonal = {'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior4.4', 'bior6.8'};
wavelets.ReverseBiorthogonal = {'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio3.1'};
wavelets.DiscreteMeyer = {'dmey'};
wavelets.Analytic = {'amor', 'morse'};  % For CWT

fprintf('Available Wavelet Families:\n');
fprintf('==========================\n');
fields = fieldnames(wavelets);
for i = 1:length(fields)
    fprintf('%s: %s\n', fields{i}, strjoin(wavelets.(fields{i}), ', '));
end
end

function info = getInfo(processor)
% Get processor information

info = struct();
info.HasWaveletToolbox = processor.HasWaveletToolbox;
info.Version = processor.Version;

if processor.HasWaveletToolbox
    info.AvailableWavelets = listWavelets();
    info.Capabilities = {
        'Wavelet denoising (wdenoise)', ...
        'Continuous wavelet transform (cwt)', ...
        'Wavelet synchrosqueezing (wsst)', ...
        'Wavelet coherence (wcoherence)', ...
        'Maximal overlap DWT (modwt)', ...
        'Wavelet packet analysis (wpspectrum)', ...
        'Component separation', ...
        'Wavelet compression'
    };
else
    info.Capabilities = {
        'Basic wavelet denoising', ...
        'Basic wavelet decomposition', ...
        'Limited functionality without Wavelet Toolbox'
    };
end
end
