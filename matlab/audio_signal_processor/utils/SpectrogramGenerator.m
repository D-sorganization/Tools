function [S, F, T] = SpectrogramGenerator(audioData, varargin)
%SPECTROGRAMGENERATOR Generate spectrogram with configurable parameters
%
%   [S, F, T] = SPECTROGRAMGENERATOR(AUDIODATA) generates a spectrogram of the
%   input audio data using default parameters.
%
%   [S, F, T] = SPECTROGRAMGENERATOR(AUDIODATA, 'Property', Value, ...)
%   specifies additional spectrogram parameters using property-value pairs.
%
%   Input Arguments:
%   ---------------
%   AUDIODATA - Audio data matrix (samples x channels)
%
%   Optional Properties:
%   ------------------
%   'SampleRate'      - Sample rate in Hz (default: 44100)
%   'Window'          - Window function: 'hann', 'hamming', 'blackman', 'kaiser' (default: 'hann')
%   'WindowLength'    - Window length in samples (default: 1024)
%   'Overlap'         - Overlap between windows 0-1 (default: 0.75)
%   'NFFT'            - FFT length (default: 1024)
%   'FrequencyRange'  - Frequency range: 'full', 'audible', [low, high] (default: 'audible')
%   'Normalize'       - Normalize spectrogram (default: true)
%   'LogScale'        - Use logarithmic frequency scale (default: false)
%   'DBScale'         - Convert to dB scale (default: true)
%   'MinDB'           - Minimum dB value (default: -80)
%
%   Output Arguments:
%   ----------------
%   S - Spectrogram matrix (frequencies x time)
%   F - Frequency vector in Hz
%   T - Time vector in seconds
%
%   Example:
%   --------
%   % Load audio data
%   [data, fs] = audioread('song.wav');
%
%   % Generate basic spectrogram
%   [S, F, T] = SpectrogramGenerator(data, 'SampleRate', fs);
%
%   % Generate high-resolution spectrogram
%   [S, F, T] = SpectrogramGenerator(data, 'SampleRate', fs, ...
%                                   'WindowLength', 2048, 'NFFT', 4096);
%
%   % Generate spectrogram with custom frequency range
%   [S, F, T] = SpectrogramGenerator(data, 'SampleRate', fs, ...
%                                   'FrequencyRange', [20, 20000]);
%
%   See also: spectrogram, stft, pspectrum

arguments
    audioData (:,:) double
    options.SampleRate (1,1) double {mustBePositive} = 44100
    options.Window (1,1) string {mustBeMember(options.Window, ["hann", "hamming", "blackman", "kaiser", "rectangular"])} = "hann"
    options.WindowLength (1,1) double {mustBePositive, mustBeInteger} = 1024
    options.Overlap (1,1) double {mustBeInRange(options.Overlap, 0, 0.99)} = 0.75
    options.NFFT (1,1) double {mustBePositive, mustBeInteger} = 1024
    options.FrequencyRange (1,:) = "audible"
    options.Normalize (1,1) logical = true
    options.LogScale (1,1) logical = false
    options.DBScale (1,1) logical = true
    options.MinDB (1,1) double = -80
end

% Validate input
if isempty(audioData)
    error('SpectrogramGenerator:EmptyInput', 'Input audio data is empty');
end

% Convert to mono if stereo
if size(audioData, 2) > 1
    audioData = mean(audioData, 2);
end

% Calculate overlap in samples
overlapSamples = round(options.WindowLength * options.Overlap);

% Generate spectrogram using MATLAB's spectrogram function
try
    % Build a concrete window vector from the requested type and length
    windowVector = constructWindow(options.Window, options.WindowLength);

    % MATLAB's spectrogram expects the window vector (or its length),
    % not a separate window-type argument
    [S, F, T] = spectrogram(audioData, windowVector, overlapSamples, ...
        options.NFFT, options.SampleRate);
catch ME
    error('SpectrogramGenerator:SpectrogramError', ...
        'Error generating spectrogram: %s', ME.message);
end

% Apply frequency range filtering
[S, F] = applyFrequencyRange(S, F, options.FrequencyRange);

% Apply logarithmic frequency scale if requested
if options.LogScale
    [S, F] = applyLogFrequencyScale(S, F);
end

% Convert to dB scale if requested
if options.DBScale
    S = convertToDB(S, options.MinDB);
end

% Normalize if requested
if options.Normalize && ~options.DBScale
    S = S / max(S(:));
end
end

function [S_filtered, F_filtered] = applyFrequencyRange(S, F, frequencyRange)
% Apply frequency range filtering

if ischar(frequencyRange) || isstring(frequencyRange)
    switch lower(frequencyRange)
        case 'full'
            % Keep all frequencies
            S_filtered = S;
            F_filtered = F;

        case 'audible'
            % Human audible range (20 Hz to 20 kHz)
            audibleMask = (F >= 20) & (F <= 20000);
            S_filtered = S(audibleMask, :);
            F_filtered = F(audibleMask);

        otherwise
            error('SpectrogramGenerator:InvalidRange', ...
                'Invalid frequency range: %s', frequencyRange);
    end
elseif isnumeric(frequencyRange) && length(frequencyRange) == 2
    % Custom frequency range
    lowFreq = frequencyRange(1);
    highFreq = frequencyRange(2);

    if lowFreq >= highFreq
        error('SpectrogramGenerator:InvalidRange', ...
            'Low frequency must be less than high frequency');
    end

    rangeMask = (F >= lowFreq) & (F <= highFreq);
    S_filtered = S(rangeMask, :);
    F_filtered = F(rangeMask);
else
    error('SpectrogramGenerator:InvalidRange', ...
        'Invalid frequency range specification');
end
end

function [S_log, F_log] = applyLogFrequencyScale(S, F)
% Apply logarithmic frequency scale

% Create logarithmic frequency vector
minFreq = min(F(F > 0));
maxFreq = max(F);
nFreqs = length(F);

F_log = logspace(log10(minFreq), log10(maxFreq), nFreqs);

% Interpolate spectrogram to logarithmic scale
S_log = zeros(length(F_log), size(S, 2));

for i = 1:size(S, 2)
    S_log(:, i) = interp1(F, S(:, i), F_log, 'linear', 'extrap');
end
end

function S_db = convertToDB(S, minDB)
% Convert spectrogram to dB scale

% Avoid log of zero
S(S <= 0) = eps;

% Convert to dB
S_db = 20 * log10(abs(S));

% Apply minimum dB threshold
S_db(S_db < minDB) = minDB;
end

function windowVector = constructWindow(windowType, windowLength)
% Construct a window vector from type and length

switch lower(string(windowType))
    case "hann"
        windowVector = hann(windowLength);
    case "hamming"
        windowVector = hamming(windowLength);
    case "blackman"
        windowVector = blackman(windowLength);
    case "kaiser"
        % Default beta chosen for general-purpose spectral analysis
        defaultBeta = 8;
        windowVector = kaiser(windowLength, defaultBeta);
    case "rectangular"
        windowVector = ones(windowLength, 1);
    otherwise
        error('SpectrogramGenerator:InvalidWindow', 'Unsupported window type: %s', windowType);
end
end
