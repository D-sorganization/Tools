function filteredData = AudioFilterEngine(audioData, filterType, varargin)
%AUDIOFILTERENGINE Unified audio filter interface supporting multiple filter types
%
%   FILTEREDDATA = AUDIOFILTERENGINE(AUDIODATA, FILTERTYPE) applies the specified
%   filter type to the input audio data.
%
%   FILTEREDDATA = AUDIOFILTERENGINE(AUDIODATA, FILTERTYPE, 'Property', Value, ...)
%   specifies additional filter parameters using property-value pairs.
%
%   Input Arguments:
%   ---------------
%   AUDIODATA - Audio data matrix (samples x channels)
%   FILTERTYPE - Filter type string
%
%   Supported Filter Types:
%   ----------------------
%   FFT Filters (ported from Python):
%   - 'FFT Low-pass', 'FFT High-pass', 'FFT Band-pass', 'FFT Band-stop'
%
%   MATLAB Built-in Filters:
%   - 'Butterworth Low-pass', 'Butterworth High-pass', 'Butterworth Band-pass'
%   - 'Chebyshev1 Low-pass', 'Chebyshev1 High-pass', 'Chebyshev1 Band-pass'
%   - 'Chebyshev2 Low-pass', 'Chebyshev2 High-pass', 'Chebyshev2 Band-pass'
%   - 'Elliptic Low-pass', 'Elliptic High-pass', 'Elliptic Band-pass'
%   - 'FIR Low-pass', 'FIR High-pass', 'FIR Band-pass'
%
%   Time-domain Filters:
%   - 'Moving Average', 'Median', 'Gaussian'
%
%   Optional Properties:
%   ------------------
%   'SampleRate'      - Sample rate in Hz (default: 44100)
%   'Order'           - Filter order (default: 4)
%   'CutoffFreq'      - Cutoff frequency in Hz (default: 1000)
%   'HighCutoffFreq'  - High cutoff for band filters in Hz (default: 2000)
%   'PassbandRipple'  - Passband ripple in dB (default: 1)
%   'StopbandAtten'   - Stopband attenuation in dB (default: 60)
%   'WindowSize'      - Window size for time-domain filters (default: 10)
%   'Sigma'           - Sigma for Gaussian filter (default: 1.0)
%
%   FFT Filter Properties (when using FFT filters):
%   'WindowShape'     - Window function (default: 'Gaussian')
%   'TransitionBW'    - Transition bandwidth (default: 0.05)
%   'ZeroPhase'       - Zero-phase filtering (default: true)
%
%   Output Arguments:
%   ----------------
%   FILTEREDDATA - Filtered audio data matrix (same size as input)
%
%   Example:
%   --------
%   % Load audio data
%   [data, fs] = audioread('song.wav');
%
%   % Use FFT low-pass filter
%   filtered = AudioFilterEngine(data, 'FFT Low-pass', 'CutoffFreq', 1000, 'SampleRate', fs);
%
%   % Use MATLAB built-in Butterworth filter
%   filtered = AudioFilterEngine(data, 'Butterworth Low-pass', 'CutoffFreq', 1000, 'SampleRate', fs);
%
%   % Use time-domain moving average
%   filtered = AudioFilterEngine(data, 'Moving Average', 'WindowSize', 20);
%
%   See also: FFTFilters, designfilt, fdesign, filter, filtfilt

arguments
    audioData (:,:) double
    filterType (1,1) string
    options.SampleRate (1,1) double {mustBePositive} = 44100
    options.Order (1,1) double {mustBePositive} = 4
    options.CutoffFreq (1,1) double {mustBePositive} = 1000
    options.HighCutoffFreq (1,1) double {mustBePositive} = 2000
    options.PassbandRipple (1,1) double {mustBePositive} = 1
    options.StopbandAtten (1,1) double {mustBePositive} = 60
    options.WindowSize (1,1) double {mustBePositive} = 10
    options.Sigma (1,1) double {mustBePositive} = 1.0
    options.WindowShape (1,1) string {mustBeMember(options.WindowShape, ...
        ["Gaussian", "Rectangular", "Hamming", "Hann", "Blackman", "Kaiser", "Tukey", "Bartlett"])} = "Gaussian"
    options.TransitionBW (1,1) double {mustBeInRange(options.TransitionBW, 0.001, 0.2)} = 0.05
    options.ZeroPhase (1,1) logical = true
end

% Validate input
if isempty(audioData)
    error('AudioFilterEngine:EmptyInput', 'Input audio data is empty');
end

% Determine filter category and apply appropriate filter
if startsWith(filterType, 'FFT')
    filteredData = applyFFTFilter(audioData, filterType, options);
elseif contains(filterType, 'Butterworth') || contains(filterType, 'Chebyshev') || ...
        contains(filterType, 'Elliptic') || contains(filterType, 'FIR')
    filteredData = applyMATLABBuiltInFilter(audioData, filterType, options);
elseif contains(filterType, 'Moving Average') || contains(filterType, 'Median') || ...
        contains(filterType, 'Gaussian')
    filteredData = applyTimeDomainFilter(audioData, filterType, options);
else
    error('AudioFilterEngine:UnknownFilter', ...
        'Unknown filter type: %s', filterType);
end
end

function filteredData = applyFFTFilter(audioData, filterType, options)
% Apply FFT-based filters using the ported implementation

% Extract FFT-specific parameters
freqLow = options.CutoffFreq / (options.SampleRate / 2);
freqHigh = options.HighCutoffFreq / (options.SampleRate / 2);

% Apply FFT filter
filteredData = FFTFilters(audioData, filterType, ...
    'FreqLow', freqLow, ...
    'FreqHigh', freqHigh, ...
    'WindowShape', options.WindowShape, ...
    'TransitionBW', options.TransitionBW, ...
    'ZeroPhase', options.ZeroPhase, ...
    'FreqUnit', 'normalized');
end

function filteredData = applyMATLABBuiltInFilter(audioData, filterType, options)
% Apply MATLAB built-in filters using designfilt

% Determine filter design parameters
if contains(filterType, 'Low-pass')
    filterDesign = 'lowpass';
    cutoffFreq = options.CutoffFreq;
elseif contains(filterType, 'High-pass')
    filterDesign = 'highpass';
    cutoffFreq = options.CutoffFreq;
elseif contains(filterType, 'Band-pass')
    filterDesign = 'bandpass';
    cutoffFreq = [options.CutoffFreq, options.HighCutoffFreq];
else
    error('AudioFilterEngine:UnsupportedFilter', ...
        'Unsupported MATLAB filter type: %s', filterType);
end

% Design filter based on type
try
    if contains(filterType, 'Butterworth')
        Hd = designfilt(filterDesign, 'FilterOrder', options.Order, ...
            'HalfPowerFrequency', cutoffFreq, ...
            'SampleRate', options.SampleRate);
    elseif contains(filterType, 'Chebyshev1')
        Hd = designfilt(filterDesign, 'FilterOrder', options.Order, ...
            'PassbandFrequency', cutoffFreq, ...
            'PassbandRipple', options.PassbandRipple, ...
            'SampleRate', options.SampleRate);
    elseif contains(filterType, 'Chebyshev2')
        Hd = designfilt(filterDesign, 'FilterOrder', options.Order, ...
            'StopbandFrequency', cutoffFreq, ...
            'StopbandAttenuation', options.StopbandAtten, ...
            'SampleRate', options.SampleRate);
    elseif contains(filterType, 'Elliptic')
        Hd = designfilt(filterDesign, 'FilterOrder', options.Order, ...
            'PassbandFrequency', cutoffFreq, ...
            'PassbandRipple', options.PassbandRipple, ...
            'StopbandAttenuation', options.StopbandAtten, ...
            'SampleRate', options.SampleRate);
    elseif contains(filterType, 'FIR')
        Hd = designfilt(filterDesign, 'FilterOrder', options.Order, ...
            'CutoffFrequency', cutoffFreq, ...
            'SampleRate', options.SampleRate);
    end

    % Apply filter to each channel
    filteredData = zeros(size(audioData));
    for ch = 1:size(audioData, 2)
        filteredData(:, ch) = filtfilt(Hd, audioData(:, ch));
    end

catch ME
    error('AudioFilterEngine:FilterDesignError', ...
        'Error designing MATLAB filter: %s', ME.message);
end
end

function filteredData = applyTimeDomainFilter(audioData, filterType, options)
% Apply time-domain filters

filteredData = zeros(size(audioData));

for ch = 1:size(audioData, 2)
    signal = audioData(:, ch);

    switch filterType
        case 'Moving Average'
            % Moving average filter
            windowSize = round(options.WindowSize);
            if windowSize > length(signal)
                windowSize = length(signal);
            end

            % Use MATLAB's filter function for moving average
            b = ones(1, windowSize) / windowSize;
            a = 1;
            filteredData(:, ch) = filter(b, a, signal);

        case 'Median'
            % Median filter
            windowSize = round(options.WindowSize);
            if mod(windowSize, 2) == 0
                windowSize = windowSize + 1; % Ensure odd window size
            end

            filteredData(:, ch) = medfilt1(signal, windowSize);

        case 'Gaussian'
            % Gaussian filter
            sigma = options.Sigma;
            filteredData(:, ch) = imgaussfilt(signal, sigma);

        otherwise
            error('AudioFilterEngine:UnknownTimeDomainFilter', ...
                'Unknown time-domain filter: %s', filterType);
    end
end
end
