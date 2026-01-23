function filteredData = FFTFilters(audioData, filterType, varargin)
%FFTFILTERS FFT-based frequency domain filtering with multiple window functions
%
%   FILTEREDDATA = FFTFILTERS(AUDIODATA, FILTERTYPE) applies FFT-based filtering
%   to the input audio data using the specified filter type.
%
%   FILTEREDDATA = FFTFILTERS(AUDIODATA, FILTERTYPE, 'Property', Value, ...)
%   specifies additional filter parameters using property-value pairs.
%
%   Input Arguments:
%   ---------------
%   AUDIODATA - Audio data matrix (samples x channels)
%   FILTERTYPE - Filter type: 'Low-pass', 'High-pass', 'Band-pass', 'Band-stop'
%
%   Optional Properties:
%   ------------------
%   'WindowShape'     - Window function: 'Gaussian', 'Rectangular', 'Hamming',
%                      'Hann', 'Blackman', 'Kaiser', 'Tukey', 'Bartlett' (default: 'Gaussian')
%   'FreqLow'         - Lower cutoff frequency (normalized 0-0.5) (default: 0.1)
%   'FreqHigh'        - Upper cutoff frequency (normalized 0-0.5) (default: 0.3)
%   'TransitionBW'    - Transition bandwidth (normalized) (default: 0.05)
%   'ZeroPhase'       - Use zero-phase filtering (default: true)
%   'FreqUnit'        - Frequency unit: 'normalized', 'Hz' (default: 'normalized')
%   'SampleRate'       - Sample rate in Hz (required if FreqUnit is 'Hz')
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
%   % Apply low-pass filter
%   filtered = FFTFilters(data, 'Low-pass', 'FreqLow', 0.2);
%
%   % Apply band-pass filter with Hamming window
%   filtered = FFTFilters(data, 'Band-pass', 'FreqLow', 0.1, 'FreqHigh', 0.3, ...
%                        'WindowShape', 'Hamming');
%
%   % Apply filter with Hz units
%   filtered = FFTFilters(data, 'High-pass', 'FreqHigh', 1000, ...
%                        'FreqUnit', 'Hz', 'SampleRate', fs);
%
%   See also: designfilt, fdesign, filter, filtfilt

arguments
    audioData (:,:) double
    filterType (1,1) string {mustBeMember(filterType, ["Low-pass", "High-pass", "Band-pass", "Band-stop"])}
    options.WindowShape (1,1) string {mustBeMember(options.WindowShape, ...
        ["Gaussian", "Rectangular", "Hamming", "Hann", "Blackman", "Kaiser", "Tukey", "Bartlett"])} = "Gaussian"
    options.FreqLow (1,1) double {mustBeInRange(options.FreqLow, 0, 0.5)} = 0.1
    options.FreqHigh (1,1) double {mustBeInRange(options.FreqHigh, 0, 0.5)} = 0.3
    options.TransitionBW (1,1) double {mustBeInRange(options.TransitionBW, 0.001, 0.2)} = 0.05
    options.ZeroPhase (1,1) logical = true
    options.FreqUnit (1,1) string {mustBeMember(options.FreqUnit, ["normalized", "Hz"])} = "normalized"
    options.SampleRate (1,1) double {mustBePositive} = []
end

% Validate input
if isempty(audioData)
    error('FFTFilters:EmptyInput', 'Input audio data is empty');
end

% Convert Hz to normalized if needed
if strcmp(options.FreqUnit, 'Hz')
    if isempty(options.SampleRate)
        error('FFTFilters:MissingSampleRate', ...
            'SampleRate must be specified when using Hz frequency units');
    end
    freqLow = options.FreqLow / (options.SampleRate / 2);
    freqHigh = options.FreqHigh / (options.SampleRate / 2);
    transitionBW = options.TransitionBW / (options.SampleRate / 2);
else
    freqLow = options.FreqLow;
    freqHigh = options.FreqHigh;
    transitionBW = options.TransitionBW;
end

% Ensure frequency bounds
freqLow = max(0.0, min(freqLow, 0.5));
freqHigh = max(freqLow, min(freqHigh, 0.5));

% Get signal dimensions
[nSamples, nChannels] = size(audioData);

% Design frequency window
filterCoeffs = designFrequencyWindow(filterType, freqLow, freqHigh, ...
    options.WindowShape, nSamples, transitionBW);

% Apply filter to each channel
filteredData = zeros(size(audioData));

for ch = 1:nChannels
    filteredData(:, ch) = applyFFTFilterCore(audioData(:, ch), filterCoeffs, options.ZeroPhase);
end
end

function filterCoeffs = designFrequencyWindow(filterType, freqLow, freqHigh, windowShape, nSamples, transitionBW)
% Design frequency domain window for FFT filtering
% Ported from Python implementation

% Create frequency array
freqs = abs(fftfreq(nSamples));

% Initialize filter response
filterResponse = zeros(size(freqs));

% Design ideal filter response based on filter type
switch filterType
    case "Low-pass"
        filterResponse(freqs <= freqLow) = 1.0;
        % Add transition band
        transitionMask = (freqs > freqLow) & (freqs <= freqLow + transitionBW);
        filterResponse(transitionMask) = 0.5 * (1 + cos(pi * (freqs(transitionMask) - freqLow) / transitionBW));

    case "High-pass"
        filterResponse(freqs >= freqHigh) = 1.0;
        % Add transition band
        transitionMask = (freqs >= freqHigh - transitionBW) & (freqs < freqHigh);
        filterResponse(transitionMask) = 0.5 * (1 - cos(pi * (freqs(transitionMask) - freqHigh + transitionBW) / transitionBW));

    case "Band-pass"
        filterResponse((freqs >= freqLow) & (freqs <= freqHigh)) = 1.0;
        % Add transition bands
        lowTransition = (freqs > freqLow - transitionBW) & (freqs <= freqLow);
        highTransition = (freqs >= freqHigh) & (freqs < freqHigh + transitionBW);
        filterResponse(lowTransition) = 0.5 * (1 + cos(pi * (freqs(lowTransition) - freqLow + transitionBW) / transitionBW));
        filterResponse(highTransition) = 0.5 * (1 - cos(pi * (freqs(highTransition) - freqHigh) / transitionBW));

    case "Band-stop"
        filterResponse((freqs < freqLow) | (freqs > freqHigh)) = 1.0;
        % Add transition bands
        lowTransition = (freqs >= freqLow) & (freqs < freqLow + transitionBW);
        highTransition = (freqs > freqHigh - transitionBW) & (freqs <= freqHigh);
        filterResponse(lowTransition) = 0.5 * (1 - cos(pi * (freqs(lowTransition) - freqLow) / transitionBW));
        filterResponse(highTransition) = 0.5 * (1 + cos(pi * (freqs(highTransition) - freqHigh + transitionBW) / transitionBW));
end

% Apply window function to smooth the response
if ~strcmp(windowShape, "Rectangular")
    filterResponse = applyWindowFunction(filterResponse, windowShape);
end

filterCoeffs = filterResponse;
end

function smoothedResponse = applyWindowFunction(filterResponse, windowShape)
% Apply window function to smooth frequency response
% Ported from Python implementation

n = length(filterResponse);

switch windowShape
    case "Gaussian"
        % Gaussian window
        sigma = n / 8; % Adjust sigma for smoothness
        window = exp(-0.5 * ((1:n)' - n/2).^2 / sigma^2);

    case "Hamming"
        window = hamming(n);

    case "Hann"
        window = hann(n);

    case "Blackman"
        window = blackman(n);

    case "Kaiser"
        window = kaiser(n, 8.6); % Beta for good stopband attenuation

    case "Tukey"
        window = tukeywin(n, 0.5);

    case "Bartlett"
        window = bartlett(n);

    otherwise % Rectangular or unknown
        smoothedResponse = filterResponse;
        return;
end

% Apply window smoothing using convolution
% Use FFT-based convolution for efficiency
windowFFT = fft(window);
responseFFT = fft(filterResponse);
smoothedFFT = responseFFT .* windowFFT;
smoothedResponse = real(ifft(smoothedFFT));

% Normalize to maintain magnitude
maxVal = max(smoothedResponse);
if maxVal > 0
    smoothedResponse = smoothedResponse / maxVal;
end
end

function filteredSignal = applyFFTFilterCore(signalData, filterCoeffs, zeroPhase)
% Core FFT filtering implementation
% Ported from Python implementation

% Ensure filter coefficients match signal length
if length(filterCoeffs) ~= length(signalData)
    % Interpolate filter coefficients to match signal length
    oldIndices = linspace(0, length(filterCoeffs)-1, length(filterCoeffs));
    newIndices = linspace(0, length(filterCoeffs)-1, length(signalData));
    filterCoeffs = interp1(oldIndices, filterCoeffs, newIndices, 'linear');
end

% Apply filter in frequency domain
signalFFT = fft(signalData);
filteredFFT = signalFFT .* filterCoeffs;

if zeroPhase
    % Zero-phase filtering: apply filter forward and backward
    filteredSignal = real(ifft(filteredFFT));
    % Apply filter again in reverse direction
    filteredFFTRev = fft(flip(filteredSignal));
    filteredFFTRev = filteredFFTRev .* filterCoeffs;
    filteredSignalRev = real(ifft(filteredFFTRev));
    filteredSignal = flip(filteredSignalRev);
else
    % Linear phase filtering
    filteredSignal = real(ifft(filteredFFT));
end
end

function freqs = fftfreq(n)
% Generate frequency array similar to numpy.fft.fftfreq
if mod(n, 2) == 0
    freqs = [0:n/2-1, -n/2:-1] / n;
else
    freqs = [0:(n-1)/2, -(n-1)/2:-1] / n;
end
end
