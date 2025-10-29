function analyzer = FrequencyAnalyzer(audioData, sampleRate, varargin)
%FREQUENCYANALYZER FFT spectrum analyzer with peak detection and analysis
%
%   ANALYZER = FREQUENCYANALYZER(AUDIODATA, SAMPLERATE) creates a frequency
%   analyzer for the input audio data.
%
%   ANALYZER = FREQUENCYANALYZER(AUDIODATA, SAMPLERATE, 'Property', Value, ...)
%   specifies additional analysis parameters using property-value pairs.
%
%   Input Arguments:
%   ---------------
%   AUDIODATA - Audio data matrix (samples x channels)
%   SAMPLERATE - Sample rate in Hz
%
%   Optional Properties:
%   ------------------
%   'Window'          - Window function: 'hann', 'hamming', 'blackman', 'kaiser' (default: 'hann')
%   'WindowLength'    - Window length in samples (default: 4096)
%   'Overlap'         - Overlap between windows 0-1 (default: 0.5)
%   'NFFT'            - FFT length (default: 4096)
%   'Smoothing'       - Apply smoothing to spectrum (default: false)
%   'SmoothingFactor' - Smoothing factor 0-1 (default: 0.1)
%   'PeakDetection'   - Enable peak detection (default: true)
%   'PeakThreshold'   - Peak detection threshold in dB (default: -40)
%   'MinPeakDistance' - Minimum distance between peaks in Hz (default: 50)
%
%   Properties:
%   ----------
%   AudioData - Input audio data
%   SampleRate - Sample rate in Hz
%   Spectrum - Frequency spectrum
%   Frequencies - Frequency vector
%   Peaks - Detected frequency peaks
%   RMS - RMS spectrum
%   PeakFrequencies - Peak frequencies
%   PeakMagnitudes - Peak magnitudes
%
%   Methods:
%   --------
%   analyze() - Perform frequency analysis
%   detectPeaks() - Detect frequency peaks
%   getSpectrum() - Get frequency spectrum
%   getRMS() - Get RMS spectrum
%   getPeaks() - Get detected peaks
%   plotSpectrum() - Plot frequency spectrum
%   plotPeaks() - Plot spectrum with peaks
%
%   Example:
%   --------
%   % Load audio data
%   [data, fs] = audioread('song.wav');
%
%   % Create frequency analyzer
%   analyzer = FrequencyAnalyzer(data, fs);
%
%   % Perform analysis
%   analyzer.analyze();
%
%   % Get spectrum
%   [spectrum, frequencies] = analyzer.getSpectrum();
%
%   % Detect peaks
%   peaks = analyzer.detectPeaks();
%
%   % Plot results
%   analyzer.plotSpectrum();
%
%   See also: SpectrogramGenerator, fft, pspectrum

arguments
    audioData (:,:) double
    sampleRate (1,1) double {mustBePositive}
    options.Window (1,1) string {mustBeMember(options.Window, ["hann", "hamming", "blackman", "kaiser", "rectangular"])} = "hann"
    options.WindowLength (1,1) double {mustBePositive, mustBeInteger} = 4096
    options.Overlap (1,1) double {mustBeInRange(options.Overlap, 0, 0.99)} = 0.5
    options.NFFT (1,1) double {mustBePositive, mustBeInteger} = 4096
    options.Smoothing (1,1) logical = false
    options.SmoothingFactor (1,1) double {mustBeInRange(options.SmoothingFactor, 0, 1)} = 0.1
    options.PeakDetection (1,1) logical = true
    options.PeakThreshold (1,1) double = -40
    options.MinPeakDistance (1,1) double {mustBePositive} = 50
end

% Validate input
if isempty(audioData)
    error('FrequencyAnalyzer:EmptyInput', 'Input audio data is empty');
end

% Initialize analyzer structure
analyzer = struct();
analyzer.AudioData = audioData;
analyzer.SampleRate = sampleRate;
analyzer.Spectrum = [];
analyzer.Frequencies = [];
analyzer.Peaks = struct();
analyzer.RMS = [];
analyzer.PeakFrequencies = [];
analyzer.PeakMagnitudes = [];

% Store analysis parameters
analyzer.Parameters = struct();
analyzer.Parameters.Window = options.Window;
analyzer.Parameters.WindowLength = options.WindowLength;
analyzer.Parameters.Overlap = options.Overlap;
analyzer.Parameters.NFFT = options.NFFT;
analyzer.Parameters.Smoothing = options.Smoothing;
analyzer.Parameters.SmoothingFactor = options.SmoothingFactor;
analyzer.Parameters.PeakDetection = options.PeakDetection;
analyzer.Parameters.PeakThreshold = options.PeakThreshold;
analyzer.Parameters.MinPeakDistance = options.MinPeakDistance;

% Convert to class-like structure with methods
analyzer.analyze = @() analyze(analyzer);
analyzer.detectPeaks = @() detectPeaks(analyzer);
analyzer.getSpectrum = @() getSpectrum(analyzer);
analyzer.getRMS = @() getRMS(analyzer);
analyzer.getPeaks = @() getPeaks(analyzer);
analyzer.plotSpectrum = @() plotSpectrum(analyzer);
analyzer.plotPeaks = @() plotPeaks(analyzer);
end

function analyze(analyzer)
% Perform frequency analysis

% Convert to mono if stereo
audioData = analyzer.AudioData;
if size(audioData, 2) > 1
    audioData = mean(audioData, 2);
end

% Calculate window parameters
windowLength = analyzer.Parameters.WindowLength;
overlap = analyzer.Parameters.Overlap;
nfft = analyzer.Parameters.NFFT;
sampleRate = analyzer.SampleRate;

% Calculate overlap in samples
overlapSamples = round(windowLength * overlap);

% Generate spectrogram
[S, F, T] = spectrogram(audioData, windowLength, overlapSamples, ...
    nfft, sampleRate, analyzer.Parameters.Window);

% Calculate average spectrum
analyzer.Spectrum = mean(abs(S), 2);
analyzer.Frequencies = F;

% Calculate RMS spectrum
analyzer.RMS = sqrt(mean(S .* conj(S), 2));

% Apply smoothing if requested
if analyzer.Parameters.Smoothing
    smoothingFactor = analyzer.Parameters.SmoothingFactor;
    analyzer.Spectrum = smoothSpectrum(analyzer.Spectrum, smoothingFactor);
    analyzer.RMS = smoothSpectrum(analyzer.RMS, smoothingFactor);
end

% Detect peaks if requested
if analyzer.Parameters.PeakDetection
    analyzer.detectPeaks();
end
end

function peaks = detectPeaks(analyzer)
% Detect frequency peaks in the spectrum

if isempty(analyzer.Spectrum)
    error('FrequencyAnalyzer:NoSpectrum', 'Must call analyze() before detectPeaks()');
end

% Convert spectrum to dB
spectrumDB = 20 * log10(abs(analyzer.Spectrum) + eps);

% Find peaks using findpeaks
[peakValues, peakIndices] = findpeaks(spectrumDB, ...
    'MinPeakHeight', analyzer.Parameters.PeakThreshold, ...
    'MinPeakDistance', round(analyzer.Parameters.MinPeakDistance * length(analyzer.Spectrum) / (analyzer.SampleRate / 2)));

% Convert indices to frequencies
analyzer.PeakFrequencies = analyzer.Frequencies(peakIndices);
analyzer.PeakMagnitudes = peakValues;

% Create peaks structure
peaks = struct();
peaks.Frequencies = analyzer.PeakFrequencies;
peaks.Magnitudes = analyzer.PeakMagnitudes;
peaks.Count = length(analyzer.PeakFrequencies);

% Find fundamental frequency (lowest peak)
if ~isempty(analyzer.PeakFrequencies)
    [~, fundamentalIdx] = min(analyzer.PeakFrequencies);
    peaks.FundamentalFreq = analyzer.PeakFrequencies(fundamentalIdx);
    peaks.FundamentalMag = analyzer.PeakMagnitudes(fundamentalIdx);
else
    peaks.FundamentalFreq = [];
    peaks.FundamentalMag = [];
end

% Find dominant frequency (highest peak)
if ~isempty(analyzer.PeakFrequencies)
    [~, dominantIdx] = max(analyzer.PeakMagnitudes);
    peaks.DominantFreq = analyzer.PeakFrequencies(dominantIdx);
    peaks.DominantMag = analyzer.PeakMagnitudes(dominantIdx);
else
    peaks.DominantFreq = [];
    peaks.DominantMag = [];
end

analyzer.Peaks = peaks;
end

function [spectrum, frequencies] = getSpectrum(analyzer)
% Get frequency spectrum

if isempty(analyzer.Spectrum)
    error('FrequencyAnalyzer:NoSpectrum', 'Must call analyze() before getSpectrum()');
end

spectrum = analyzer.Spectrum;
frequencies = analyzer.Frequencies;
end

function rms = getRMS(analyzer)
% Get RMS spectrum

if isempty(analyzer.RMS)
    error('FrequencyAnalyzer:NoRMS', 'Must call analyze() before getRMS()');
end

rms = analyzer.RMS;
end

function peaks = getPeaks(analyzer)
% Get detected peaks

if isempty(analyzer.Peaks)
    error('FrequencyAnalyzer:NoPeaks', 'Must call detectPeaks() before getPeaks()');
end

peaks = analyzer.Peaks;
end

function plotSpectrum(analyzer)
% Plot frequency spectrum

if isempty(analyzer.Spectrum)
    error('FrequencyAnalyzer:NoSpectrum', 'Must call analyze() before plotSpectrum()');
end

figure;
semilogx(analyzer.Frequencies, 20 * log10(abs(analyzer.Spectrum) + eps));
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Frequency Spectrum');
grid on;
xlim([20, analyzer.SampleRate/2]);
end

function plotPeaks(analyzer)
% Plot spectrum with detected peaks

if isempty(analyzer.Spectrum)
    error('FrequencyAnalyzer:NoSpectrum', 'Must call analyze() before plotPeaks()');
end

if isempty(analyzer.Peaks)
    error('FrequencyAnalyzer:NoPeaks', 'Must call detectPeaks() before plotPeaks()');
end

figure;
semilogx(analyzer.Frequencies, 20 * log10(abs(analyzer.Spectrum) + eps));
hold on;

% Plot peaks
semilogx(analyzer.PeakFrequencies, analyzer.PeakMagnitudes, 'ro', 'MarkerSize', 8);

xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Frequency Spectrum with Peaks');
legend('Spectrum', 'Peaks', 'Location', 'best');
grid on;
xlim([20, analyzer.SampleRate/2]);
end

function smoothedSpectrum = smoothSpectrum(spectrum, smoothingFactor)
% Apply smoothing to spectrum

% Simple moving average smoothing
windowSize = max(3, round(length(spectrum) * smoothingFactor));
if mod(windowSize, 2) == 0
    windowSize = windowSize + 1;
end

smoothedSpectrum = movmean(spectrum, windowSize);
end
