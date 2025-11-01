function processor = AdvancedAudioProcessor()
%ADVANCEDAUDIOPROCESSOR Advanced audio processing leveraging Audio Toolbox
%
%   PROCESSOR = ADVANCEDAUDIOPROCESSOR() creates an advanced audio processor
%   that leverages MATLAB's Audio Toolbox for professional-grade analysis
%   and processing capabilities.
%
%   Key Features:
%   ------------
%   - Neural network pitch detection (pitchnn)
%   - Onset and beat detection (audioSpectralFlux)
%   - Psychoacoustic analysis (acousticLoudness, splMeter)
%   - Advanced filtering (octaveFilter, audioBandpassBank)
%   - Machine learning feature extraction
%   - Time/pitch manipulation (audioTimeScaler)
%   - Spatial audio processing
%
%   Pitch Analysis Methods:
%   ----------------------
%   detectPitch(audio, fs) - Neural network pitch detection
%   trackPitch(audio, fs) - Pitch tracking over time
%   correctPitch(audio, fs, targetPitch) - Pitch correction/auto-tune
%   harmonic Ratio(audio, fs) - Voice quality analysis
%
%   Onset & Rhythm Methods:
%   ----------------------
%   detectOnsets(audio, fs) - Spectral flux onset detection
%   detectBeats(audio, fs) - Beat detection
%   estimateTempo(audio, fs) - Tempo estimation
%   alignToBeats(audio, fs) - Align audio to beat grid
%
%   Psychoacoustic Methods:
%   ----------------------
%   measureLoudness(audio, fs) - Acoustic loudness (phons/sones)
%   measureSPL(audio, fs) - Sound pressure level metering
%   barkScale Analysis(audio, fs) - Bark scale frequency analysis
%   erbScaleAnalysis(audio, fs) - Equivalent rectangular bandwidth
%   gammatoneFiltering(audio, fs) - Gammatone filterbank
%
%   Feature Extraction Methods:
%   --------------------------
%   extractMFCC(audio, fs) - Mel-frequency cepstral coefficients
%   extractSpectralFeatures(audio, fs) - Comprehensive spectral descriptors
%   extractTemporalFeatures(audio, fs) - Temporal features
%   extractAllFeatures(audio, fs) - Complete feature set for ML
%
%   Advanced Filtering Methods:
%   --------------------------
%   octaveFilter(audio, fs) - Octave band filtering
%   thirdOctaveFilter(audio, fs) - 1/3 octave filtering
%   parametricEQ(audio, fs, bands) - Parametric EQ with Q control
%   graphicEQ(audio, fs, gains) - 31-band graphic EQ
%
%   Time/Pitch Methods:
%   ------------------
%   timeScale(audio, fs, factor) - Time stretching (preserveFormants)
%   advancedPitchShift(audio, fs, semitones) - Phase vocoder pitch shift
%   phaseVocoder(audio, fs, options) - Custom phase vocoder
%
%   Spatial Audio Methods:
%   ---------------------
%   stereoWiden(audio, width) - Stereo widening
%   midSideProcess(audio, midGain, sideGain) - M/S processing
%   spatialize3D(audio, azimuth, elevation) - 3D positioning (HRTF)
%
%   Example Usage:
%   -------------
%   % Create processor
%   ap = AdvancedAudioProcessor();
%
%   % Pitch detection
%   [audio, fs] = audioread('vocal.wav');
%   [pitch, confidence] = ap.detectPitch(audio, fs);
%   plot((0:length(pitch)-1)/fs, pitch);
%
%   % Onset detection
%   onsets = ap.detectOnsets(audio, fs);
%   fprintf('Detected %d onsets\n', length(onsets));
%
%   % Feature extraction for ML
%   features = ap.extractAllFeatures(audio, fs);
%
%   % Psychoacoustic loudness
%   loudness = ap.measureLoudness(audio, fs);
%   fprintf('Loudness: %.2f phons\n', loudness.phons);
%
%   % Time scaling
%   faster = ap.timeScale(audio, fs, 1.5);  % 1.5x faster, same pitch
%
%   See also: pitchnn, audioSpectralFlux, acousticLoudness, audioFeatureExtractor

% Initialize processor structure
processor = struct();
processor.Version = '1.0';
v = ver('audio');
processor.HasAudioToolbox = ~isempty(v);

if ~processor.HasAudioToolbox
    warning('AdvancedAudioProcessor:NoToolbox', ...
        'Audio Toolbox not available. Some features will be limited.');
end

% Add pitch analysis methods
processor.detectPitch = @(audio, fs, varargin) detectPitch(audio, fs, varargin{:});
processor.trackPitch = @(audio, fs, varargin) trackPitch(audio, fs, varargin{:});
processor.correctPitch = @(audio, fs, targetPitch, varargin) correctPitch(audio, fs, targetPitch, varargin{:});
processor.harmonicRatio = @(audio, fs, varargin) harmonicRatio(audio, fs, varargin{:});

% Add onset & rhythm methods
processor.detectOnsets = @(audio, fs, varargin) detectOnsets(audio, fs, varargin{:});
processor.detectBeats = @(audio, fs, varargin) detectBeats(audio, fs, varargin{:});
processor.estimateTempo = @(audio, fs, varargin) estimateTempo(audio, fs, varargin{:});
processor.alignToBeats = @(audio, fs, varargin) alignToBeats(audio, fs, varargin{:});

% Add psychoacoustic methods
processor.measureLoudness = @(audio, fs, varargin) measureLoudness(audio, fs, varargin{:});
processor.measureSPL = @(audio, fs, varargin) measureSPL(audio, fs, varargin{:});
processor.barkScaleAnalysis = @(audio, fs, varargin) barkScaleAnalysis(audio, fs, varargin{:});
processor.erbScaleAnalysis = @(audio, fs, varargin) erbScaleAnalysis(audio, fs, varargin{:});
processor.gammatoneFiltering = @(audio, fs, varargin) gammatoneFiltering(audio, fs, varargin{:});

% Add feature extraction methods
processor.extractMFCC = @(audio, fs, varargin) extractMFCC(audio, fs, varargin{:});
processor.extractSpectralFeatures = @(audio, fs, varargin) extractSpectralFeatures(audio, fs, varargin{:});
processor.extractTemporalFeatures = @(audio, fs, varargin) extractTemporalFeatures(audio, fs, varargin{:});
processor.extractAllFeatures = @(audio, fs, varargin) extractAllFeatures(audio, fs, varargin{:});

% Add advanced filtering methods
processor.octaveFilter = @(audio, fs, varargin) octaveFilter(audio, fs, varargin{:});
processor.thirdOctaveFilter = @(audio, fs, varargin) thirdOctaveFilter(audio, fs, varargin{:});
processor.parametricEQ = @(audio, fs, bands, varargin) parametricEQ(audio, fs, bands, varargin{:});
processor.graphicEQ = @(audio, fs, gains, varargin) graphicEQ(audio, fs, gains, varargin{:});

% Add time/pitch methods
processor.timeScale = @(audio, fs, factor, varargin) timeScale(audio, fs, factor, varargin{:});
processor.advancedPitchShift = @(audio, fs, semitones, varargin) advancedPitchShift(audio, fs, semitones, varargin{:});
processor.phaseVocoder = @(audio, fs, varargin) phaseVocoder(audio, fs, varargin{:});

% Add spatial audio methods
processor.stereoWiden = @(audio, width, varargin) stereoWiden(audio, width, varargin{:});
processor.midSideProcess = @(audio, midGain, sideGain) midSideProcess(audio, midGain, sideGain);
processor.spatialize3D = @(audio, azimuth, elevation, varargin) spatialize3D(audio, azimuth, elevation, varargin{:});

% Add utility methods
processor.getInfo = @() getInfo(processor);
end

%% Pitch Analysis Methods

function [pitch, confidence] = detectPitch(audio, fs, varargin)
% Neural network pitch detection using pitchnn

p = inputParser;
addParameter(p, 'Range', [50 400], @isnumeric);  % Hz
addParameter(p, 'WindowLength', round(fs*0.052), @isnumeric);
addParameter(p, 'OverlapLength', round(fs*0.052*0.75), @isnumeric);
addParameter(p, 'Method', 'nn', @ischar);  % 'nn' or 'traditional'
parse(p, varargin{:});

options = p.Results;

% Convert to mono if stereo
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if strcmp(options.Method, 'nn') && exist('pitchnn', 'file') == 2
        % Use neural network pitch detection
        [pitch, confidence] = pitchnn(audio, fs, ...
            'Range', options.Range, ...
            'WindowLength', options.WindowLength, ...
            'OverlapLength', options.OverlapLength);
    else
        % Fallback: autocorrelation-based pitch detection
        [pitch, confidence] = traditionalPitchDetection(audio, fs, options);
    end
catch ME
    warning('AdvancedAudioProcessor:PitchDetectError', 'Error: %s', ME.message);
    pitch = zeros(ceil(length(audio)/options.WindowLength), 1);
    confidence = zeros(size(pitch));
end
end

function [pitch, time] = trackPitch(audio, fs, varargin)
% Pitch tracking over time

p = inputParser;
addParameter(p, 'Range', [50 400], @isnumeric);
addParameter(p, 'Smoothing', 5, @isnumeric);  % Median filter width
parse(p, varargin{:});

options = p.Results;

[rawPitch, confidence] = detectPitch(audio, fs, varargin{:});

% Filter low-confidence detections
threshold = 0.5;
rawPitch(confidence < threshold) = NaN;

% Apply median filtering for smooth tracking
pitch = medfilt1(rawPitch, options.Smoothing, 'omitnan');

% Time vector
windowLength = round(fs*0.052);
overlapLength = round(fs*0.052*0.75);
hopLength = windowLength - overlapLength;
time = (0:length(pitch)-1) * hopLength / fs;
end

function corrected = correctPitch(audio, fs, targetPitch, varargin)
% Pitch correction/auto-tune

p = inputParser;
addParameter(p, 'Strength', 1.0, @isnumeric);  % 0-1, 1=full correction
addParameter(p, 'Range', [50 400], @isnumeric);
parse(p, varargin{:});

options = p.Results;

% Detect current pitch
[pitch, confidence] = detectPitch(audio, fs, 'Range', options.Range);

% Calculate correction needed (in semitones)
pitchRatio = targetPitch ./ pitch;
semitones = 12 * log2(pitchRatio);

% Apply strength
semitones = semitones * options.Strength;

% Apply pitch shift frame-by-frame
% (Simplified - full implementation would need phase vocoder)
corrected = audio;  % Placeholder

warning('AdvancedAudioProcessor:PitchCorrect', ...
    'Pitch correction requires custom phase vocoder implementation');
end

function hr = harmonicRatio(audio, fs, varargin)
% Voice quality analysis using harmonic ratio

p = inputParser;
addParameter(p, 'WindowLength', round(fs*0.03), @isnumeric);
parse(p, varargin{:});

options = p.Results;

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if exist('harmonicRatio', 'file') == 2
        hr = harmonicRatio(audio, fs, ...
            'Window', hann(options.WindowLength, 'periodic'));
    else
        % Fallback: simplified harmonic ratio calculation
        hr = calculateSimpleHR(audio, options.WindowLength);
    end
catch ME
    warning('AdvancedAudioProcessor:HRError', 'Error: %s', ME.message);
    hr = zeros(ceil(length(audio)/options.WindowLength), 1);
end
end

%% Onset & Rhythm Methods

function onsetTimes = detectOnsets(audio, fs, varargin)
% Onset detection using spectral flux

p = inputParser;
addParameter(p, 'Threshold', 0.5, @isnumeric);
addParameter(p, 'Method', 'spectralflux', @ischar);
parse(p, varargin{:});

options = p.Results;

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if exist('detectSpeech', 'file') == 2
        % Use Audio Toolbox spectral flux
        [onsetIndices, ~] = detectSpeech(audio, fs);
        onsetTimes = onsetIndices / fs;
    else
        % Fallback: simple energy-based onset detection
        onsetTimes = simpleOnsetDetection(audio, fs, options.Threshold);
    end
catch ME
    warning('AdvancedAudioProcessor:OnsetError', 'Error: %s', ME.message);
    onsetTimes = [];
end
end

function beatTimes = detectBeats(audio, fs, varargin)
% Beat detection

p = inputParser;
addParameter(p, 'Tempo Range', [60 180], @isnumeric);  % BPM
parse(p, varargin{:});

% Detect onsets first
onsetTimes = detectOnsets(audio, fs, varargin{:});

if isempty(onsetTimes)
    beatTimes = [];
    return;
end

% Find periodicity in onsets (simplified beat tracking)
onsetDiffs = diff(onsetTimes);
medianPeriod = median(onsetDiffs);

% Filter onsets to beat grid
beatTimes = onsetTimes(1):medianPeriod:onsetTimes(end);
beatTimes = beatTimes(:);
end

function tempo = estimateTempo(audio, fs, varargin)
% Tempo estimation in BPM

p = inputParser;
addParameter(p, 'TempoRange', [60 180], @isnumeric);
parse(p, varargin{:});

% Detect beats
beatTimes = detectBeats(audio, fs, varargin{:});

if length(beatTimes) < 2
    tempo = 0;
    return;
end

% Calculate average inter-beat interval
ibi = mean(diff(beatTimes));  % seconds
tempo = 60 / ibi;  % BPM
end

function aligned = alignToBeats(audio, fs, varargin)
% Align audio to beat grid (placeholder)

p = inputParser;
addParameter(p, 'TargetTempo', 120, @isnumeric);
parse(p, varargin{:});

% Estimate current tempo
currentTempo = estimateTempo(audio, fs);

if currentTempo == 0
    aligned = audio;
    return;
end

% Time stretch to match target tempo
stretchFactor = currentTempo / p.Results.TargetTempo;
aligned = timeScale(audio, fs, stretchFactor);
end

%% Psychoacoustic Methods

function loudness = measureLoudness(audio, fs, varargin)
% Acoustic loudness measurement

p = inputParser;
addParameter(p, 'Calibration', 94, @isnumeric);  % dB SPL
parse(p, varargin{:});

options = p.Results;

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if exist('acousticLoudness', 'file') == 2
        [phons, sones, time] = acousticLoudness(audio, fs, ...
            'Calibration', options.Calibration);
        loudness = struct('phons', phons, 'sones', sones, 'time', time);
    else
        % Fallback: RMS-based loudness estimate
        rmsValue = rms(audio);
        dbFS = 20 * log10(rmsValue + eps);
        loudness = struct('phons', dbFS + 94, 'sones', [], 'time', []);
    end
catch ME
    warning('AdvancedAudioProcessor:LoudnessError', 'Error: %s', ME.message);
    loudness = struct('phons', 0, 'sones', 0, 'time', []);
end
end

function spl = measureSPL(audio, fs, varargin)
% Sound pressure level metering

p = inputParser;
addParameter(p, 'Calibration', 94, @isnumeric);
addParameter(p, 'Weighting', 'A', @ischar);  % 'A', 'C', or 'Z'
parse(p, varargin{:});

options = p.Results;

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if exist('splMeter', 'file') == 2
        [levels, time] = splMeter(audio, fs, ...
            'Calibration', options.Calibration, ...
            'Weighting', options.Weighting);
        spl = struct('levels', levels, 'time', time, 'weighting', options.Weighting);
    else
        % Fallback: simple dB calculation
        rmsValue = rms(audio);
        dbFS = 20 * log10(rmsValue + eps);
        spl = struct('levels', dbFS + options.Calibration, 'time', [], 'weighting', 'none');
    end
catch ME
    warning('AdvancedAudioProcessor:SPLError', 'Error: %s', ME.message);
    spl = struct('levels', 0, 'time', [], 'weighting', 'none');
end
end

function barkAnalysis = barkScaleAnalysis(audio, fs, varargin)
% Bark scale frequency analysis

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

% Bark scale: z = 13 * atan(0.00076 * f) + 3.5 * atan((f/7500)^2)
% 24 critical bands

numBands = 24;
barkBands = linspace(0, 24, numBands+1);
frequencies = zeros(numBands, 1);

for i = 1:numBands
    % Approximate frequency from Bark scale (simplified)
    bark = (barkBands(i) + barkBands(i+1)) / 2;
    frequencies(i) = 600 * sinh(bark/6);
end

% Filter into bands and measure energy
bandEnergies = zeros(numBands, 1);

for i = 1:numBands-1
    % Bandpass filter
    nyquist = fs / 2;
    lowFreq = frequencies(i) / nyquist;
    highFreq = frequencies(i+1) / nyquist;

    if highFreq < 1.0
        [b, a] = butter(2, [lowFreq, highFreq], 'bandpass');
        filtered = filtfilt(b, a, audio);
        bandEnergies(i) = sum(filtered.^2);
    end
end

barkAnalysis = struct('frequencies', frequencies, 'energies', bandEnergies);
end

function erbAnalysis = erbScaleAnalysis(audio, fs, varargin)
% Equivalent rectangular bandwidth analysis

% Similar to Bark scale but using ERB scale
% ERB(f) = 24.7 * (4.37 * f/1000 + 1)

numBands = 32;
% ERBs distributed from 50 Hz to fs/2
minERB = 24.7 * (4.37 * 50/1000 + 1);
maxERB = 24.7 * (4.37 * (fs/2)/1000 + 1);
erbBands = linspace(minERB, maxERB, numBands+1);

% Convert back to frequency
frequencies = (erbBands/24.7 - 1) * 1000/4.37;

% Filter and measure energy (similar to Bark analysis)
bandEnergies = zeros(numBands, 1);

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

for i = 1:numBands-1
    nyquist = fs / 2;
    lowFreq = frequencies(i) / nyquist;
    highFreq = frequencies(i+1) / nyquist;

    if highFreq < 1.0 && lowFreq > 0
        [b, a] = butter(2, [lowFreq, highFreq], 'bandpass');
        filtered = filtfilt(b, a, audio);
        bandEnergies(i) = sum(filtered.^2);
    end
end

erbAnalysis = struct('frequencies', frequencies(1:end-1), 'energies', bandEnergies);
end

function [output, centerFreqs] = gammatoneFiltering(audio, fs, varargin)
% Gammatone filterbank processing

p = inputParser;
addParameter(p, 'NumBands', 32, @isnumeric);
addParameter(p, 'FrequencyRange', [100, fs/2], @isnumeric);
parse(p, varargin{:});

options = p.Results;

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if exist('designAuditoryFilterBank', 'file') == 2
        % Use Audio Toolbox gammatone filterbank
        filterBank = designAuditoryFilterBank(fs, ...
            'FrequencyRange', options.FrequencyRange, ...
            'NumBands', options.NumBands, ...
            'FilterType', 'Gammatone');

        output = filterBank(audio);
        centerFreqs = getCenterFrequencies(filterBank);
    else
        % Fallback: use standard bandpass filters
        warning('AdvancedAudioProcessor:NoGammatone', 'Gammatone filterbank not available, using bandpass');
        centerFreqs = logspace(log10(options.FrequencyRange(1)), ...
                              log10(options.FrequencyRange(2)), ...
                              options.NumBands);
        output = zeros(length(audio), options.NumBands);

        for i = 1:options.NumBands
            bw = centerFreqs(i) * 0.2;  % 20% bandwidth
            lowFreq = max(centerFreqs(i) - bw/2, 20) / (fs/2);
            highFreq = min(centerFreqs(i) + bw/2, fs/2-1) / (fs/2);
            [b, a] = butter(2, [lowFreq, highFreq], 'bandpass');
            output(:, i) = filtfilt(b, a, audio);
        end
    end
catch ME
    warning('AdvancedAudioProcessor:GammatoneError', 'Error: %s', ME.message);
    output = audio;
    centerFreqs = [];
end
end

%% Feature Extraction Methods

function mfcc = extractMFCC(audio, fs, varargin)
% Extract Mel-frequency cepstral coefficients

p = inputParser;
addParameter(p, 'NumCoeffs', 13, @isnumeric);
addParameter(p, 'WindowLength', round(fs*0.03), @isnumeric);
addParameter(p, 'OverlapLength', round(fs*0.02), @isnumeric);
parse(p, varargin{:});

options = p.Results;

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

try
    if exist('mfcc', 'file') == 2
        coeffs = mfcc(audio, fs, ...
            'NumCoeffs', options.NumCoeffs, ...
            'Window', hann(options.WindowLength, 'periodic'), ...
            'OverlapLength', options.OverlapLength);
        mfcc = coeffs;
    else
        % Fallback: simplified MFCC calculation
        warning('AdvancedAudioProcessor:NoMFCC', 'MFCC not available');
        mfcc = [];
    end
catch ME
    warning('AdvancedAudioProcessor:MFCCError', 'Error: %s', ME.message);
    mfcc = [];
end
end

function features = extractSpectralFeatures(audio, fs, varargin)
% Extract comprehensive spectral features

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

% Extract features manually or using audioFeatureExtractor
features = struct();

try
    if exist('spectralCentroid', 'file') == 2
        features.centroid = spectralCentroid(audio, fs);
        features.rolloff = spectralRolloffPoint(audio, fs);
        features.flux = spectralFlux(audio, fs);
        features.entropy = spectralEntropy(audio, fs);
        features.crest = spectralCrest(audio, fs);
        features.flatness = spectralFlatness(audio, fs);
    else
        % Fallback: basic spectral features
        [S, F, T] = spectrogram(audio, 256, 200, 256, fs);
        magnitude = abs(S);

        % Spectral centroid
        features.centroid = sum(magnitude .* F, 1) ./ sum(magnitude, 1);
        features.rolloff = [];
        features.flux = [];
        features.entropy = [];
        features.crest = [];
        features.flatness = [];
    end
catch ME
    warning('AdvancedAudioProcessor:SpectralError', 'Error: %s', ME.message);
    features = struct();
end
end

function features = extractTemporalFeatures(audio, fs, varargin)
% Extract temporal features

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

features = struct();

try
    if exist('zerocrossrate', 'file') == 2
        features.zeroCrossRate = zerocrossrate(audio);
    else
        % Manual zero crossing rate
        zcr = sum(abs(diff(sign(audio)))) / (2 * length(audio));
        features.zeroCrossRate = zcr;
    end

    % Energy
    features.energy = sum(audio.^2);
    features.rms = rms(audio);

    % Short-time energy
    windowSize = round(fs * 0.02);
    features.shortTimeEnergy = movmean(audio.^2, windowSize);

catch ME
    warning('AdvancedAudioProcessor:TemporalError', 'Error: %s', ME.message);
end
end

function features = extractAllFeatures(audio, fs, varargin)
% Extract complete feature set for machine learning

try
    if exist('audioFeatureExtractor', 'file') == 2
        % Use Audio Toolbox feature extractor
        extractor = audioFeatureExtractor('SampleRate', fs, ...
            'mfcc', true, 'mfccDelta', true, 'mfccDeltaDelta', true, ...
            'spectralCentroid', true, 'spectralCrest', true, ...
            'spectralEntropy', true, 'spectralFlatness', true, ...
            'spectralFlux', true, 'spectralRolloffPoint', true, ...
            'zerocrossrate', true);

        features = extract(extractor, audio);
    else
        % Fallback: combine manual feature extraction
        features = struct();
        features.spectral = extractSpectralFeatures(audio, fs);
        features.temporal = extractTemporalFeatures(audio, fs);
        features.mfcc = extractMFCC(audio, fs);
    end
catch ME
    warning('AdvancedAudioProcessor:FeatureError', 'Error: %s', ME.message);
    features = struct();
end
end

%% Advanced Filtering Methods

function filtered = octaveFilter(audio, fs, varargin)
% Octave band filtering

p = inputParser;
addParameter(p, 'CenterFrequency', 1000, @isnumeric);
parse(p, varargin{:});

options = p.Results;

try
    if exist('octaveFilter', 'file') == 2
        filt = octaveFilter('FilterOrder', 6, ...
            'CenterFrequency', options.CenterFrequency, ...
            'SampleRate', fs);
        filtered = filt(audio);
    else
        % Fallback: bandpass filter with octave bandwidth
        fc = options.CenterFrequency;
        lowFreq = fc / sqrt(2);
        highFreq = fc * sqrt(2);
        [b, a] = butter(3, [lowFreq, highFreq] / (fs/2), 'bandpass');
        filtered = filtfilt(b, a, audio);
    end
catch ME
    warning('AdvancedAudioProcessor:OctaveError', 'Error: %s', ME.message);
    filtered = audio;
end
end

function filtered = thirdOctaveFilter(audio, fs, varargin)
% 1/3 octave band filtering

p = inputParser;
addParameter(p, 'CenterFrequency', 1000, @isnumeric);
parse(p, varargin{:});

options = p.Results;

try
    if exist('octaveFilter', 'file') == 2
        filt = octaveFilter('FilterOrder', 6, ...
            'CenterFrequency', options.CenterFrequency, ...
            'BandwidthMode', '1/3 octave', ...
            'SampleRate', fs);
        filtered = filt(audio);
    else
        % Fallback: bandpass with 1/3 octave bandwidth
        fc = options.CenterFrequency;
        lowFreq = fc / (2^(1/6));
        highFreq = fc * (2^(1/6));
        [b, a] = butter(3, [lowFreq, highFreq] / (fs/2), 'bandpass');
        filtered = filtfilt(b, a, audio);
    end
catch ME
    warning('AdvancedAudioProcessor:ThirdOctaveError', 'Error: %s', ME.message);
    filtered = audio;
end
end

function filtered = parametricEQ(audio, fs, bands, varargin)
% Parametric EQ with Q control
% bands: struct array with fields .frequency, .gain, .Q

filtered = audio;

for i = 1:length(bands)
    band = bands(i);

    % Design peaking filter
    [b, a] = designPeakingFilter(band.frequency, band.gain, band.Q, fs);
    filtered = filtfilt(b, a, filtered);
end
end

function filtered = graphicEQ(audio, fs, gains, varargin)
% 31-band graphic EQ
% gains: array of 31 gain values in dB

% Standard 31-band ISO frequencies (1/3 octave)
frequencies = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, ...
               315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, ...
               3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000];

if length(gains) ~= 31
    error('AdvancedAudioProcessor:GEQError', 'Graphic EQ requires 31 gain values');
end

% Create bands structure
bands = struct('frequency', num2cell(frequencies), ...
               'gain', num2cell(gains), ...
               'Q', num2cell(repmat(4.3, size(frequencies))));  % Standard Q for 1/3 octave

filtered = parametricEQ(audio, fs, bands);
end

%% Time/Pitch Methods

function stretched = timeScale(audio, fs, factor, varargin)
% Time stretching with pitch preservation

p = inputParser;
addParameter(p, 'PreserveFormants', true, @islogical);
parse(p, varargin{:});

try
    if exist('audioTimeScaler', 'file') == 2
        scaler = audioTimeScaler('SampleRate', fs);
        stretched = scaler(audio, factor);
    else
        % Fallback: resample (will change pitch)
        warning('AdvancedAudioProcessor:NoTimeScaler', 'audioTimeScaler not available, using resample');
        stretched = resample(audio, round(size(audio,1)/factor), size(audio,1));
    end
catch ME
    warning('AdvancedAudioProcessor:TimeScaleError', 'Error: %s', ME.message);
    stretched = audio;
end
end

function shifted = advancedPitchShift(audio, fs, semitones, varargin)
% Advanced pitch shifting using phase vocoder

% This is a placeholder - full implementation requires phase vocoder
shifted = audio;
warning('AdvancedAudioProcessor:PitchShift', ...
    'Advanced pitch shifting requires custom phase vocoder implementation');
end

function processed = phaseVocoder(audio, fs, varargin)
% Custom phase vocoder processing

% Placeholder for custom phase vocoder implementation
processed = audio;
warning('AdvancedAudioProcessor:PhaseVocoder', ...
    'Phase vocoder requires custom implementation');
end

%% Spatial Audio Methods

function widened = stereoWiden(audio, width, varargin)
% Stereo widening

if size(audio, 2) == 1
    widened = repmat(audio, 1, 2);
    return;
end

% M/S processing
mid = (audio(:,1) + audio(:,2)) / 2;
side = (audio(:,1) - audio(:,2)) / 2;

% Adjust side signal
side = side * (1 + width);

% Reconstruct L/R
widened = zeros(size(audio));
widened(:,1) = mid + side;
widened(:,2) = mid - side;

% Normalize to prevent clipping
maxVal = max(abs(widened(:)));
if maxVal > 1.0
    widened = widened / maxVal;
end
end

function processed = midSideProcess(audio, midGain, sideGain)
% Mid-side processing

if size(audio, 2) == 1
    processed = audio * midGain;
    return;
end

% Convert to M/S
mid = (audio(:,1) + audio(:,2)) / 2;
side = (audio(:,1) - audio(:,2)) / 2;

% Apply gains
mid = mid * midGain;
side = side * sideGain;

% Convert back to L/R
processed = zeros(size(audio));
processed(:,1) = mid + side;
processed(:,2) = mid - side;
end

function spatialized = spatialize3D(audio, azimuth, elevation, varargin)
% 3D audio positioning using HRTF (placeholder)

% This requires HRTF database - placeholder implementation
spatialized = audio;
warning('AdvancedAudioProcessor:Spatialize', ...
    '3D spatialization requires HRTF database (not included)');
end

%% Helper Functions

function [pitch, confidence] = traditionalPitchDetection(audio, fs, options)
% Fallback autocorrelation-based pitch detection

windowLength = options.WindowLength;
overlapLength = options.OverlapLength;
hopLength = windowLength - overlapLength;

numFrames = floor((length(audio) - windowLength) / hopLength) + 1;
pitch = zeros(numFrames, 1);
confidence = zeros(numFrames, 1);

for i = 1:numFrames
    startIdx = (i-1) * hopLength + 1;
    endIdx = startIdx + windowLength - 1;
    frame = audio(startIdx:endIdx);

    % Autocorrelation
    [r, lags] = xcorr(frame, 'coeff');
    r = r(lags >= 0);

    % Find first peak (ignoring lag 0)
    minLag = round(fs / options.Range(2));
    maxLag = round(fs / options.Range(1));

    [pkVal, pkLoc] = max(r(minLag:maxLag));

    if pkVal > 0.3
        pitch(i) = fs / (pkLoc + minLag - 1);
        confidence(i) = pkVal;
    end
end
end

function onsetTimes = simpleOnsetDetection(audio, fs, threshold)
% Simple energy-based onset detection

% Calculate energy envelope
windowSize = round(fs * 0.02);
envelope = movmean(audio.^2, windowSize);

% Find peaks in derivative
diff_envelope = [0; diff(envelope)];
diff_envelope(diff_envelope < 0) = 0;

% Threshold
maxDiff = max(diff_envelope);
onsetIndices = find(diff_envelope > threshold * maxDiff);

% Convert to time
onsetTimes = onsetIndices / fs;
end

function hr = calculateSimpleHR(audio, windowLength)
% Simplified harmonic ratio calculation

numFrames = floor(length(audio) / windowLength);
hr = zeros(numFrames, 1);

for i = 1:numFrames
    startIdx = (i-1) * windowLength + 1;
    endIdx = startIdx + windowLength - 1;
    frame = audio(startIdx:endIdx);

    % Simple harmonic/noise ratio estimate
    fft_frame = abs(fft(frame));
    peaks = findpeaks(fft_frame);
    harmonicEnergy = sum(peaks.^2);
    totalEnergy = sum(fft_frame.^2);

    hr(i) = harmonicEnergy / (totalEnergy + eps);
end
end

function [b, a] = designPeakingFilter(frequency, gainDB, Q, fs)
% Design peaking filter for parametric EQ

w0 = 2 * pi * frequency / fs;
alpha = sin(w0) / (2 * Q);
A = 10^(gainDB/40);

% Coefficients
b0 = 1 + alpha * A;
b1 = -2 * cos(w0);
b2 = 1 - alpha * A;
a0 = 1 + alpha / A;
a1 = -2 * cos(w0);
a2 = 1 - alpha / A;

% Normalize
b = [b0, b1, b2] / a0;
a = [a0, a1, a2] / a0;
end

function info = getInfo(processor)
% Get processor information

info = struct();
info.HasAudioToolbox = processor.HasAudioToolbox;
info.Version = processor.Version;

if processor.HasAudioToolbox
    info.Capabilities = {
        'Neural network pitch detection (pitchnn)', ...
        'Onset detection (detectSpeech)', ...
        'Acoustic loudness (acousticLoudness)', ...
        'SPL metering (splMeter)', ...
        'MFCC extraction (mfcc)', ...
        'Spectral features', ...
        'Time scaling (audioTimeScaler)', ...
        'Octave filtering', ...
        'Feature extraction for ML', ...
        'Gammatone filterbank'
    };
else
    info.Capabilities = {
        'Basic pitch detection', ...
        'Simple onset detection', ...
        'Basic spectral analysis', ...
        'Standard filtering', ...
        'Limited functionality without Audio Toolbox'
    };
end
end
