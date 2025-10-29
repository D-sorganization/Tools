function processedData = AudioEffects(audioData, effectType, varargin)
%AUDIOEFFECTS Comprehensive audio effects processing library
%
%   PROCESSEDDATA = AUDIOEFFECTS(AUDIODATA, EFFECTTYPE) applies the specified
%   audio effect to the input audio data.
%
%   PROCESSEDDATA = AUDIOEFFECTS(AUDIODATA, EFFECTTYPE, 'Property', Value, ...)
%   specifies additional effect parameters using property-value pairs.
%
%   Input Arguments:
%   ---------------
%   AUDIODATA - Audio data matrix (samples x channels)
%   EFFECTTYPE - Effect type string
%
%   Supported Effects:
%   -----------------
%   - 'Reverb' - Algorithmic and convolution-based reverb
%   - 'Delay' - Delay/echo with feedback control
%   - 'EQ' - Parametric equalizer
%   - 'Compression' - Dynamic range compression
%   - 'Limiting' - Peak limiting
%   - 'Distortion' - Harmonic distortion/overdrive
%   - 'Chorus' - Chorus effect with LFO modulation
%   - 'Flanger' - Flanger effect with LFO modulation
%   - 'PitchShift' - Real-time pitch shifting
%   - 'TimeStretch' - Independent tempo/pitch control
%
%   Optional Properties:
%   ------------------
%   'SampleRate'      - Sample rate in Hz (default: 44100)
%   'Mix'             - Dry/wet mix ratio 0-1 (default: 0.5)
%   'Bypass'          - Bypass effect (default: false)
%
%   Reverb Properties:
%   'RoomSize'        - Room size 0-1 (default: 0.5)
%   'DecayTime'       - Decay time in seconds (default: 2.0)
%   'Damping'         - High-frequency damping 0-1 (default: 0.5)
%   'PreDelay'         - Pre-delay in seconds (default: 0.02)
%
%   Delay Properties:
%   'DelayTime'       - Delay time in seconds (default: 0.25)
%   'Feedback'         - Feedback amount 0-0.95 (default: 0.3)
%   'TempoSync'        - Sync to tempo (default: false)
%   'Tempo'            - Tempo in BPM (default: 120)
%
%   EQ Properties:
%   'LowGain'          - Low frequency gain in dB (default: 0)
%   'MidGain'          - Mid frequency gain in dB (default: 0)
%   'HighGain'         - High frequency gain in dB (default: 0)
%   'LowFreq'          - Low frequency crossover in Hz (default: 250)
%   'HighFreq'         - High frequency crossover in Hz (default: 4000)
%
%   Compression Properties:
%   'Threshold'        - Compression threshold in dB (default: -12)
%   'Ratio'            - Compression ratio (default: 4)
%   'Attack'           - Attack time in ms (default: 10)
%   'Release'          - Release time in ms (default: 100)
%   'Knee'             - Soft knee width in dB (default: 2)
%
%   Limiting Properties:
%   'Limit'            - Limiting threshold in dB (default: -0.1)
%
%   Distortion Properties:
%   'Drive'            - Distortion amount 0-1 (default: 0.5)
%   'Tone'             - Tone control 0-1 (default: 0.5)
%   'Level'            - Output level 0-1 (default: 0.7)
%
%   Modulation Properties (Chorus/Flanger):
%   'Rate'             - LFO rate in Hz (default: 0.5)
%   'Depth'            - Modulation depth 0-1 (default: 0.3)
%
%   Pitch/Time Properties:
%   'PitchShift'       - Pitch shift in semitones (default: 0)
%   'TimeStretch'      - Time stretch factor (default: 1.0)
%
%   Output Arguments:
%   ----------------
%   PROCESSEDDATA - Processed audio data matrix (same size as input)
%
%   Example:
%   --------
%   % Load audio data
%   [data, fs] = audioread('song.wav');
%
%   % Apply reverb
%   processed = AudioEffects(data, 'Reverb', 'RoomSize', 0.7, 'DecayTime', 3.0, 'SampleRate', fs);
%
%   % Apply compression
%   processed = AudioEffects(data, 'Compression', 'Threshold', -6, 'Ratio', 3, 'SampleRate', fs);
%
%   % Apply EQ
%   processed = AudioEffects(data, 'EQ', 'LowGain', 3, 'HighGain', -2, 'SampleRate', fs);
%
%   See also: AudioFilterEngine, FFTFilters

arguments
    audioData (:,:) double
    effectType (1,1) string
    options.SampleRate (1,1) double {mustBePositive} = 44100
    options.Mix (1,1) double {mustBeInRange(options.Mix, 0, 1)} = 0.5
    options.Bypass (1,1) logical = false
    % Reverb parameters
    options.RoomSize (1,1) double {mustBeInRange(options.RoomSize, 0, 1)} = 0.5
    options.DecayTime (1,1) double {mustBePositive} = 2.0
    options.Damping (1,1) double {mustBeInRange(options.Damping, 0, 1)} = 0.5
    options.PreDelay (1,1) double {mustBeNonnegative} = 0.02
    % Delay parameters
    options.DelayTime (1,1) double {mustBePositive} = 0.25
    options.Feedback (1,1) double {mustBeInRange(options.Feedback, 0, 0.95)} = 0.3
    options.TempoSync (1,1) logical = false
    options.Tempo (1,1) double {mustBePositive} = 120
    % EQ parameters
    options.LowGain (1,1) double = 0
    options.MidGain (1,1) double = 0
    options.HighGain (1,1) double = 0
    options.LowFreq (1,1) double {mustBePositive} = 250
    options.HighFreq (1,1) double {mustBePositive} = 4000
    % Compression parameters
    options.Threshold (1,1) double = -12
    options.Ratio (1,1) double {mustBePositive} = 4
    options.Attack (1,1) double {mustBePositive} = 10
    options.Release (1,1) double {mustBePositive} = 100
    options.Knee (1,1) double {mustBeNonnegative} = 2
    % Limiting parameters
    options.Limit (1,1) double = -0.1
    % Distortion parameters
    options.Drive (1,1) double {mustBeInRange(options.Drive, 0, 1)} = 0.5
    options.Tone (1,1) double {mustBeInRange(options.Tone, 0, 1)} = 0.5
    options.Level (1,1) double {mustBeInRange(options.Level, 0, 1)} = 0.7
    % Modulation parameters
    options.Rate (1,1) double {mustBePositive} = 0.5
    options.Depth (1,1) double {mustBeInRange(options.Depth, 0, 1)} = 0.3
    % Pitch/Time parameters
    options.PitchShift (1,1) double = 0
    options.TimeStretch (1,1) double {mustBePositive} = 1.0
end

% Validate input
if isempty(audioData)
    error('AudioEffects:EmptyInput', 'Input audio data is empty');
end

% Bypass effect if requested
if options.Bypass
    processedData = audioData;
    return;
end

% Apply effect based on type
switch effectType
    case 'Reverb'
        processedData = applyReverb(audioData, options);
    case 'Delay'
        processedData = applyDelay(audioData, options);
    case 'EQ'
        processedData = applyEQ(audioData, options);
    case 'Compression'
        processedData = applyCompression(audioData, options);
    case 'Limiting'
        processedData = applyLimiting(audioData, options);
    case 'Distortion'
        processedData = applyDistortion(audioData, options);
    case 'Chorus'
        processedData = applyChorus(audioData, options);
    case 'Flanger'
        processedData = applyFlanger(audioData, options);
    case 'PitchShift'
        processedData = applyPitchShift(audioData, options);
    case 'TimeStretch'
        processedData = applyTimeStretch(audioData, options);
    otherwise
        error('AudioEffects:UnknownEffect', 'Unknown effect type: %s', effectType);
end

% Apply dry/wet mix
if options.Mix < 1.0
    processedData = options.Mix * processedData + (1 - options.Mix) * audioData;
end
end

function processedData = applyReverb(audioData, options)
% Apply algorithmic reverb using Schroeder/Moorer algorithm

[nSamples, nChannels] = size(audioData);
processedData = zeros(size(audioData));

% Reverb parameters
roomSize = options.RoomSize;
decayTime = options.DecayTime;
damping = options.Damping;
preDelay = options.PreDelay;
sampleRate = options.SampleRate;

% Calculate delay line lengths (prime numbers for diffusion)
delayLengths = round([37, 59, 83, 97, 113, 127] * roomSize * sampleRate / 1000);

% Initialize delay lines
delayLines = zeros(max(delayLengths), length(delayLengths));
delayIndices = ones(length(delayLengths), 1);

% Pre-delay
preDelaySamples = round(preDelay * sampleRate);
preDelayBuffer = zeros(preDelaySamples, 1);
preDelayIndex = 1;

% Process each sample
for n = 1:nSamples
    % Get input sample (mono sum for reverb)
    inputSample = sum(audioData(n, :)) / nChannels;

    % Apply pre-delay
    if preDelaySamples > 0
        delayedSample = preDelayBuffer(preDelayIndex);
        preDelayBuffer(preDelayIndex) = inputSample;
        preDelayIndex = mod(preDelayIndex, preDelaySamples) + 1;
        inputSample = delayedSample;
    end

    % Process through delay lines
    outputSample = 0;
    for i = 1:length(delayLengths)
        % Read from delay line
        delayOutput = delayLines(delayIndices(i), i);

        % Apply damping (simple low-pass filter)
        delayOutput = delayOutput * (1 - damping) + delayOutput * damping * 0.5;

        % Write to delay line
        delayLines(delayIndices(i), i) = inputSample + delayOutput * 0.3;

        % Update delay index
        delayIndices(i) = mod(delayIndices(i), delayLengths(i)) + 1;

        % Accumulate output
        outputSample = outputSample + delayOutput;
    end

    % Normalize and apply decay
    outputSample = outputSample / length(delayLengths);
    outputSample = outputSample * exp(-n / (decayTime * sampleRate));

    % Apply to all channels
    processedData(n, :) = outputSample;
end
end

function processedData = applyDelay(audioData, options)
% Apply delay/echo effect

[nSamples, nChannels] = size(audioData);
processedData = zeros(size(audioData));

% Delay parameters
delayTime = options.DelayTime;
feedback = options.Feedback;
sampleRate = options.SampleRate;

% Calculate delay in samples
if options.TempoSync
    % Sync to tempo (quarter note delay)
    beatTime = 60 / options.Tempo;
    delaySamples = round(beatTime * sampleRate);
else
    delaySamples = round(delayTime * sampleRate);
end

% Initialize delay buffer
delayBuffer = zeros(delaySamples, nChannels);
delayIndex = 1;

% Process each sample
for n = 1:nSamples
    for ch = 1:nChannels
        % Get delayed sample
        delayedSample = delayBuffer(delayIndex, ch);

        % Mix input with delayed signal
        outputSample = audioData(n, ch) + delayedSample * feedback;

        % Store in delay buffer
        delayBuffer(delayIndex, ch) = outputSample;

        % Set output
        processedData(n, ch) = outputSample;
    end

    % Update delay index
    delayIndex = mod(delayIndex, delaySamples) + 1;
end
end

function processedData = applyEQ(audioData, options)
% Apply parametric equalizer

sampleRate = options.SampleRate;

% Design EQ filters
% Low shelf
if options.LowGain ~= 0
    lowShelf = designfilt('lowshelf', 'FilterOrder', 2, ...
        'HalfPowerFrequency', options.LowFreq, ...
        'SampleRate', sampleRate);
    audioData = filtfilt(lowShelf, audioData);
end

% High shelf
if options.HighGain ~= 0
    highShelf = designfilt('highshelf', 'FilterOrder', 2, ...
        'HalfPowerFrequency', options.HighFreq, ...
        'SampleRate', sampleRate);
    audioData = filtfilt(highShelf, audioData);
end

% Mid bell (parametric)
if options.MidGain ~= 0
    midFreq = sqrt(options.LowFreq * options.HighFreq);
    midBell = designfilt('peaking', 'FilterOrder', 2, ...
        'CenterFrequency', midFreq, ...
        'SampleRate', sampleRate);
    audioData = filtfilt(midBell, audioData);
end

processedData = audioData;
end

function processedData = applyCompression(audioData, options)
% Apply dynamic range compression

[nSamples, nChannels] = size(audioData);
processedData = zeros(size(audioData));

% Compression parameters
threshold = db2mag(options.Threshold);
ratio = options.Ratio;
attackTime = options.Attack / 1000; % Convert to seconds
releaseTime = options.Release / 1000;
knee = options.Knee;
sampleRate = options.SampleRate;

% Calculate time constants
attackCoeff = exp(-1 / (attackTime * sampleRate));
releaseCoeff = exp(-1 / (releaseTime * sampleRate));

% Process each channel
for ch = 1:nChannels
    signal = audioData(:, ch);
    envelope = zeros(size(signal));
    gain = ones(size(signal));

    % Calculate envelope (simple peak detection)
    for n = 1:nSamples
        if n == 1
            envelope(n) = abs(signal(n));
        else
            if abs(signal(n)) > envelope(n-1)
                envelope(n) = attackCoeff * envelope(n-1) + (1 - attackCoeff) * abs(signal(n));
            else
                envelope(n) = releaseCoeff * envelope(n-1) + (1 - releaseCoeff) * abs(signal(n));
            end
        end
    end

    % Calculate gain reduction
    for n = 1:nSamples
        if envelope(n) > threshold
            % Soft knee
            if knee > 0
                kneeStart = threshold / db2mag(knee);
                if envelope(n) < kneeStart
                    gain(n) = 1;
                else
                    % Soft knee compression
                    gainReduction = 1 - (1/ratio) * (envelope(n) - threshold) / (envelope(n) + eps);
                    gain(n) = max(gainReduction, 1/ratio);
                end
            else
                % Hard knee compression
                gainReduction = 1 - (1/ratio) * (envelope(n) - threshold) / (envelope(n) + eps);
                gain(n) = max(gainReduction, 1/ratio);
            end
        else
            gain(n) = 1;
        end
    end

    % Apply gain
    processedData(:, ch) = signal .* gain;
end
end

function processedData = applyLimiting(audioData, options)
% Apply peak limiting

limit = db2mag(options.Limit);

% Simple hard limiting
processedData = audioData;
processedData(processedData > limit) = limit;
processedData(processedData < -limit) = -limit;
end

function processedData = applyDistortion(audioData, options)
% Apply harmonic distortion/overdrive

drive = options.Drive;
tone = options.Tone;
level = options.Level;

% Apply drive (soft clipping)
processedData = tanh(audioData * (1 + drive * 10));

% Apply tone control (simple high-pass filter)
if tone < 0.5
    % Darken (low-pass)
    cutoff = 2000 * (tone * 2);
    [b, a] = butter(2, cutoff / (options.SampleRate/2), 'low');
    processedData = filtfilt(b, a, processedData);
elseif tone > 0.5
    % Brighten (high-pass)
    cutoff = 2000 * ((tone - 0.5) * 2);
    [b, a] = butter(2, cutoff / (options.SampleRate/2), 'high');
    processedData = filtfilt(b, a, processedData);
end

% Apply output level
processedData = processedData * level;
end

function processedData = applyChorus(audioData, options)
% Apply chorus effect with LFO modulation

[nSamples, nChannels] = size(audioData);
processedData = zeros(size(audioData));

% Chorus parameters
rate = options.Rate;
depth = options.Depth;
feedback = options.Feedback;
sampleRate = options.SampleRate;

% Calculate modulation
t = (0:nSamples-1)' / sampleRate;
modulation = depth * sin(2 * pi * rate * t);

% Maximum delay for chorus
maxDelay = round(0.05 * sampleRate); % 50ms max delay
delayBuffer = zeros(maxDelay, nChannels);
delayIndex = 1;

% Process each sample
for n = 1:nSamples
    for ch = 1:nChannels
        % Calculate variable delay
        delaySamples = round(maxDelay/2 + modulation(n) * maxDelay/2);
        delaySamples = max(1, min(delaySamples, maxDelay));

        % Get delayed sample
        readIndex = mod(delayIndex - delaySamples - 1, maxDelay) + 1;
        delayedSample = delayBuffer(readIndex, ch);

        % Mix input with delayed signal
        outputSample = audioData(n, ch) + delayedSample * feedback;

        % Store in delay buffer
        delayBuffer(delayIndex, ch) = outputSample;

        % Set output
        processedData(n, ch) = outputSample;
    end

    % Update delay index
    delayIndex = mod(delayIndex, maxDelay) + 1;
end
end

function processedData = applyFlanger(audioData, options)
% Apply flanger effect (similar to chorus but with shorter delays)

% Use chorus implementation with different parameters
flangerOptions = options;
flangerOptions.Depth = options.Depth * 0.1; % Shorter delays for flanger
processedData = applyChorus(audioData, flangerOptions);
end

function processedData = applyPitchShift(audioData, options)
% Apply pitch shifting (simplified implementation)

pitchShift = options.PitchShift;

if abs(pitchShift) < 0.1
    processedData = audioData;
    return;
end

% Simple pitch shifting using resampling and time stretching
shiftFactor = 2^(pitchShift / 12); % Convert semitones to frequency ratio

% Resample to change pitch
processedData = resample(audioData, round(shiftFactor * 1000), 1000);

% Resize to original length
originalLength = size(audioData, 1);
if size(processedData, 1) ~= originalLength
    processedData = resample(processedData, originalLength, size(processedData, 1));
end
end

function processedData = applyTimeStretch(audioData, options)
% Apply time stretching (simplified implementation)

timeStretch = options.TimeStretch;

if abs(timeStretch - 1.0) < 0.01
    processedData = audioData;
    return;
end

% Simple time stretching using resampling
originalLength = size(audioData, 1);
newLength = round(originalLength / timeStretch);

processedData = resample(audioData, newLength, originalLength);

% Resize to original length
if size(processedData, 1) ~= originalLength
    processedData = resample(processedData, originalLength, size(processedData, 1));
end
end
