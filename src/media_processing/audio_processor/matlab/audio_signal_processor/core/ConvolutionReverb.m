function reverb = ConvolutionReverb()
%CONVOLUTIONREVERB Convolution-based reverb using impulse responses
%
%   REVERB = CONVOLUTIONREVERB() creates a convolution reverb processor
%   that uses impulse responses (IRs) to simulate acoustic environments.
%
%   What is an Impulse Response?
%   ---------------------------
%   An impulse response (IR) is a recording of how a space responds to a
%   short, sharp sound (like a balloon pop or starter pistol). It captures
%   the acoustic characteristics of that space - all the reflections, echoes,
%   and reverberations.
%
%   By convolving your audio with an IR, you make it sound like it was
%   recorded in that space!
%
%   Key Features:
%   ------------
%   - Load impulse responses from WAV files
%   - Built-in library of common spaces
%   - Control echo amount (wet/dry mix)
%   - Pre-delay control
%   - EQ on reverb tail
%   - Stereo width control
%   - Reverse reverb
%   - IR trimming and normalization
%
%   Core Methods:
%   ------------
%   loadIR(filename) - Load impulse response from file
%   process(audio, fs) - Apply convolution reverb
%   setWetDry(wet, dry) - Control reverb amount
%   setPreDelay(seconds) - Add pre-delay before reverb
%
%   IR Management:
%   -------------
%   listAvailableIRs() - Show built-in impulse responses
%   generateSyntheticIR(type, params) - Create synthetic IRs
%   trimIR() - Remove silence from IR
%   normalizeIR() - Normalize IR level
%   reverseIR() - Create reverse reverb effect
%
%   Advanced Control:
%   ----------------
%   setEQ(low, mid, high) - EQ the reverb
%   setStereoWidth(width) - Control stereo image
%   setDamping(amount) - High-frequency damping
%   setTailLength(seconds) - Truncate reverb tail
%
%   Example Usage:
%   -------------
%   % Create reverb and load concert hall IR
%   reverb = ConvolutionReverb();
%   reverb.loadIR('concert_hall.wav');
%
%   % Process audio
%   [audio, fs] = audioread('dry_vocal.wav');
%   wet = reverb.process(audio, fs, 'WetDry', 0.3);  % 30% wet
%
%   % Try different spaces
%   reverb.loadIR('church.wav');
%   church = reverb.process(audio, fs);
%
%   reverb.loadIR('small_room.wav');
%   room = reverb.process(audio, fs);
%
%   See also: AudioEffects, fftfilt, conv

% Initialize reverb structure
reverb = struct();
reverb.Version = '1.0';
reverb.IR = [];  % Current impulse response
reverb.IRSampleRate = [];
reverb.IRName = '';

% Processing parameters
reverb.WetLevel = 0.3;   % Reverb amount (0-1)
reverb.DryLevel = 0.7;   % Direct sound (0-1)
reverb.PreDelay = 0;     % Pre-delay in seconds
reverb.StereoWidth = 1.0; % Stereo width (0-2)
reverb.TailLength = [];  % Truncate IR (seconds, [] = full)
reverb.Damping = 0;      % High frequency damping (0-1)

% EQ on reverb
reverb.EQ = struct('LowGain', 0, 'MidGain', 0, 'HighGain', 0);

% Built-in library
reverb.Library = initializeLibrary();

% Add methods
reverb.loadIR = @(filename) loadIR(reverb, filename);
reverb.loadBuiltIn = @(name) loadBuiltIn(reverb, name);
reverb.process = @(audio, fs, varargin) processConvolutionReverb(reverb, audio, fs, varargin{:});
reverb.setWetDry = @(wet, dry) setWetDry(reverb, wet, dry);
reverb.setPreDelay = @(seconds) setPreDelay(reverb, seconds);
reverb.setEQ = @(low, mid, high) setEQ(reverb, low, mid, high);
reverb.setStereoWidth = @(width) setStereoWidth(reverb, width);
reverb.setDamping = @(amount) setDamping(reverb, amount);
reverb.setTailLength = @(seconds) setTailLength(reverb, seconds);

% IR manipulation
reverb.trimIR = @() trimIR(reverb);
reverb.normalizeIR = @() normalizeIR(reverb);
reverb.reverseIR = @() reverseIR(reverb);
reverb.generateSyntheticIR = @(type, varargin) generateSyntheticIR(reverb, type, varargin{:});

% Library
reverb.listAvailableIRs = @() listAvailableIRs(reverb);
reverb.getIRInfo = @() getIRInfo(reverb);
reverb.plotIR = @() plotIR(reverb);

% Utility
reverb.getInfo = @() getInfo(reverb);
end

%% Core Methods

function success = loadIR(reverb, filename)
% Load impulse response from WAV file

success = false;

try
    % Load IR
    [ir, fs] = audioread(filename);

    % Store IR
    reverb.IR = ir;
    reverb.IRSampleRate = fs;
    [~, name, ~] = fileparts(filename);
    reverb.IRName = name;

    fprintf('Loaded IR: %s\n', name);
    fprintf('Sample Rate: %d Hz\n', fs);
    fprintf('Length: %.2f seconds\n', size(ir, 1) / fs);
    fprintf('Channels: %d\n', size(ir, 2));

    success = true;

catch ME
    error('ConvolutionReverb:LoadError', 'Error loading IR: %s', ME.message);
end
end

function success = loadBuiltIn(reverb, name)
% Load built-in synthetic impulse response

success = false;

% Check if exists in library
if ~isfield(reverb.Library, name)
    available = fieldnames(reverb.Library);
    error('ConvolutionReverb:NotFound', ...
        'Built-in IR "%s" not found. Available: %s', ...
        name, strjoin(available, ', '));
end

% Generate the IR
irParams = reverb.Library.(name);
reverb.generateSyntheticIR(irParams.type, ...
    'SampleRate', irParams.sampleRate, ...
    'Length', irParams.length, ...
    'Params', irParams.params);

reverb.IRName = name;
fprintf('Loaded built-in IR: %s\n', name);

success = true;
end

function processed = processConvolutionReverb(reverb, audio, fs, varargin)
% Apply convolution reverb to audio

p = inputParser;
addParameter(p, 'WetDry', [], @isnumeric);  % Override wet/dry
addParameter(p, 'PreDelay', [], @isnumeric);
addParameter(p, 'StereoWidth', [], @isnumeric);
parse(p, varargin{:});

% Check if IR loaded
if isempty(reverb.IR)
    error('ConvolutionReverb:NoIR', 'No impulse response loaded. Use loadIR() first.');
end

% Use parameter overrides if provided
wetLevel = ternary(~isempty(p.Results.WetDry), p.Results.WetDry, reverb.WetLevel);
preDelay = ternary(~isempty(p.Results.PreDelay), p.Results.PreDelay, reverb.PreDelay);
stereoWidth = ternary(~isempty(p.Results.StereoWidth), p.Results.StereoWidth, reverb.StereoWidth);

% Resample IR if needed
ir = reverb.IR;
if reverb.IRSampleRate ~= fs
    fprintf('Resampling IR from %d Hz to %d Hz...\n', reverb.IRSampleRate, fs);
    ir = resample(ir, fs, reverb.IRSampleRate);
end

% Truncate IR if requested
if ~isempty(reverb.TailLength)
    maxSamples = round(reverb.TailLength * fs);
    if maxSamples < size(ir, 1)
        ir = ir(1:maxSamples, :);
    end
end

% Apply damping to IR
if reverb.Damping > 0
    ir = applyDampingToIR(ir, reverb.Damping, fs);
end

% Apply EQ to IR
if reverb.EQ.LowGain ~= 0 || reverb.EQ.MidGain ~= 0 || reverb.EQ.HighGain ~= 0
    ir = applyEQToIR(ir, reverb.EQ, fs);
end

% Apply stereo width to IR
if stereoWidth ~= 1.0 && size(ir, 2) == 2
    ir = applyStereoWidth(ir, stereoWidth);
end

% Handle channel matching
[nSamples, nChannels] = size(audio);
irChannels = size(ir, 2);

if nChannels == 1 && irChannels == 2
    % Mono audio, stereo IR - convert audio to stereo
    audio = repmat(audio, 1, 2);
    nChannels = 2;
elseif nChannels == 2 && irChannels == 1
    % Stereo audio, mono IR - use same IR for both channels
    ir = repmat(ir, 1, 2);
    irChannels = 2;
end

% Convolve
fprintf('Convolving audio with IR...\n');
wetSignal = zeros(nSamples + size(ir, 1) - 1, nChannels);

for ch = 1:nChannels
    if nChannels == 1
        wetSignal(:, ch) = fftfilt(ir, audio(:, ch));
    else
        % For stereo, use appropriate IR channel
        irChannel = min(ch, irChannels);
        wetSignal(:, ch) = fftfilt(ir(:, irChannel), audio(:, ch));
    end
end

% Trim to original length (or leave longer for natural decay)
% wetSignal = wetSignal(1:nSamples, :);  % Trim to original length
% Or preserve reverb tail:
% (keep full length)

% Apply pre-delay
if preDelay > 0
    preDelaySamples = round(preDelay * fs);
    wetSignal = [zeros(preDelaySamples, nChannels); wetSignal];
end

% Match lengths for mixing
minLength = min(size(audio, 1), size(wetSignal, 1));
maxLength = max(size(audio, 1), size(wetSignal, 1));

% Pad shorter signal
if size(audio, 1) < maxLength
    audio = [audio; zeros(maxLength - size(audio, 1), nChannels)];
end
if size(wetSignal, 1) < maxLength
    wetSignal = [wetSignal; zeros(maxLength - size(wetSignal, 1), nChannels)];
end

% Mix dry and wet
dryLevel = reverb.DryLevel;
processed = dryLevel * audio + wetLevel * wetSignal;

% Normalize to prevent clipping
maxVal = max(abs(processed(:)));
if maxVal > 0.99
    processed = processed * 0.99 / maxVal;
    fprintf('Note: Output normalized to prevent clipping\n');
end

fprintf('Reverb applied: %.0f%% wet, %.0f%% dry\n', wetLevel*100, dryLevel*100);
end

%% Parameter Control

function setWetDry(reverb, wet, dry)
% Set wet/dry mix levels

arguments
    reverb
    wet (1,1) double {mustBeInRange(wet, 0, 1)}
    dry (1,1) double {mustBeInRange(dry, 0, 1)}
end

reverb.WetLevel = wet;
reverb.DryLevel = dry;

fprintf('Reverb mix: %.0f%% wet, %.0f%% dry\n', wet*100, dry*100);
end

function setPreDelay(reverb, seconds)
% Set pre-delay time

arguments
    reverb
    seconds (1,1) double {mustBeNonnegative}
end

reverb.PreDelay = seconds;
fprintf('Pre-delay: %.0f ms\n', seconds * 1000);
end

function setEQ(reverb, low, mid, high)
% Set EQ on reverb tail

arguments
    reverb
    low (1,1) double
    mid (1,1) double
    high (1,1) double
end

reverb.EQ.LowGain = low;
reverb.EQ.MidGain = mid;
reverb.EQ.HighGain = high;

fprintf('Reverb EQ: Low=%+.1f dB, Mid=%+.1f dB, High=%+.1f dB\n', low, mid, high);
end

function setStereoWidth(reverb, width)
% Set stereo width

arguments
    reverb
    width (1,1) double {mustBeInRange(width, 0, 2)}
end

reverb.StereoWidth = width;
fprintf('Stereo width: %.1f (0=mono, 1=normal, 2=wide)\n', width);
end

function setDamping(reverb, amount)
% Set high-frequency damping

arguments
    reverb
    amount (1,1) double {mustBeInRange(amount, 0, 1)}
end

reverb.Damping = amount;
fprintf('Damping: %.0f%%\n', amount * 100);
end

function setTailLength(reverb, seconds)
% Set reverb tail length (truncate IR)

if isempty(seconds)
    reverb.TailLength = [];
    fprintf('Tail length: full (no truncation)\n');
else
    reverb.TailLength = seconds;
    fprintf('Tail length: %.2f seconds\n', seconds);
end
end

%% IR Manipulation

function trimIR(reverb)
% Remove silence from beginning and end of IR

if isempty(reverb.IR)
    error('ConvolutionReverb:NoIR', 'No IR loaded');
end

threshold = 0.001;  % -60 dB

% Find first non-silent sample
startIdx = find(abs(max(reverb.IR, [], 2)) > threshold, 1, 'first');
if isempty(startIdx)
    startIdx = 1;
end

% Find last significant sample
endIdx = find(abs(max(reverb.IR, [], 2)) > threshold, 1, 'last');
if isempty(endIdx)
    endIdx = size(reverb.IR, 1);
end

% Trim
originalLength = size(reverb.IR, 1);
reverb.IR = reverb.IR(startIdx:endIdx, :);
newLength = size(reverb.IR, 1);

fprintf('IR trimmed: %.2f → %.2f seconds (saved %.2f seconds)\n', ...
    originalLength / reverb.IRSampleRate, ...
    newLength / reverb.IRSampleRate, ...
    (originalLength - newLength) / reverb.IRSampleRate);
end

function normalizeIR(reverb)
% Normalize IR to peak = 1.0

if isempty(reverb.IR)
    error('ConvolutionReverb:NoIR', 'No IR loaded');
end

maxVal = max(abs(reverb.IR(:)));
if maxVal > 0
    reverb.IR = reverb.IR / maxVal;
    fprintf('IR normalized (peak was %.4f)\n', maxVal);
else
    warning('IR is silent - cannot normalize');
end
end

function reverseIR(reverb)
% Reverse IR for reverse reverb effect

if isempty(reverb.IR)
    error('ConvolutionReverb:NoIR', 'No IR loaded');
end

reverb.IR = flipud(reverb.IR);
reverb.IRName = [reverb.IRName, '_reversed'];

fprintf('IR reversed - reverse reverb effect active\n');
end

function generateSyntheticIR(reverb, type, varargin)
% Generate synthetic impulse response

p = inputParser;
addParameter(p, 'SampleRate', 44100, @isnumeric);
addParameter(p, 'Length', 2, @isnumeric);  % seconds
addParameter(p, 'Params', struct(), @isstruct);
parse(p, varargin{:});

fs = p.Results.SampleRate;
lengthSec = p.Results.Length;
params = p.Results.Params;

numSamples = round(lengthSec * fs);

switch lower(type)
    case 'room'
        ir = generateRoomIR(numSamples, fs, params);
    case 'hall'
        ir = generateHallIR(numSamples, fs, params);
    case 'chamber'
        ir = generateChamberIR(numSamples, fs, params);
    case 'plate'
        ir = generatePlateIR(numSamples, fs, params);
    case 'spring'
        ir = generateSpringIR(numSamples, fs, params);
    case 'ambience'
        ir = generateAmbienceIR(numSamples, fs, params);
    otherwise
        error('ConvolutionReverb:UnknownType', 'Unknown IR type: %s', type);
end

reverb.IR = ir;
reverb.IRSampleRate = fs;
reverb.IRName = sprintf('Synthetic_%s', type);

fprintf('Generated synthetic %s IR: %.2f seconds\n', type, lengthSec);
end

%% Library Management

function list = listAvailableIRs(reverb)
% List available built-in impulse responses

fprintf('\n=== Built-in Impulse Responses ===\n\n');

list = fieldnames(reverb.Library);

for i = 1:length(list)
    irInfo = reverb.Library.(list{i});
    fprintf('%d. %s\n', i, list{i});
    fprintf('   Type: %s\n', irInfo.type);
    fprintf('   Description: %s\n', irInfo.description);
    fprintf('   Length: %.2f seconds\n\n', irInfo.length);
end

fprintf('Usage: reverb.loadBuiltIn(''%s'')\n', list{1});
fprintf('===================================\n\n');
end

function info = getIRInfo(reverb)
% Get information about loaded IR

if isempty(reverb.IR)
    fprintf('No IR loaded\n');
    info = struct();
    return;
end

info = struct();
info.name = reverb.IRName;
info.sampleRate = reverb.IRSampleRate;
info.lengthSamples = size(reverb.IR, 1);
info.lengthSeconds = size(reverb.IR, 1) / reverb.IRSampleRate;
info.channels = size(reverb.IR, 2);
info.peakLevel = max(abs(reverb.IR(:)));
info.rms = rms(reverb.IR(:));

fprintf('\n=== Impulse Response Info ===\n');
fprintf('Name: %s\n', info.name);
fprintf('Sample Rate: %d Hz\n', info.sampleRate);
fprintf('Length: %.3f seconds (%d samples)\n', info.lengthSeconds, info.lengthSamples);
fprintf('Channels: %d\n', info.channels);
fprintf('Peak Level: %.4f\n', info.peakLevel);
fprintf('RMS Level: %.4f (%.2f dB)\n', info.rms, 20*log10(info.rms));
fprintf('=============================\n\n');
end

function plotIR(reverb)
% Plot impulse response

if isempty(reverb.IR)
    error('ConvolutionReverb:NoIR', 'No IR loaded');
end

figure('Name', sprintf('Impulse Response: %s', reverb.IRName));

% Time domain
subplot(3, 1, 1);
t = (0:size(reverb.IR, 1)-1) / reverb.IRSampleRate;
plot(t, reverb.IR);
title(sprintf('Impulse Response: %s', reverb.IRName));
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend('Left', 'Right');

% Envelope
subplot(3, 1, 2);
envelope = max(abs(reverb.IR), [], 2);
plot(t, 20*log10(envelope + eps));
title('Decay Envelope');
xlabel('Time (s)');
ylabel('Level (dB)');
grid on;

% Frequency response
subplot(3, 1, 3);
nfft = 2^nextpow2(size(reverb.IR, 1));
[h, f] = freqz(reverb.IR(:, 1), 1, nfft, reverb.IRSampleRate);
semilogx(f, 20*log10(abs(h)));
title('Frequency Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
xlim([20, reverb.IRSampleRate/2]);
end

%% Helper Functions

function library = initializeLibrary()
% Initialize built-in IR library

library = struct();

% Small Room
library.small_room = struct(...
    'type', 'room', ...
    'description', 'Small room (bedroom, studio)', ...
    'sampleRate', 44100, ...
    'length', 0.5, ...
    'params', struct('size', 'small', 'damping', 0.7));

% Medium Room
library.medium_room = struct(...
    'type', 'room', ...
    'description', 'Medium room (living room)', ...
    'sampleRate', 44100, ...
    'length', 1.0, ...
    'params', struct('size', 'medium', 'damping', 0.5));

% Large Hall
library.concert_hall = struct(...
    'type', 'hall', ...
    'description', 'Large concert hall', ...
    'sampleRate', 44100, ...
    'length', 3.0, ...
    'params', struct('size', 'large', 'damping', 0.3));

% Chamber
library.chamber = struct(...
    'type', 'chamber', ...
    'description', 'Small chamber/booth', ...
    'sampleRate', 44100, ...
    'length', 0.3, ...
    'params', struct('damping', 0.8));

% Plate Reverb
library.plate = struct(...
    'type', 'plate', ...
    'description', 'Classic plate reverb', ...
    'sampleRate', 44100, ...
    'length', 2.0, ...
    'params', struct('density', 'high'));

% Spring Reverb
library.spring = struct(...
    'type', 'spring', ...
    'description', 'Vintage spring reverb', ...
    'sampleRate', 44100, ...
    'length', 1.5, ...
    'params', struct('springs', 3));

% Ambience
library.ambience = struct(...
    'type', 'ambience', ...
    'description', 'Subtle room ambience', ...
    'sampleRate', 44100, ...
    'length', 0.8, ...
    'params', struct('density', 'low'));
end

%% Synthetic IR Generators

function ir = generateRoomIR(numSamples, fs, params)
% Generate room-like impulse response

% Get parameters
if isfield(params, 'size')
    roomSize = params.size;
else
    roomSize = 'medium';
end

if isfield(params, 'damping')
    damping = params.damping;
else
    damping = 0.5;
end

% Room size determines decay time
switch lower(roomSize)
    case 'small'
        rt60 = 0.3;  % seconds
    case 'medium'
        rt60 = 0.7;
    case 'large'
        rt60 = 1.5;
    otherwise
        rt60 = 0.7;
end

% Generate exponential decay envelope
t = (0:numSamples-1)' / fs;
decay = exp(-6.91 * t / rt60);  % -60 dB decay

% Apply damping (high-frequency rolloff)
if damping > 0
    [b, a] = butter(2, (1 - damping) * 0.9, 'low');
    decay = filtfilt(b, a, decay);
end

% Add early reflections
ir = generateEarlyReflections(numSamples, fs, roomSize);

% Add diffuse reverb tail
noise = randn(numSamples, 1) * 0.1;
tail = noise .* decay;

% Combine
ir = ir + tail;

% Make stereo with slight decorrelation
ir = [ir, delayAndDecorrelate(ir, fs)];

% Normalize
ir = ir / max(abs(ir(:)));
end

function ir = generateHallIR(numSamples, fs, params)
% Generate hall-like impulse response

rt60 = 2.5;  % Long decay for hall

t = (0:numSamples-1)' / fs;
decay = exp(-6.91 * t / rt60);

% More complex early reflections for large space
ir = generateEarlyReflections(numSamples, fs, 'hall');

% Dense reverb tail
noise = randn(numSamples, 1) * 0.15;
tail = noise .* decay;

ir = ir + tail;

% Stereo
ir = [ir, delayAndDecorrelate(ir, fs)];
ir = ir / max(abs(ir(:)));
end

function ir = generateChamberIR(numSamples, fs, params)
% Generate small chamber IR

rt60 = 0.2;  % Very short

t = (0:numSamples-1)' / fs;
decay = exp(-6.91 * t / rt60);

ir = generateEarlyReflections(numSamples, fs, 'chamber');

noise = randn(numSamples, 1) * 0.05;
tail = noise .* decay;

ir = ir + tail;
ir = [ir, delayAndDecorrelate(ir, fs)];
ir = ir / max(abs(ir(:)));
end

function ir = generatePlateIR(numSamples, fs, params)
% Generate plate reverb IR

rt60 = 1.8;

t = (0:numSamples-1)' / fs;
decay = exp(-6.91 * t / rt60);

% Plate has dense, smooth reflections
noise = randn(numSamples, 1);

% Multiple passes through allpass filters for density
ir = noise;
for i = 1:8
    delay = round(fs * (0.005 + 0.003 * rand()));
    ir = allPassFilter(ir, delay, 0.7);
end

ir = ir .* decay;

ir = [ir, delayAndDecorrelate(ir, fs)];
ir = ir / max(abs(ir(:)));
end

function ir = generateSpringIR(numSamples, fs, params)
% Generate spring reverb IR

rt60 = 1.2;

t = (0:numSamples-1)' / fs;
decay = exp(-6.91 * t / rt60);

% Spring has bouncy, resonant character
ir = randn(numSamples, 1);

% Add resonances
resonances = [150, 400, 900, 2000];  % Hz
for freq = resonances
    resonance = sin(2 * pi * freq * t) .* exp(-t * 5);
    ir = ir + resonance * 0.1;
end

ir = ir .* decay;

ir = [ir, delayAndDecorrelate(ir, fs)];
ir = ir / max(abs(ir(:)));
end

function ir = generateAmbienceIR(numSamples, fs, params)
% Generate subtle ambience IR

rt60 = 0.5;

t = (0:numSamples-1)' / fs;
decay = exp(-6.91 * t / rt60);

ir = generateEarlyReflections(numSamples, fs, 'small');

noise = randn(numSamples, 1) * 0.03;
tail = noise .* decay;

ir = ir + tail;
ir = [ir, delayAndDecorrelate(ir, fs)];
ir = ir / max(abs(ir(:)));
end

function earlyReflections = generateEarlyReflections(numSamples, fs, spaceType)
% Generate early reflections pattern

earlyReflections = zeros(numSamples, 1);

% Direct sound (impulse at start)
earlyReflections(1) = 1.0;

% Reflection times and gains depend on space type
switch lower(spaceType)
    case 'small'
        times = [0.005, 0.011, 0.023, 0.041, 0.067];
        gains = [0.6, 0.4, 0.3, 0.2, 0.15];
    case 'medium'
        times = [0.008, 0.019, 0.037, 0.061, 0.093, 0.127];
        gains = [0.5, 0.35, 0.25, 0.18, 0.12, 0.08];
    case 'large'
        times = [0.015, 0.035, 0.063, 0.098, 0.141, 0.187, 0.241];
        gains = [0.4, 0.3, 0.22, 0.16, 0.11, 0.07, 0.04];
    case {'hall', 'church'}
        times = [0.025, 0.057, 0.098, 0.153, 0.221, 0.297, 0.381, 0.473];
        gains = [0.35, 0.28, 0.21, 0.15, 0.10, 0.06, 0.04, 0.02];
    case 'chamber'
        times = [0.003, 0.007, 0.013, 0.021];
        gains = [0.7, 0.5, 0.3, 0.15];
    otherwise
        times = [0.008, 0.019, 0.037, 0.061];
        gains = [0.5, 0.35, 0.25, 0.15];
end

% Add reflections
for i = 1:length(times)
    idx = round(times(i) * fs);
    if idx <= numSamples
        earlyReflections(idx) = earlyReflections(idx) + gains(i);
    end
end
end

function decorrelated = delayAndDecorrelate(signal, fs)
% Create decorrelated version for stereo

% Small delay
delay = round(0.001 * fs);  % 1ms
decorrelated = [zeros(delay, 1); signal(1:end-delay)];

% Add slight filtering difference
[b, a] = butter(1, 0.95, 'low');
decorrelated = filtfilt(b, a, decorrelated);
end

function output = allPassFilter(input, delay, gain)
% All-pass filter for dense reflections

output = zeros(size(input));
buffer = zeros(delay, 1);
idx = 1;

for i = 1:length(input)
    delayedSample = buffer(idx);
    output(i) = -gain * input(i) + delayedSample + gain * output(max(1, i-1));
    buffer(idx) = input(i);
    idx = mod(idx, delay) + 1;
end
end

function dampedIR = applyDampingToIR(ir, damping, fs)
% Apply high-frequency damping to IR

cutoff = (1 - damping) * 0.8;  % 0 damping = cutoff at 0.8*Nyquist
[b, a] = butter(2, cutoff, 'low');
dampedIR = filtfilt(b, a, ir);
end

function eqIR = applyEQToIR(ir, eq, fs)
% Apply EQ to impulse response

eqIR = ir;

% Low shelf
if eq.LowGain ~= 0
    lowShelf = designfilt('lowshelf', 'FilterOrder', 2, ...
        'HalfPowerFrequency', 250, 'SampleRate', fs);
    eqIR = filtfilt(lowShelf, eqIR);
    eqIR = eqIR * db2mag(eq.LowGain);
end

% High shelf
if eq.HighGain ~= 0
    highShelf = designfilt('highshelf', 'FilterOrder', 2, ...
        'HalfPowerFrequency', 4000, 'SampleRate', fs);
    eqIR = filtfilt(highShelf, eqIR);
    eqIR = eqIR * db2mag(eq.HighGain);
end
end

function widened = applyStereoWidth(audio, width)
% Apply stereo width adjustment

if size(audio, 2) ~= 2
    widened = audio;
    return;
end

% M/S processing
mid = (audio(:, 1) + audio(:, 2)) / 2;
side = (audio(:, 1) - audio(:, 2)) / 2;

% Adjust side
side = side * width;

% Back to L/R
widened = zeros(size(audio));
widened(:, 1) = mid + side;
widened(:, 2) = mid - side;
end

function result = ternary(condition, trueVal, falseVal)
% Ternary operator helper
if condition
    result = trueVal;
else
    result = falseVal;
end
end

function info = getInfo(reverb)
% Get reverb information

info = struct();
info.hasIR = ~isempty(reverb.IR);

if info.hasIR
    info.IRName = reverb.IRName;
    info.IRLength = size(reverb.IR, 1) / reverb.IRSampleRate;
end

info.wetLevel = reverb.WetLevel;
info.dryLevel = reverb.DryLevel;
info.preDelay = reverb.PreDelay;
info.stereoWidth = reverb.StereoWidth;
info.damping = reverb.Damping;

fprintf('\n=== Convolution Reverb Settings ===\n');
if info.hasIR
    fprintf('Loaded IR: %s (%.2f seconds)\n', info.IRName, info.IRLength);
else
    fprintf('No IR loaded\n');
end
fprintf('Wet/Dry: %.0f%% / %.0f%%\n', info.wetLevel*100, info.dryLevel*100);
fprintf('Pre-delay: %.0f ms\n', info.preDelay*1000);
fprintf('Stereo Width: %.1f\n', info.stereoWidth);
fprintf('Damping: %.0f%%\n', info.damping*100);
fprintf('=====================================\n\n');
end
