function mixer = MixerCoreEnhanced(numTracks, sampleRate)
%MIXERCOREENHANCED Enhanced multi-track mixer with time offsets and advanced features
%
%   MIXER = MIXERCOREENHANCED(NUMTRACKS, SAMPLERATE) creates an enhanced
%   multi-track mixer with time offset support, automation, and advanced
%   mixing features.
%
%   MIXER = MIXERCOREENHANCED() creates a default 8-track mixer at 44.1 kHz.
%
%   Input Arguments:
%   ---------------
%   NUMTRACKS - Number of audio tracks (default: 8)
%   SAMPLERATE - Sample rate in Hz (default: 44100)
%
%   Enhanced Properties vs MixerCore:
%   --------------------------------
%   ADDED: StartOffset - Per-track time offset in seconds
%   ADDED: FadeIn - Per-track fade in duration in seconds
%   ADDED: FadeOut - Per-track fade out duration in seconds
%   ADDED: Automation - Per-track automation data
%   ADDED: Markers - Timeline markers and regions
%
%   Track Properties (Enhanced):
%   ---------------------------
%   AudioData - Audio data matrix (samples x channels)
%   Volume - Volume level 0-1 (default: 0.7)
%   Pan - Pan position -1 to 1 (default: 0)
%   Solo - Solo state (default: false)
%   Mute - Mute state (default: false)
%   Effects - Array of effect structures
%   IsLoaded - Whether track has audio loaded
%   StartOffset - Start time offset in seconds (NEW)
%   FadeIn - Fade in duration in seconds (NEW)
%   FadeOut - Fade out duration in seconds (NEW)
%   Automation - Struct with parameter automation (NEW)
%
%   New Methods:
%   -----------
%   setTrackOffset(trackIndex, offsetSeconds) - Set track start time offset
%   setTrackFadeIn(trackIndex, duration, curve) - Set fade in
%   setTrackFadeOut(trackIndex, duration, curve) - Set fade out
%   addAutomation(trackIndex, parameter, timePoints, values) - Add automation
%   alignTracks(method) - Auto-align tracks ('peak', 'onset', 'correlation')
%   addMarker(time, label) - Add timeline marker
%   addRegion(startTime, endTime, label) - Add timeline region
%   getTotalDuration() - Get total project duration including offsets
%   bounceToTrack(sourceIndices, destIndex) - Bounce tracks together
%
%   Example:
%   --------
%   % Create enhanced mixer
%   mixer = MixerCoreEnhanced(8, 44100);
%
%   % Load audio to tracks
%   [drum, fs1] = audioread('drums.wav');
%   [bass, fs2] = audioread('bass.wav');
%   [vocal, fs3] = audioread('vocal.wav');
%
%   mixer.loadTrack(1, drum, fs1);
%   mixer.loadTrack(2, bass, fs2);
%   mixer.loadTrack(3, vocal, fs3);
%
%   % Set time offsets (bass starts 0.5s later, vocal at 1.0s)
%   mixer.setTrackOffset(2, 0.5);
%   mixer.setTrackOffset(3, 1.0);
%
%   % Add fades
%   mixer.setTrackFadeIn(3, 0.2, 'linear');
%   mixer.setTrackFadeOut(3, 0.5, 'exponential');
%
%   % Add markers
%   mixer.addMarker(0.0, 'Intro');
%   mixer.addMarker(8.0, 'Verse');
%   mixer.addMarker(24.0, 'Chorus');
%
%   % Process mix with offsets
%   mixedAudio = mixer.processMix();
%
%   See also: MixerCore, AudioEffects, AudioEditor

arguments
    numTracks (1,1) double {mustBePositive, mustBeInteger} = 8
    sampleRate (1,1) double {mustBePositive} = 44100
end

% Initialize mixer structure
mixer = struct();
mixer.NumTracks = numTracks;
mixer.SampleRate = sampleRate;
mixer.IsPlaying = false;
mixer.CurrentPosition = 1;

% Initialize tracks with enhanced properties
mixer.Tracks = struct();
for i = 1:numTracks
    mixer.Tracks(i).AudioData = [];
    mixer.Tracks(i).Volume = 0.7;
    mixer.Tracks(i).Pan = 0.0;
    mixer.Tracks(i).Solo = false;
    mixer.Tracks(i).Mute = false;
    mixer.Tracks(i).Effects = [];
    mixer.Tracks(i).IsLoaded = false;
    mixer.Tracks(i).SampleRate = sampleRate;
    mixer.Tracks(i).Length = 0;

    % Enhanced properties
    mixer.Tracks(i).StartOffset = 0.0;  % In seconds
    mixer.Tracks(i).FadeIn = struct('Duration', 0, 'Curve', 'linear');
    mixer.Tracks(i).FadeOut = struct('Duration', 0, 'Curve', 'linear');
    mixer.Tracks(i).Automation = struct('Volume', [], 'Pan', []);
    mixer.Tracks(i).Gain = 1.0;  % Additional gain (0-2, default 1.0)
    mixer.Tracks(i).Color = [rand(), rand(), rand()];  % Track color for GUI
    mixer.Tracks(i).Name = sprintf('Track %d', i);
end

% Initialize master bus
mixer.MasterBus = struct();
mixer.MasterBus.Volume = 1.0;
mixer.MasterBus.Effects = [];
mixer.MasterBus.Limiter = struct('Enabled', true, 'Threshold', -0.1);

% Timeline features
mixer.Markers = struct('Time', {}, 'Label', {});
mixer.Regions = struct('StartTime', {}, 'EndTime', {}, 'Label', {}, 'Color', {});

% Add all methods (existing + enhanced)
mixer.loadTrack = @(trackIndex, audioData, trackSampleRate) loadTrack(mixer, trackIndex, audioData, trackSampleRate);
mixer.setTrackVolume = @(trackIndex, volume) setTrackVolume(mixer, trackIndex, volume);
mixer.setTrackPan = @(trackIndex, pan) setTrackPan(mixer, trackIndex, pan);
mixer.setTrackSolo = @(trackIndex, solo) setTrackSolo(mixer, trackIndex, solo);
mixer.setTrackMute = @(trackIndex, mute) setTrackMute(mixer, trackIndex, mute);
mixer.addEffect = @(trackIndex, effectType, params) addEffect(mixer, trackIndex, effectType, params);
mixer.removeEffect = @(trackIndex, effectIndex) removeEffect(mixer, trackIndex, effectIndex);

% Enhanced methods
mixer.setTrackOffset = @(trackIndex, offsetSeconds) setTrackOffset(mixer, trackIndex, offsetSeconds);
mixer.setTrackFadeIn = @(trackIndex, duration, curve) setTrackFadeIn(mixer, trackIndex, duration, curve);
mixer.setTrackFadeOut = @(trackIndex, duration, curve) setTrackFadeOut(mixer, trackIndex, duration, curve);
mixer.setTrackGain = @(trackIndex, gain) setTrackGain(mixer, trackIndex, gain);
mixer.setTrackName = @(trackIndex, name) setTrackName(mixer, trackIndex, name);
mixer.addAutomation = @(trackIndex, parameter, timePoints, values) addAutomation(mixer, trackIndex, parameter, timePoints, values);
mixer.alignTracks = @(method) alignTracks(mixer, method);
mixer.addMarker = @(time, label) addMarker(mixer, time, label);
mixer.addRegion = @(startTime, endTime, label, color) addRegion(mixer, startTime, endTime, label, color);
mixer.getTotalDuration = @() getTotalDuration(mixer);
mixer.bounceToTrack = @(sourceIndices, destIndex) bounceToTrack(mixer, sourceIndices, destIndex);

% Processing methods
mixer.processMix = @() processMix(mixer);
mixer.play = @() play(mixer);
mixer.pause = @() pause(mixer);
mixer.stop = @() stop(mixer);
mixer.seek = @(position) seek(mixer, position);
end

%% Enhanced Methods

function success = setTrackOffset(mixer, trackIndex, offsetSeconds)
% Set track start time offset in seconds

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    offsetSeconds (1,1) double {mustBeNonnegative}
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

mixer.Tracks(trackIndex).StartOffset = offsetSeconds;
success = true;
end

function success = setTrackFadeIn(mixer, trackIndex, duration, curve)
% Set track fade in

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    duration (1,1) double {mustBeNonnegative}
    curve (1,1) string {mustBeMember(curve, ["linear", "exponential", "logarithmic", "scurve"])} = "linear"
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

mixer.Tracks(trackIndex).FadeIn.Duration = duration;
mixer.Tracks(trackIndex).FadeIn.Curve = curve;
success = true;
end

function success = setTrackFadeOut(mixer, trackIndex, duration, curve)
% Set track fade out

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    duration (1,1) double {mustBeNonnegative}
    curve (1,1) string {mustBeMember(curve, ["linear", "exponential", "logarithmic", "scurve"])} = "linear"
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

mixer.Tracks(trackIndex).FadeOut.Duration = duration;
mixer.Tracks(trackIndex).FadeOut.Curve = curve;
success = true;
end

function success = setTrackGain(mixer, trackIndex, gain)
% Set track gain (0-2, where 1.0 is unity)

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    gain (1,1) double {mustBeInRange(gain, 0, 2)}
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

mixer.Tracks(trackIndex).Gain = gain;
success = true;
end

function success = setTrackName(mixer, trackIndex, name)
% Set track name

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    name (1,1) string
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

mixer.Tracks(trackIndex).Name = name;
success = true;
end

function success = addAutomation(mixer, trackIndex, parameter, timePoints, values)
% Add automation for a parameter

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    parameter (1,1) string {mustBeMember(parameter, ["Volume", "Pan", "Gain"])}
    timePoints (:,1) double {mustBeNonnegative}
    values (:,1) double
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

% Validate value ranges
switch parameter
    case "Volume"
        if any(values < 0 | values > 1)
            error('MixerCoreEnhanced:InvalidAutomation', 'Volume values must be in range [0, 1]');
        end
    case "Pan"
        if any(values < -1 | values > 1)
            error('MixerCoreEnhanced:InvalidAutomation', 'Pan values must be in range [-1, 1]');
        end
    case "Gain"
        if any(values < 0 | values > 2)
            error('MixerCoreEnhanced:InvalidAutomation', 'Gain values must be in range [0, 2]');
        end
end

% Store automation data
automation = struct('TimePoints', timePoints, 'Values', values);
mixer.Tracks(trackIndex).Automation.(parameter) = automation;
success = true;
end

function success = addMarker(mixer, time, label)
% Add timeline marker

arguments
    mixer
    time (1,1) double {mustBeNonnegative}
    label (1,1) string
end

marker = struct('Time', time, 'Label', label);
mixer.Markers(end+1) = marker;
success = true;
end

function success = addRegion(mixer, startTime, endTime, label, color)
% Add timeline region

arguments
    mixer
    startTime (1,1) double {mustBeNonnegative}
    endTime (1,1) double {mustBeNonnegative}
    label (1,1) string
    color (1,3) double = [0.5, 0.7, 1.0]
end

if endTime <= startTime
    error('MixerCoreEnhanced:InvalidRegion', 'End time must be greater than start time');
end

region = struct('StartTime', startTime, 'EndTime', endTime, 'Label', label, 'Color', color);
mixer.Regions(end+1) = region;
success = true;
end

function duration = getTotalDuration(mixer)
% Get total project duration including offsets

maxDuration = 0;
for i = 1:mixer.NumTracks
    if mixer.Tracks(i).IsLoaded
        trackDuration = mixer.Tracks(i).StartOffset + mixer.Tracks(i).Length / mixer.SampleRate;
        maxDuration = max(maxDuration, trackDuration);
    end
end
duration = maxDuration;
end

function success = alignTracks(mixer, method)
% Auto-align tracks using specified method

arguments
    mixer
    method (1,1) string {mustBeMember(method, ["peak", "onset", "correlation"])}
end

% Find reference track (first loaded track)
refTrackIdx = 0;
for i = 1:mixer.NumTracks
    if mixer.Tracks(i).IsLoaded
        refTrackIdx = i;
        break;
    end
end

if refTrackIdx == 0
    error('MixerCoreEnhanced:NoTracksLoaded', 'No tracks loaded for alignment');
end

refAudio = mixer.Tracks(refTrackIdx).AudioData;
if size(refAudio, 2) > 1
    refAudio = mean(refAudio, 2);  % Convert to mono for alignment
end

% Align other tracks
for i = 1:mixer.NumTracks
    if i == refTrackIdx || ~mixer.Tracks(i).IsLoaded
        continue;
    end

    trackAudio = mixer.Tracks(i).AudioData;
    if size(trackAudio, 2) > 1
        trackAudio = mean(trackAudio, 2);
    end

    switch method
        case "peak"
            offset = alignByPeak(refAudio, trackAudio, mixer.SampleRate);
        case "onset"
            offset = alignByOnset(refAudio, trackAudio, mixer.SampleRate);
        case "correlation"
            offset = alignByCorrelation(refAudio, trackAudio, mixer.SampleRate);
    end

    mixer.Tracks(i).StartOffset = offset;
end

success = true;
end

function offset = alignByPeak(refAudio, trackAudio, sampleRate)
% Align by finding peak positions

[~, refPeakIdx] = max(abs(refAudio));
[~, trackPeakIdx] = max(abs(trackAudio));

offsetSamples = refPeakIdx - trackPeakIdx;
offset = offsetSamples / sampleRate;
end

function offset = alignByOnset(refAudio, trackAudio, sampleRate)
% Align by onset detection (simplified)

% Find first significant amplitude
threshold = 0.1 * max(abs(refAudio));
refOnset = find(abs(refAudio) > threshold, 1, 'first');

threshold = 0.1 * max(abs(trackAudio));
trackOnset = find(abs(trackAudio) > threshold, 1, 'first');

if isempty(refOnset)
    refOnset = 1;
end
if isempty(trackOnset)
    trackOnset = 1;
end

offsetSamples = refOnset - trackOnset;
offset = offsetSamples / sampleRate;
end

function offset = alignByCorrelation(refAudio, trackAudio, sampleRate)
% Align using cross-correlation

% Limit search to first 10 seconds for efficiency
maxSamples = min([length(refAudio), length(trackAudio), 10 * sampleRate]);
refShort = refAudio(1:maxSamples);
trackShort = trackAudio(1:maxSamples);

% Cross-correlation
[c, lags] = xcorr(refShort, trackShort);
[~, maxIdx] = max(abs(c));
lagSamples = lags(maxIdx);

offset = -lagSamples / sampleRate;  % Negative because xcorr convention
end

function success = bounceToTrack(mixer, sourceIndices, destIndex)
% Bounce multiple tracks to a single track

arguments
    mixer
    sourceIndices (:,1) double {mustBePositive, mustBeInteger}
    destIndex (1,1) double {mustBePositive, mustBeInteger}
end

if destIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', 'Destination track index invalid');
end

% Temporarily mute all non-source tracks
originalMuteStates = [mixer.Tracks.Mute];
for i = 1:mixer.NumTracks
    if ~ismember(i, sourceIndices)
        mixer.Tracks(i).Mute = true;
    end
end

% Process mix (only source tracks)
bouncedAudio = mixer.processMix();

% Restore mute states
for i = 1:mixer.NumTracks
    mixer.Tracks(i).Mute = originalMuteStates(i);
end

% Load bounced audio to destination track
mixer.loadTrack(destIndex, bouncedAudio, mixer.SampleRate);

success = true;
end

%% Core Methods (adapted from MixerCore)

function success = loadTrack(mixer, trackIndex, audioData, trackSampleRate)
% Load audio data to specified track

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    audioData (:,:) double
    trackSampleRate (1,1) double {mustBePositive} = mixer.SampleRate
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

success = false;

try
    % Resample if necessary
    if trackSampleRate ~= mixer.SampleRate
        audioData = resample(audioData, mixer.SampleRate, trackSampleRate);
    end

    % Store track data
    mixer.Tracks(trackIndex).AudioData = audioData;
    mixer.Tracks(trackIndex).SampleRate = mixer.SampleRate;
    mixer.Tracks(trackIndex).Length = size(audioData, 1);
    mixer.Tracks(trackIndex).IsLoaded = true;

    success = true;

catch ME
    warning('MixerCoreEnhanced:LoadError', 'Error loading track %d: %s', trackIndex, ME.message);
end
end

function setTrackVolume(mixer, trackIndex, volume)
% Set track volume level

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    volume (1,1) double {mustBeInRange(volume, 0, 1)}
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

mixer.Tracks(trackIndex).Volume = volume;
end

function setTrackPan(mixer, trackIndex, pan)
% Set track pan position

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    pan (1,1) double {mustBeInRange(pan, -1, 1)}
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

mixer.Tracks(trackIndex).Pan = pan;
end

function setTrackSolo(mixer, trackIndex, solo)
% Set track solo state

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    solo (1,1) logical
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

mixer.Tracks(trackIndex).Solo = solo;
end

function setTrackMute(mixer, trackIndex, mute)
% Set track mute state

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    mute (1,1) logical
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

mixer.Tracks(trackIndex).Mute = mute;
end

function success = addEffect(mixer, trackIndex, effectType, params)
% Add effect to track

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    effectType (1,1) string
    params (1,1) struct = struct()
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

success = false;

try
    effect = struct();
    effect.Type = effectType;
    effect.Parameters = params;
    effect.Enabled = true;

    mixer.Tracks(trackIndex).Effects = [mixer.Tracks(trackIndex).Effects, effect];
    success = true;

catch ME
    warning('MixerCoreEnhanced:EffectError', 'Error adding effect to track %d: %s', trackIndex, ME.message);
end
end

function success = removeEffect(mixer, trackIndex, effectIndex)
% Remove effect from track

arguments
    mixer
    trackIndex (1,1) double {mustBePositive, mustBeInteger}
    effectIndex (1,1) double {mustBePositive, mustBeInteger}
end

if trackIndex > mixer.NumTracks
    error('MixerCoreEnhanced:InvalidTrackIndex', ...
        'Track index %d exceeds number of tracks (%d)', trackIndex, mixer.NumTracks);
end

success = false;

try
    effects = mixer.Tracks(trackIndex).Effects;
    if effectIndex <= length(effects)
        effects(effectIndex) = [];
        mixer.Tracks(trackIndex).Effects = effects;
        success = true;
    end

catch ME
    warning('MixerCoreEnhanced:EffectError', 'Error removing effect from track %d: %s', trackIndex, ME.message);
end
end

function mixedAudio = processMix(mixer)
% Process and mix all tracks WITH TIME OFFSETS

% Calculate total duration including offsets
totalDuration = mixer.getTotalDuration();
if totalDuration == 0
    mixedAudio = [];
    return;
end

maxSamples = ceil(totalDuration * mixer.SampleRate);

% Initialize mix buffer
mixedAudio = zeros(maxSamples, 2); % Stereo output

% Check if any track is soloed
hasSolo = any([mixer.Tracks.Solo]);

% Process each loaded track
for trackIdx = 1:mixer.NumTracks
    track = mixer.Tracks(trackIdx);

    if ~track.IsLoaded
        continue;
    end

    % Skip if muted or soloed (when other tracks are soloed)
    if track.Mute || (hasSolo && ~track.Solo)
        continue;
    end

    % Get track audio
    trackAudio = track.AudioData;

    % Apply effects
    for effectIdx = 1:length(track.Effects)
        effect = track.Effects(effectIdx);
        if effect.Enabled
            try
                params = effect.Parameters;
                params.SampleRate = mixer.SampleRate;
                trackAudio = AudioEffects(trackAudio, effect.Type, params);
            catch ME
                warning('MixerCoreEnhanced:EffectProcessError', ...
                    'Error processing effect %s on track %d: %s', ...
                    effect.Type, trackIdx, ME.message);
            end
        end
    end

    % Apply fades
    trackAudio = applyFades(trackAudio, track.FadeIn, track.FadeOut, mixer.SampleRate);

    % Apply gain
    trackAudio = trackAudio * track.Gain;

    % Apply automation (volume, pan)
    trackAudio = applyAutomation(trackAudio, track.Automation, track.Volume, track.Pan, mixer.SampleRate);

    % Apply volume and pan (if no automation)
    if isempty(track.Automation.Volume) && isempty(track.Automation.Pan)
        trackAudio = applyVolumeAndPan(trackAudio, track.Volume, track.Pan);
    end

    % Calculate start position based on offset
    startSample = round(track.StartOffset * mixer.SampleRate) + 1;
    endSample = startSample + size(trackAudio, 1) - 1;

    % Ensure buffer is large enough
    if endSample > size(mixedAudio, 1)
        mixedAudio = [mixedAudio; zeros(endSample - size(mixedAudio, 1), 2)];
    end

    % Mix to output at offset position
    trackLength = size(trackAudio, 1);
    if size(trackAudio, 2) == 1
        % Mono to stereo
        mixedAudio(startSample:endSample, 1) = mixedAudio(startSample:endSample, 1) + trackAudio(:, 1);
        mixedAudio(startSample:endSample, 2) = mixedAudio(startSample:endSample, 2) + trackAudio(:, 1);
    else
        % Stereo
        mixedAudio(startSample:endSample, :) = mixedAudio(startSample:endSample, :) + trackAudio;
    end
end

% Apply master bus processing
mixedAudio = applyMasterBusProcessing(mixedAudio, mixer.MasterBus, mixer.SampleRate);
end

function processedAudio = applyFades(audioData, fadeIn, fadeOut, sampleRate)
% Apply fade in and fade out

processedAudio = audioData;
numSamples = size(audioData, 1);

% Apply fade in
if fadeIn.Duration > 0
    fadeInSamples = min(round(fadeIn.Duration * sampleRate), numSamples);
    fadeInEnvelope = createFadeEnvelope(fadeInSamples, fadeIn.Curve, 'in');
    processedAudio(1:fadeInSamples, :) = processedAudio(1:fadeInSamples, :) .* fadeInEnvelope;
end

% Apply fade out
if fadeOut.Duration > 0
    fadeOutSamples = min(round(fadeOut.Duration * sampleRate), numSamples);
    fadeOutEnvelope = createFadeEnvelope(fadeOutSamples, fadeOut.Curve, 'out');
    startIdx = max(1, numSamples - fadeOutSamples + 1);
    processedAudio(startIdx:end, :) = processedAudio(startIdx:end, :) .* fadeOutEnvelope;
end
end

function envelope = createFadeEnvelope(numSamples, curve, direction)
% Create fade envelope based on curve type

t = linspace(0, 1, numSamples)';

switch curve
    case "linear"
        envelope = t;
    case "exponential"
        envelope = t.^2;
    case "logarithmic"
        envelope = sqrt(t);
    case "scurve"
        envelope = (1 - cos(t * pi)) / 2;  % S-curve (sine)
    otherwise
        envelope = t;
end

if strcmp(direction, 'out')
    envelope = flip(envelope);
end

% Replicate for stereo
envelope = repmat(envelope, 1, 2);
end

function processedAudio = applyAutomation(audioData, automation, baseVolume, basePan, sampleRate)
% Apply automation curves

processedAudio = audioData;
numSamples = size(audioData, 1);

% Apply volume automation
if ~isempty(automation.Volume)
    timePoints = automation.Volume.TimePoints;
    values = automation.Volume.Values;

    % Interpolate automation to sample rate
    sampleIndices = (0:numSamples-1)' / sampleRate;
    volumeEnvelope = interp1(timePoints, values, sampleIndices, 'linear', baseVolume);
    processedAudio = processedAudio .* volumeEnvelope;
else
    processedAudio = processedAudio * baseVolume;
end

% Apply pan automation
if ~isempty(automation.Pan)
    timePoints = automation.Pan.TimePoints;
    values = automation.Pan.Values;

    % Interpolate automation
    sampleIndices = (0:numSamples-1)' / sampleRate;
    panEnvelope = interp1(timePoints, values, sampleIndices, 'linear', basePan);

    % Apply pan curve
    for i = 1:numSamples
        pan = panEnvelope(i);
        leftGain = sqrt(0.5 * (1 - pan));
        rightGain = sqrt(0.5 * (1 + pan));

        if size(processedAudio, 2) == 1
            % Mono
            processedAudio(i, 1) = processedAudio(i, 1) * leftGain;
            processedAudio = [processedAudio, processedAudio(:,1) * rightGain / leftGain];
        else
            % Stereo
            processedAudio(i, 1) = processedAudio(i, 1) * leftGain;
            processedAudio(i, 2) = processedAudio(i, 2) * rightGain;
        end
    end
elseif size(processedAudio, 2) == 1
    % Convert mono to stereo with base pan
    processedAudio = applyVolumeAndPan(processedAudio, 1.0, basePan);
end
end

function processedAudio = applyVolumeAndPan(audioData, volume, pan)
% Apply volume and pan to audio data (from original MixerCore)

processedAudio = audioData * volume;

% Apply panning
if size(processedAudio, 2) == 1
    % Mono to stereo with pan
    leftGain = sqrt(0.5 * (1 - pan));
    rightGain = sqrt(0.5 * (1 + pan));
    processedAudio = [processedAudio * leftGain, processedAudio * rightGain];
elseif size(processedAudio, 2) == 2
    % Stereo panning
    leftGain = sqrt(0.5 * (1 - pan));
    rightGain = sqrt(0.5 * (1 + pan));
    processedAudio(:, 1) = processedAudio(:, 1) * leftGain;
    processedAudio(:, 2) = processedAudio(:, 2) * rightGain;
end
end

function processedAudio = applyMasterBusProcessing(audioData, masterBus, sampleRate)
% Apply master bus effects and processing (from original MixerCore)

processedAudio = audioData;

% Apply master volume
processedAudio = processedAudio * masterBus.Volume;

% Apply master effects
for effectIdx = 1:length(masterBus.Effects)
    effect = masterBus.Effects(effectIdx);
    if effect.Enabled
        try
            params = effect.Parameters;
            params.SampleRate = sampleRate;
            processedAudio = AudioEffects(processedAudio, effect.Type, params);
        catch ME
            warning('MixerCoreEnhanced:MasterEffectError', ...
                'Error processing master effect %s: %s', effect.Type, ME.message);
        end
    end
end

% Apply limiter
if masterBus.Limiter.Enabled
    limit = db2mag(masterBus.Limiter.Threshold);
    processedAudio(processedAudio > limit) = limit;
    processedAudio(processedAudio < -limit) = -limit;
end
end

function play(mixer)
% Start playback
mixer.IsPlaying = true;
end

function pause(mixer)
% Pause playback
mixer.IsPlaying = false;
end

function stop(mixer)
% Stop playback and reset position
mixer.IsPlaying = false;
mixer.CurrentPosition = 1;
end

function seek(mixer, position)
% Seek to position

arguments
    mixer
    position (1,1) double {mustBeNonnegative, mustBeInteger}
end

mixer.CurrentPosition = position;
end
