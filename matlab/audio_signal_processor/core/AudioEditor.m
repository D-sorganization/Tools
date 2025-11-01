function editor = AudioEditor(audioData, sampleRate)
%AUDIOEDITOR Audio waveform editing and manipulation tool
%
%   EDITOR = AUDIOEDITOR(AUDIODATA, SAMPLERATE) creates an audio editor
%   object with comprehensive editing capabilities including trimming,
%   cutting, fading, normalization, and more.
%
%   Input Arguments:
%   ---------------
%   AUDIODATA - Audio data matrix (samples x channels)
%   SAMPLERATE - Sample rate in Hz
%
%   Properties:
%   ----------
%   AudioData - Current audio data
%   SampleRate - Sample rate in Hz
%   Selection - Current selection [startTime, endTime] in seconds
%   History - Undo history stack
%   Clipboard - Clipboard for copy/paste operations
%
%   Editing Methods:
%   ---------------
%   setSelection(startTime, endTime) - Set time selection
%   trim() - Keep selection, delete rest
%   cut() - Remove selection and copy to clipboard
%   copy() - Copy selection to clipboard
%   paste(position) - Paste clipboard at position
%   delete() - Delete selection (replace with silence)
%   crop() - Keep selection only (same as trim)
%   insertSilence(position, duration) - Insert silence
%
%   Fade Methods:
%   ------------
%   fadeIn(duration, curve) - Apply fade in
%   fadeOut(duration, curve) - Apply fade out
%   crossfade(audio2, duration, curve) - Crossfade with another audio
%   applyEnvelope(envelope) - Apply custom amplitude envelope
%
%   Gain Methods:
%   ------------
%   normalize(method, targetLevel) - Normalize audio
%   changeGain(gainDB) - Apply gain in dB
%   removeOffset() - Remove DC offset
%   invert() - Invert phase (multiply by -1)
%
%   Time Methods:
%   ------------
%   reverse() - Reverse audio
%   timeStretch(factor) - Change duration without pitch change
%   pitchShift(semitones) - Change pitch without duration change
%   changeSpeed(factor) - Change both speed and pitch
%
%   Utility Methods:
%   ---------------
%   removeSilence(threshold, minDuration) - Remove silent regions
%   detectSilence(threshold) - Detect silent regions
%   split(time) - Split audio at time position
%   merge(audio2, position) - Merge another audio at position
%   duplicate() - Duplicate audio (append copy)
%
%   History Methods:
%   ---------------
%   undo() - Undo last operation
%   redo() - Redo last undone operation
%   clearHistory() - Clear undo/redo history
%
%   Export Methods:
%   --------------
%   getAudio() - Get current audio data
%   getSelection() - Get selected audio region
%   export(filename, options) - Export to file
%
%   Example:
%   --------
%   % Load and edit audio
%   [audio, fs] = audioread('speech.wav');
%   editor = AudioEditor(audio, fs);
%
%   % Trim first 2 seconds
%   editor.setSelection(0, 2);
%   editor.delete();
%
%   % Apply fade in/out
%   editor.fadeIn(0.5, 'linear');
%   editor.fadeOut(1.0, 'exponential');
%
%   % Normalize
%   editor.normalize('lufs', -16);
%
%   % Remove silence
%   editor.removeSilence(0.01, 0.5);
%
%   % Export
%   processedAudio = editor.getAudio();
%   audiowrite('processed.wav', processedAudio, fs);
%
%   See also: MixerCoreEnhanced, AudioEffects

arguments
    audioData (:,:) double
    sampleRate (1,1) double {mustBePositive}
end

% Initialize editor structure
editor = struct();
editor.AudioData = audioData;
editor.SampleRate = sampleRate;
editor.Selection = []; % [startTime, endTime] in seconds
editor.Clipboard = [];
editor.History = {};
editor.HistoryIndex = 0;
editor.MaxHistorySize = 50;

% Add methods - Selection
editor.setSelection = @(startTime, endTime) setSelection(editor, startTime, endTime);
editor.clearSelection = @() clearSelection(editor);
editor.selectAll = @() selectAll(editor);
editor.hasSelection = @() hasSelection(editor);

% Add methods - Basic editing
editor.trim = @() trim(editor);
editor.cut = @() cut(editor);
editor.copy = @() copy(editor);
editor.paste = @(position) paste(editor, position);
editor.delete = @() deleteSelection(editor);
editor.crop = @() trim(editor);  % Alias for trim
editor.insertSilence = @(position, duration) insertSilence(editor, position, duration);

% Add methods - Fades
editor.fadeIn = @(duration, curve) fadeIn(editor, duration, curve);
editor.fadeOut = @(duration, curve) fadeOut(editor, duration, curve);
editor.crossfade = @(audio2, duration, curve) crossfade(editor, audio2, duration, curve);
editor.applyEnvelope = @(envelope) applyEnvelope(editor, envelope);

% Add methods - Gain
editor.normalize = @(method, targetLevel) normalize(editor, method, targetLevel);
editor.changeGain = @(gainDB) changeGain(editor, gainDB);
editor.removeOffset = @() removeOffset(editor);
editor.invert = @() invert(editor);

% Add methods - Time
editor.reverse = @() reverseAudio(editor);
editor.timeStretch = @(factor) timeStretch(editor, factor);
editor.pitchShift = @(semitones) pitchShift(editor, semitones);
editor.changeSpeed = @(factor) changeSpeed(editor, factor);

% Add methods - Utility
editor.removeSilence = @(threshold, minDuration) removeSilence(editor, threshold, minDuration);
editor.detectSilence = @(threshold) detectSilence(editor, threshold);
editor.split = @(time) splitAudio(editor, time);
editor.merge = @(audio2, position) merge(editor, audio2, position);
editor.duplicate = @() duplicate(editor);

% Add methods - History
editor.undo = @() undo(editor);
editor.redo = @() redo(editor);
editor.clearHistory = @() clearHistory(editor);

% Add methods - Export
editor.getAudio = @() getAudio(editor);
editor.getSelection = @() getSelection(editor);
editor.export = @(filename, varargin) exportAudio(editor, filename, varargin{:});

% Add methods - Info
editor.getDuration = @() size(editor.AudioData, 1) / editor.SampleRate;
editor.getInfo = @() getInfo(editor);
end

%% Selection Methods

function setSelection(editor, startTime, endTime)
% Set time selection in seconds

arguments
    editor
    startTime (1,1) double {mustBeNonnegative}
    endTime (1,1) double {mustBeNonnegative}
end

duration = size(editor.AudioData, 1) / editor.SampleRate;

if startTime > endTime
    error('AudioEditor:InvalidSelection', 'Start time must be less than end time');
end

if endTime > duration
    error('AudioEditor:InvalidSelection', 'End time exceeds audio duration');
end

editor.Selection = [startTime, endTime];
end

function clearSelection(editor)
% Clear current selection
editor.Selection = [];
end

function selectAll(editor)
% Select entire audio
duration = size(editor.AudioData, 1) / editor.SampleRate;
editor.Selection = [0, duration];
end

function hasSelectionFlag = hasSelection(editor)
% Check if there is an active selection
hasSelectionFlag = ~isempty(editor.Selection);
end

function [startSample, endSample] = getSelectionSamples(editor)
% Convert selection time to samples

if ~editor.hasSelection()
    startSample = 1;
    endSample = size(editor.AudioData, 1);
else
    startSample = max(1, round(editor.Selection(1) * editor.SampleRate) + 1);
    endSample = min(size(editor.AudioData, 1), round(editor.Selection(2) * editor.SampleRate));
end
end

%% Basic Editing Methods

function trim(editor)
% Keep selection, delete everything else

if ~editor.hasSelection()
    warning('AudioEditor:NoSelection', 'No selection to trim');
    return;
end

saveToHistory(editor);

[startSample, endSample] = getSelectionSamples(editor);
editor.AudioData = editor.AudioData(startSample:endSample, :);
editor.Selection = [];
end

function cut(editor)
% Cut selection to clipboard

if ~editor.hasSelection()
    warning('AudioEditor:NoSelection', 'No selection to cut');
    return;
end

% Copy to clipboard first
editor.copy();

% Then delete
saveToHistory(editor);

[startSample, endSample] = getSelectionSamples(editor);
editor.AudioData = [editor.AudioData(1:startSample-1, :);
                    editor.AudioData(endSample+1:end, :)];
editor.Selection = [];
end

function copy(editor)
% Copy selection to clipboard

if ~editor.hasSelection()
    warning('AudioEditor:NoSelection', 'No selection to copy');
    return;
end

[startSample, endSample] = getSelectionSamples(editor);
editor.Clipboard = editor.AudioData(startSample:endSample, :);
end

function paste(editor, position)
% Paste clipboard at position

arguments
    editor
    position (1,1) double {mustBeNonnegative}
end

if isempty(editor.Clipboard)
    warning('AudioEditor:EmptyClipboard', 'Nothing to paste');
    return;
end

saveToHistory(editor);

insertSample = round(position * editor.SampleRate) + 1;
insertSample = min(insertSample, size(editor.AudioData, 1) + 1);

% Insert clipboard content
editor.AudioData = [editor.AudioData(1:insertSample-1, :);
                    editor.Clipboard;
                    editor.AudioData(insertSample:end, :)];
end

function deleteSelection(editor)
% Delete selection (replace with silence)

if ~editor.hasSelection()
    warning('AudioEditor:NoSelection', 'No selection to delete');
    return;
end

saveToHistory(editor);

[startSample, endSample] = getSelectionSamples(editor);
editor.AudioData(startSample:endSample, :) = 0;
editor.Selection = [];
end

function insertSilence(editor, position, duration)
% Insert silence at position

arguments
    editor
    position (1,1) double {mustBeNonnegative}
    duration (1,1) double {mustBePositive}
end

saveToHistory(editor);

insertSample = round(position * editor.SampleRate) + 1;
silenceSamples = round(duration * editor.SampleRate);
numChannels = size(editor.AudioData, 2);

silence = zeros(silenceSamples, numChannels);

editor.AudioData = [editor.AudioData(1:insertSample-1, :);
                    silence;
                    editor.AudioData(insertSample:end, :)];
end

%% Fade Methods

function fadeIn(editor, duration, curve)
% Apply fade in

arguments
    editor
    duration (1,1) double {mustBePositive}
    curve (1,1) string {mustBeMember(curve, ["linear", "exponential", "logarithmic", "scurve"])} = "linear"
end

saveToHistory(editor);

fadeInSamples = min(round(duration * editor.SampleRate), size(editor.AudioData, 1));
fadeEnvelope = createFadeEnvelope(fadeInSamples, curve, 'in', size(editor.AudioData, 2));

editor.AudioData(1:fadeInSamples, :) = editor.AudioData(1:fadeInSamples, :) .* fadeEnvelope;
end

function fadeOut(editor, duration, curve)
% Apply fade out

arguments
    editor
    duration (1,1) double {mustBePositive}
    curve (1,1) string {mustBeMember(curve, ["linear", "exponential", "logarithmic", "scurve"])} = "linear"
end

saveToHistory(editor);

numSamples = size(editor.AudioData, 1);
fadeOutSamples = min(round(duration * editor.SampleRate), numSamples);
fadeEnvelope = createFadeEnvelope(fadeOutSamples, curve, 'out', size(editor.AudioData, 2));

startIdx = numSamples - fadeOutSamples + 1;
editor.AudioData(startIdx:end, :) = editor.AudioData(startIdx:end, :) .* fadeEnvelope;
end

function crossfade(editor, audio2, duration, curve)
% Crossfade with another audio

arguments
    editor
    audio2 (:,:) double
    duration (1,1) double {mustBePositive}
    curve (1,1) string = "linear"
end

saveToHistory(editor);

% Ensure same number of channels
if size(audio2, 2) ~= size(editor.AudioData, 2)
    error('AudioEditor:ChannelMismatch', 'Audio files must have same number of channels');
end

crossfadeSamples = min([round(duration * editor.SampleRate), ...
                         size(editor.AudioData, 1), ...
                         size(audio2, 1)]);

% Create fade envelopes
fadeOutEnv = createFadeEnvelope(crossfadeSamples, curve, 'out', size(editor.AudioData, 2));
fadeInEnv = createFadeEnvelope(crossfadeSamples, curve, 'in', size(audio2, 2));

% Get overlap regions
audio1End = editor.AudioData(end-crossfadeSamples+1:end, :);
audio2Start = audio2(1:crossfadeSamples, :);

% Apply crossfade
crossfaded = audio1End .* fadeOutEnv + audio2Start .* fadeInEnv;

% Reconstruct audio
editor.AudioData = [editor.AudioData(1:end-crossfadeSamples, :);
                    crossfaded;
                    audio2(crossfadeSamples+1:end, :)];
end

function applyEnvelope(editor, envelope)
% Apply custom amplitude envelope

arguments
    editor
    envelope (:,1) double
end

saveToHistory(editor);

numSamples = size(editor.AudioData, 1);
numChannels = size(editor.AudioData, 2);

% Interpolate envelope if needed
if length(envelope) ~= numSamples
    envelope = interp1(linspace(0, 1, length(envelope)), envelope, ...
                      linspace(0, 1, numSamples), 'linear');
end

% Apply to all channels
for ch = 1:numChannels
    editor.AudioData(:, ch) = editor.AudioData(:, ch) .* envelope;
end
end

%% Gain Methods

function normalize(editor, method, targetLevel)
% Normalize audio

arguments
    editor
    method (1,1) string {mustBeMember(method, ["peak", "rms", "lufs"])}
    targetLevel (1,1) double = -3  % in dB
end

saveToHistory(editor);

switch method
    case "peak"
        currentPeak = max(abs(editor.AudioData(:)));
        targetLinear = db2mag(targetLevel);
        gain = targetLinear / currentPeak;

    case "rms"
        currentRMS = rms(editor.AudioData(:));
        targetLinear = db2mag(targetLevel);
        gain = targetLinear / currentRMS;

    case "lufs"
        % Simplified LUFS calculation (proper implementation needs K-weighting)
        currentLUFS = calculateLUFS(editor.AudioData, editor.SampleRate);
        gain = db2mag(targetLevel - currentLUFS);
end

editor.AudioData = editor.AudioData * gain;
end

function changeGain(editor, gainDB)
% Apply gain in dB

arguments
    editor
    gainDB (1,1) double
end

saveToHistory(editor);

gain = db2mag(gainDB);
editor.AudioData = editor.AudioData * gain;
end

function removeOffset(editor)
% Remove DC offset

saveToHistory(editor);

% Remove mean from each channel
for ch = 1:size(editor.AudioData, 2)
    editor.AudioData(:, ch) = editor.AudioData(:, ch) - mean(editor.AudioData(:, ch));
end
end

function invert(editor)
% Invert phase

saveToHistory(editor);
editor.AudioData = -editor.AudioData;
end

%% Time Methods

function reverseAudio(editor)
% Reverse audio

saveToHistory(editor);
editor.AudioData = flipud(editor.AudioData);
end

function timeStretch(editor, factor)
% Time stretch without pitch change

arguments
    editor
    factor (1,1) double {mustBePositive}
end

saveToHistory(editor);

% Use AudioEffects if available, otherwise simple resampling
try
    editor.AudioData = AudioEffects(editor.AudioData, 'TimeStretch', ...
        'TimeStretch', factor, 'SampleRate', editor.SampleRate);
catch
    % Fallback: simple interpolation (will change pitch)
    warning('AudioEditor:TimeStretch', 'Advanced time stretching not available, using simple resampling');
    numSamples = round(size(editor.AudioData, 1) * factor);
    newTime = linspace(1, size(editor.AudioData, 1), numSamples);
    editor.AudioData = interp1(1:size(editor.AudioData, 1), editor.AudioData, newTime, 'spline');
end
end

function pitchShift(editor, semitones)
% Pitch shift without duration change

arguments
    editor
    semitones (1,1) double
end

if semitones == 0
    return;
end

saveToHistory(editor);

% Use AudioEffects if available
try
    editor.AudioData = AudioEffects(editor.AudioData, 'PitchShift', ...
        'PitchShift', semitones, 'SampleRate', editor.SampleRate);
catch
    warning('AudioEditor:PitchShift', 'Pitch shifting not available');
end
end

function changeSpeed(editor, factor)
% Change speed (affects both duration and pitch)

arguments
    editor
    factor (1,1) double {mustBePositive}
end

saveToHistory(editor);

% Resample audio
newLength = round(size(editor.AudioData, 1) / factor);
editor.AudioData = resample(editor.AudioData, newLength, size(editor.AudioData, 1));
end

%% Utility Methods

function removeSilence(editor, threshold, minDuration)
% Remove silent regions

arguments
    editor
    threshold (1,1) double {mustBePositive} = 0.01
    minDuration (1,1) double {mustBePositive} = 0.5  % seconds
end

saveToHistory(editor);

silentRegions = editor.detectSilence(threshold);

% Filter by minimum duration
minSamples = round(minDuration * editor.SampleRate);
validRegions = silentRegions(silentRegions(:, 2) - silentRegions(:, 1) >= minSamples, :);

% Remove silent regions (from end to beginning to preserve indices)
for i = size(validRegions, 1):-1:1
    startSample = validRegions(i, 1);
    endSample = validRegions(i, 2);
    editor.AudioData(startSample:endSample, :) = [];
end
end

function silentRegions = detectSilence(editor, threshold)
% Detect silent regions

arguments
    editor
    threshold (1,1) double {mustBePositive} = 0.01
end

% Calculate envelope (RMS with window)
windowSize = round(0.1 * editor.SampleRate);  % 100ms window
audioMono = mean(abs(editor.AudioData), 2);
envelope = movmean(audioMono, windowSize);

% Find regions below threshold
isSilent = envelope < threshold;

% Find start and end of silent regions
silentStarts = find(diff([0; isSilent]) == 1);
silentEnds = find(diff([isSilent; 0]) == -1);

silentRegions = [silentStarts, silentEnds];
end

function [audio1, audio2] = splitAudio(editor, time)
% Split audio at time position

arguments
    editor
    time (1,1) double {mustBeNonnegative}
end

splitSample = round(time * editor.SampleRate);
splitSample = max(1, min(splitSample, size(editor.AudioData, 1)));

audio1 = editor.AudioData(1:splitSample, :);
audio2 = editor.AudioData(splitSample+1:end, :);
end

function merge(editor, audio2, position)
% Merge another audio at position (mix)

arguments
    editor
    audio2 (:,:) double
    position (1,1) double {mustBeNonnegative}
end

saveToHistory(editor);

% Ensure same number of channels
if size(audio2, 2) ~= size(editor.AudioData, 2)
    if size(audio2, 2) == 1 && size(editor.AudioData, 2) == 2
        audio2 = repmat(audio2, 1, 2);  % Mono to stereo
    elseif size(audio2, 2) == 2 && size(editor.AudioData, 2) == 1
        audio2 = mean(audio2, 2);  % Stereo to mono
    else
        error('AudioEditor:ChannelMismatch', 'Incompatible channel configurations');
    end
end

startSample = round(position * editor.SampleRate) + 1;
endSample = startSample + size(audio2, 1) - 1;

% Extend audio if necessary
if endSample > size(editor.AudioData, 1)
    editor.AudioData = [editor.AudioData;
                       zeros(endSample - size(editor.AudioData, 1), size(editor.AudioData, 2))];
end

% Mix audio
editor.AudioData(startSample:endSample, :) = editor.AudioData(startSample:endSample, :) + audio2;
end

function duplicate(editor)
% Duplicate audio (append copy)

saveToHistory(editor);
editor.AudioData = [editor.AudioData; editor.AudioData];
end

%% History Methods

function saveToHistory(editor)
% Save current state to history

% Create snapshot
snapshot = struct('AudioData', editor.AudioData, 'Selection', editor.Selection);

% Remove any redo entries
if editor.HistoryIndex < length(editor.History)
    editor.History(editor.HistoryIndex+1:end) = [];
end

% Add to history
editor.History{end+1} = snapshot;
editor.HistoryIndex = length(editor.History);

% Limit history size
if length(editor.History) > editor.MaxHistorySize
    editor.History(1) = [];
    editor.HistoryIndex = editor.HistoryIndex - 1;
end
end

function undo(editor)
% Undo last operation

if editor.HistoryIndex < 1
    warning('AudioEditor:NoUndo', 'Nothing to undo');
    return;
end

% Restore previous state
snapshot = editor.History{editor.HistoryIndex};
editor.AudioData = snapshot.AudioData;
editor.Selection = snapshot.Selection;

editor.HistoryIndex = editor.HistoryIndex - 1;
end

function redo(editor)
% Redo last undone operation

if editor.HistoryIndex >= length(editor.History)
    warning('AudioEditor:NoRedo', 'Nothing to redo');
    return;
end

editor.HistoryIndex = editor.HistoryIndex + 1;

% Restore next state
snapshot = editor.History{editor.HistoryIndex};
editor.AudioData = snapshot.AudioData;
editor.Selection = snapshot.Selection;
end

function clearHistory(editor)
% Clear undo/redo history

editor.History = {};
editor.HistoryIndex = 0;
end

%% Export Methods

function audioData = getAudio(editor)
% Get current audio data
audioData = editor.AudioData;
end

function audioData = getSelection(editor)
% Get selected audio region

if ~editor.hasSelection()
    audioData = editor.AudioData;
else
    [startSample, endSample] = getSelectionSamples(editor);
    audioData = editor.AudioData(startSample:endSample, :);
end
end

function exportAudio(editor, filename, varargin)
% Export audio to file

AudioExporter(editor.AudioData, filename, 'SampleRate', editor.SampleRate, varargin{:});
end

function info = getInfo(editor)
% Get audio information

info = struct();
info.SampleRate = editor.SampleRate;
info.NumSamples = size(editor.AudioData, 1);
info.NumChannels = size(editor.AudioData, 2);
info.Duration = info.NumSamples / editor.SampleRate;
info.Peak = max(abs(editor.AudioData(:)));
info.RMS = rms(editor.AudioData(:));
info.DCOffset = mean(editor.AudioData(:));

if editor.hasSelection()
    info.Selection = editor.Selection;
    info.SelectionDuration = editor.Selection(2) - editor.Selection(1);
end
end

%% Helper Functions

function envelope = createFadeEnvelope(numSamples, curve, direction, numChannels)
% Create fade envelope

t = linspace(0, 1, numSamples)';

switch curve
    case "linear"
        envelope = t;
    case "exponential"
        envelope = t.^2;
    case "logarithmic"
        envelope = sqrt(t);
    case "scurve"
        envelope = (1 - cos(t * pi)) / 2;
    otherwise
        envelope = t;
end

if strcmp(direction, 'out')
    envelope = flip(envelope);
end

% Replicate for all channels
envelope = repmat(envelope, 1, numChannels);
end

function lufs = calculateLUFS(audioData, sampleRate)
% Simplified LUFS calculation

% This is a simplified version. Full LUFS requires K-weighting filter
% and proper gating as per ITU-R BS.1770

% Convert to mono if stereo
if size(audioData, 2) > 1
    audioData = mean(audioData, 2);
end

% Calculate RMS in dB
rmsValue = rms(audioData);
lufs = 20 * log10(rmsValue) - 23;  % LUFS reference

end
