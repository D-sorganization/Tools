function [audioData, sampleRate, info] = AudioLoader(filename, varargin)
%AUDIOLOADER Load audio files with multi-format support and metadata extraction
%
%   [AUDIODATA, SAMPLERATE, INFO] = AUDIOLOADER(FILENAME) loads an audio file
%   and returns the audio data, sample rate, and metadata information.
%
%   [AUDIODATA, SAMPLERATE, INFO] = AUDIOLOADER(FILENAME, 'Property', Value, ...)
%   specifies additional options using property-value pairs.
%
%   Input Arguments:
%   ---------------
%   FILENAME - String or character vector specifying the audio file path
%
%   Optional Properties:
%   ------------------
%   'ChunkSize'     - Size of chunks for large file processing (default: 1e6)
%   'TargetRate'    - Target sample rate for conversion (default: original)
%   'Channels'      - Channel selection: 'all', 'left', 'right', 'mono' (default: 'all')
%   'Duration'      - Duration to load in seconds (default: all)
%   'StartTime'     - Start time in seconds (default: 0)
%   'Normalize'     - Normalize audio to [-1, 1] range (default: false)
%   'Metadata'      - Extract metadata (default: true)
%
%   Output Arguments:
%   ----------------
%   AUDIODATA - Audio data matrix (samples x channels)
%   SAMPLERATE - Sample rate in Hz
%   INFO - Structure containing metadata and file information
%
%   Supported Formats:
%   -----------------
%   - WAV (.wav)
%   - MP3 (.mp3) - requires Audio Toolbox
%   - FLAC (.flac) - requires Audio Toolbox
%   - OGG (.ogg) - requires Audio Toolbox
%   - M4A (.m4a) - requires Audio Toolbox
%
%   Example:
%   --------
%   % Load entire file
%   [data, fs, info] = AudioLoader('song.wav');
%
%   % Load first 30 seconds, convert to mono
%   [data, fs, info] = AudioLoader('song.wav', 'Duration', 30, 'Channels', 'mono');
%
%   % Load with sample rate conversion
%   [data, fs, info] = AudioLoader('song.wav', 'TargetRate', 44100);
%
%   See also: audioread, audioinfo, MetadataExtractor

arguments
    filename (1,1) string {mustBeFile}
    options.ChunkSize (1,1) double {mustBePositive} = 1e6
    options.TargetRate (1,1) double {mustBePositive} = []
    options.Channels (1,1) string {mustBeMember(options.Channels, ["all", "left", "right", "mono"])} = "all"
    options.Duration (1,1) double {mustBeNonnegative} = []
    options.StartTime (1,1) double {mustBeNonnegative} = 0
    options.Normalize (1,1) logical = false
    options.Metadata (1,1) logical = true
end

% Validate file exists
if ~isfile(filename)
    error('AudioLoader:FileNotFound', 'File "%s" not found.', filename);
end

% Get file information
try
    fileInfo = audioinfo(filename);
catch ME
    error('AudioLoader:UnsupportedFormat', ...
        'Cannot read audio file "%s": %s', filename, ME.message);
end

% Extract basic information
originalSampleRate = fileInfo.SampleRate;
totalSamples = fileInfo.TotalSamples;
numChannels = fileInfo.NumChannels;
duration = fileInfo.Duration;

% Determine target sample rate
if isempty(options.TargetRate)
    targetSampleRate = originalSampleRate;
else
    targetSampleRate = options.TargetRate;
end

% Calculate sample range
startSample = round(options.StartTime * originalSampleRate) + 1;
if isempty(options.Duration)
    endSample = totalSamples;
else
    endSample = min(startSample + round(options.Duration * originalSampleRate) - 1, totalSamples);
end

% Check if file is too large for memory
samplesToLoad = endSample - startSample + 1;
if samplesToLoad > options.ChunkSize
    % Use chunked loading
    audioData = loadAudioInChunks(filename, startSample, endSample, options.ChunkSize);
else
    % Load entire range at once
    try
        audioData = audioread(filename, [startSample, endSample]);
    catch ME
        error('AudioLoader:ReadError', 'Error reading audio file: %s', ME.message);
    end
end

% Handle channel selection
audioData = selectChannels(audioData, options.Channels);

% Sample rate conversion if needed
if targetSampleRate ~= originalSampleRate
    audioData = resampleAudio(audioData, originalSampleRate, targetSampleRate);
    sampleRate = targetSampleRate;
else
    sampleRate = originalSampleRate;
end

% Normalize if requested
if options.Normalize
    maxVal = max(abs(audioData(:)));
    if maxVal > 0
        audioData = audioData / maxVal;
    end
end

% Extract metadata
if options.Metadata
    info = extractMetadata(filename, fileInfo, audioData, sampleRate);
else
    info = struct('Filename', filename, 'SampleRate', sampleRate, ...
        'Duration', size(audioData,1)/sampleRate, ...
        'Channels', size(audioData,2));
end

% Add processing information
info.OriginalSampleRate = originalSampleRate;
info.TargetSampleRate = targetSampleRate;
info.ProcessedChannels = options.Channels;
info.Normalized = options.Normalize;
end

function audioData = loadAudioInChunks(filename, startSample, endSample, chunkSize)
% Load large audio files in chunks to manage memory

totalSamples = endSample - startSample + 1;
numChunks = ceil(totalSamples / chunkSize);

% Get file info to determine number of channels
fileInfo = audioinfo(filename);
numChannels = fileInfo.NumChannels;

% Pre-allocate output array
audioData = zeros(totalSamples, numChannels);

% Load chunks
for i = 1:numChunks
    chunkStart = startSample + (i-1) * chunkSize;
    chunkEnd = min(chunkStart + chunkSize - 1, endSample);

    try
        chunk = audioread(filename, [chunkStart, chunkEnd]);
        chunkIdx = (1:size(chunk,1)) + (i-1) * chunkSize;
        audioData(chunkIdx, :) = chunk;
    catch ME
        warning('AudioLoader:ChunkError', ...
            'Error loading chunk %d: %s', i, ME.message);
    end
end
end

function audioData = selectChannels(audioData, channelOption)
% Select channels based on option

switch channelOption
    case "all"
        % Keep all channels
    case "left"
        audioData = audioData(:, 1);
    case "right"
        if size(audioData, 2) >= 2
            audioData = audioData(:, 2);
        else
            audioData = audioData(:, 1);
        end
    case "mono"
        if size(audioData, 2) > 1
            audioData = mean(audioData, 2);
        end
end
end

function audioData = resampleAudio(audioData, originalRate, targetRate)
% Resample audio data

if originalRate == targetRate
    return;
end

% Use MATLAB's resample function
try
    audioData = resample(audioData, targetRate, originalRate);
catch ME
    warning('AudioLoader:ResampleError', ...
        'Resampling failed: %s. Using original sample rate.', ME.message);
end
end

function info = extractMetadata(filename, fileInfo, audioData, sampleRate)
% Extract comprehensive metadata

info = struct();

% Basic file information
info.Filename = filename;
info.SampleRate = sampleRate;
info.Duration = size(audioData, 1) / sampleRate;
info.Channels = size(audioData, 2);
info.BitsPerSample = fileInfo.BitsPerSample;

% Audio characteristics
info.MaxAmplitude = max(abs(audioData(:)));
info.RMS = rms(audioData(:));
info.DynamicRange = 20 * log10(info.MaxAmplitude / (info.RMS + eps));

% File metadata (if available)
try
    if isfield(fileInfo, 'Title')
        info.Title = fileInfo.Title;
    end
    if isfield(fileInfo, 'Artist')
        info.Artist = fileInfo.Artist;
    end
    if isfield(fileInfo, 'Comment')
        info.Comment = fileInfo.Comment;
    end
catch
    % Metadata extraction failed, continue without it
end

% Processing timestamp
info.LoadedAt = datetime('now');
end
