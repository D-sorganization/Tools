function metadata = MetadataExtractor(filename, varargin)
%METADATAEXTRACTOR Extract comprehensive metadata from audio files
%
%   METADATA = METADATAEXTRACTOR(FILENAME) extracts metadata from the specified
%   audio file.
%
%   METADATA = METADATAEXTRACTOR(FILENAME, 'Property', Value, ...)
%   specifies additional extraction parameters using property-value pairs.
%
%   Input Arguments:
%   ---------------
%   FILENAME - Audio file path
%
%   Optional Properties:
%   ------------------
%   'IncludeAudioInfo' - Include audio file information (default: true)
%   'IncludeFileInfo'  - Include file system information (default: true)
%   'IncludeAnalysis'  - Include audio analysis (default: false)
%   'AnalysisLength'   - Length of audio to analyze in seconds (default: 10)
%   'IncludeTags'      - Include ID3/audio tags (default: true)
%
%   Output Arguments:
%   ----------------
%   METADATA - Structure containing extracted metadata
%
%   Metadata Fields:
%   ---------------
%   Basic Information:
%   - Filename - File name
%   - Filepath - Full file path
%   - FileSize - File size in bytes
%   - Created - Creation date
%   - Modified - Modification date
%
%   Audio Information:
%   - SampleRate - Sample rate in Hz
%   - Duration - Duration in seconds
%   - Channels - Number of channels
%   - BitsPerSample - Bits per sample
%   - Format - Audio format
%   - Compression - Compression type
%
%   Audio Analysis (if enabled):
%   - RMS - RMS level
%   - Peak - Peak level
%   - DynamicRange - Dynamic range in dB
%   - SpectralCentroid - Spectral centroid
%   - SpectralRolloff - Spectral rolloff frequency
%   - ZeroCrossingRate - Zero crossing rate
%
%   Tags (if available):
%   - Title - Song title
%   - Artist - Artist name
%   - Album - Album name
%   - Year - Release year
%   - Genre - Genre
%   - Comment - Comments
%
%   Example:
%   --------
%   % Extract basic metadata
%   metadata = MetadataExtractor('song.wav');
%
%   % Extract with audio analysis
%   metadata = MetadataExtractor('song.wav', 'IncludeAnalysis', true);
%
%   % Extract limited metadata
%   metadata = MetadataExtractor('song.wav', 'IncludeTags', false, 'IncludeAnalysis', false);
%
%   See also: audioinfo, AudioLoader

arguments
    filename (1,1) string {mustBeFile}
    options.IncludeAudioInfo (1,1) logical = true
    options.IncludeFileInfo (1,1) logical = true
    options.IncludeAnalysis (1,1) logical = false
    options.AnalysisLength (1,1) double {mustBePositive} = 10
    options.IncludeTags (1,1) logical = true
end

% Initialize metadata structure
metadata = struct();

try
    % Extract basic file information
    if options.IncludeFileInfo
        metadata = extractFileInfo(filename, metadata);
    end

    % Extract audio information
    if options.IncludeAudioInfo
        metadata = extractAudioInfo(filename, metadata);
    end

    % Extract audio tags
    if options.IncludeTags
        metadata = extractAudioTags(filename, metadata);
    end

    % Perform audio analysis
    if options.IncludeAnalysis
        metadata = extractAudioAnalysis(filename, metadata, options.AnalysisLength);
    end

    % Add extraction timestamp
    metadata.ExtractedAt = datetime('now');

catch ME
    error('MetadataExtractor:ExtractionError', ...
        'Error extracting metadata: %s', ME.message);
end
end

function metadata = extractFileInfo(filename, metadata)
% Extract file system information

fileInfo = dir(filename);

metadata.Filename = fileInfo.name;
metadata.Filepath = filename;
metadata.FileSize = fileInfo.bytes;
metadata.Created = fileInfo.date;
metadata.Modified = fileInfo.date;

% Get file extension
[~, ~, ext] = fileparts(filename);
metadata.FileExtension = lower(ext);
end

function metadata = extractAudioInfo(filename, metadata)
% Extract audio file information

try
    audioInfo = audioinfo(filename);

    metadata.SampleRate = audioInfo.SampleRate;
    metadata.Duration = audioInfo.Duration;
    metadata.Channels = audioInfo.NumChannels;
    metadata.BitsPerSample = audioInfo.BitsPerSample;
    metadata.TotalSamples = audioInfo.TotalSamples;

    % Additional audio info if available
    if isfield(audioInfo, 'Format')
        metadata.Format = audioInfo.Format;
    end

    if isfield(audioInfo, 'CompressionMethod')
        metadata.Compression = audioInfo.CompressionMethod;
    end

catch ME
    warning('MetadataExtractor:AudioInfoError', ...
        'Error extracting audio info: %s', ME.message);
end
end

function metadata = extractAudioTags(filename, metadata)
% Extract audio tags (ID3, etc.)

try
    % Try to extract tags using audioinfo
    audioInfo = audioinfo(filename);

    % Common tag fields
    tagFields = {'Title', 'Artist', 'Album', 'Year', 'Genre', 'Comment', 'Track'};

    for i = 1:length(tagFields)
        field = tagFields{i};
        if isfield(audioInfo, field) && ~isempty(audioInfo.(field))
            metadata.(field) = audioInfo.(field);
        end
    end

catch ME
    warning('MetadataExtractor:TagError', ...
        'Error extracting tags: %s', ME.message);
end
end

function metadata = extractAudioAnalysis(filename, metadata, analysisLength)
% Extract audio analysis information

try
    % Load audio data for analysis
    [audioData, sampleRate] = audioread(filename);

    % Limit analysis length
    maxSamples = round(analysisLength * sampleRate);
    if size(audioData, 1) > maxSamples
        audioData = audioData(1:maxSamples, :);
    end

    % Convert to mono for analysis
    if size(audioData, 2) > 1
        audioDataMono = mean(audioData, 2);
    else
        audioDataMono = audioData;
    end

    % Calculate basic audio properties
    metadata.RMS = rms(audioDataMono);
    metadata.Peak = max(abs(audioDataMono));
    metadata.DynamicRange = 20 * log10(metadata.Peak / (metadata.RMS + eps));

    % Calculate spectral properties
    metadata = calculateSpectralProperties(audioDataMono, sampleRate, metadata);

    % Calculate zero crossing rate
    metadata.ZeroCrossingRate = calculateZeroCrossingRate(audioDataMono);

catch ME
    warning('MetadataExtractor:AnalysisError', ...
        'Error performing audio analysis: %s', ME.message);
end
end

function metadata = calculateSpectralProperties(audioData, sampleRate, metadata)
% Calculate spectral properties

try
    % Calculate FFT
    nfft = min(4096, length(audioData));
    fftData = fft(audioData, nfft);
    magnitude = abs(fftData(1:nfft/2+1));
    frequencies = (0:nfft/2) * sampleRate / nfft;

    % Spectral centroid
    metadata.SpectralCentroid = sum(frequencies .* magnitude') / sum(magnitude);

    % Spectral rolloff (95% of energy)
    cumulativeEnergy = cumsum(magnitude);
    totalEnergy = cumulativeEnergy(end);
    rolloffIndex = find(cumulativeEnergy >= 0.95 * totalEnergy, 1);
    if ~isempty(rolloffIndex)
        metadata.SpectralRolloff = frequencies(rolloffIndex);
    else
        metadata.SpectralRolloff = frequencies(end);
    end

    % Spectral bandwidth
    centroid = metadata.SpectralCentroid;
    bandwidth = sqrt(sum(((frequencies - centroid).^2) .* magnitude') / sum(magnitude));
    metadata.SpectralBandwidth = bandwidth;

catch ME
    warning('MetadataExtractor:SpectralError', ...
        'Error calculating spectral properties: %s', ME.message);
end
end

function zcr = calculateZeroCrossingRate(audioData)
% Calculate zero crossing rate

% Find zero crossings
signChanges = diff(sign(audioData)) ~= 0;
zcr = sum(signChanges) / length(audioData);
end
