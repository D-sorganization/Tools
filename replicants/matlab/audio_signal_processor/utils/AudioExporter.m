function success = AudioExporter(audioData, filename, varargin)
%AUDIOEXPORTER Export audio data to various formats with quality settings
%
%   SUCCESS = AUDIOEXPORTER(AUDIODATA, FILENAME) exports audio data to the
%   specified filename with default settings.
%
%   SUCCESS = AUDIOEXPORTER(AUDIODATA, FILENAME, 'Property', Value, ...)
%   specifies additional export parameters using property-value pairs.
%
%   Input Arguments:
%   ---------------
%   AUDIODATA - Audio data matrix (samples x channels)
%   FILENAME - Output filename with extension
%
%   Optional Properties:
%   ------------------
%   'SampleRate'      - Sample rate in Hz (default: 44100)
%   'BitsPerSample'   - Bits per sample: 8, 16, 24, 32 (default: 16)
%   'Quality'         - Quality setting 0-100 (default: 80)
%   'Normalize'       - Normalize audio before export (default: true)
%   'Format'          - Export format: 'auto', 'wav', 'mp3', 'flac', 'ogg' (default: 'auto')
%   'Metadata'        - Metadata structure to embed (default: struct())
%   'Dither'          - Apply dithering (default: false)
%   'DitherType'      - Dither type: 'none', 'rectangular', 'triangular' (default: 'none')
%
%   Output Arguments:
%   ----------------
%   SUCCESS - Logical indicating export success
%
%   Supported Formats:
%   -----------------
%   - WAV (.wav) - Uncompressed, high quality
%   - MP3 (.mp3) - Compressed, requires Audio Toolbox
%   - FLAC (.flac) - Lossless compression, requires Audio Toolbox
%   - OGG (.ogg) - Open source compressed, requires Audio Toolbox
%
%   Example:
%   --------
%   % Load audio data
%   [data, fs] = audioread('song.wav');
%
%   % Export as high-quality WAV
%   success = AudioExporter(data, 'output.wav', 'SampleRate', fs, 'BitsPerSample', 24);
%
%   % Export as MP3 with metadata
%   metadata = struct('Title', 'My Song', 'Artist', 'Me', 'Comment', 'Exported from MATLAB');
%   success = AudioExporter(data, 'output.mp3', 'SampleRate', fs, 'Quality', 90, 'Metadata', metadata);
%
%   % Export as FLAC (lossless)
%   success = AudioExporter(data, 'output.flac', 'SampleRate', fs, 'Format', 'flac');
%
%   See also: audiowrite, AudioLoader

arguments
    audioData (:,:) double
    filename (1,1) string
    options.SampleRate (1,1) double {mustBePositive} = 44100
    options.BitsPerSample (1,1) double {mustBeMember(options.BitsPerSample, [8, 16, 24, 32])} = 16
    options.Quality (1,1) double {mustBeInRange(options.Quality, 0, 100)} = 80
    options.Normalize (1,1) logical = true
    options.Format (1,1) string {mustBeMember(options.Format, ["auto", "wav", "mp3", "flac", "ogg"])} = "auto"
    options.Metadata (1,1) struct = struct()
    options.Dither (1,1) logical = false
    options.DitherType (1,1) string {mustBeMember(options.DitherType, ["none", "rectangular", "triangular"])} = "none"
end

success = false;

% Validate input
if isempty(audioData)
    error('AudioExporter:EmptyInput', 'Input audio data is empty');
end

% Determine format from filename if auto
if strcmp(options.Format, 'auto')
    [~, ~, ext] = fileparts(filename);
    format = lower(ext(2:end)); % Remove the dot
else
    format = options.Format;
end

% Validate format
if ~ismember(format, ["wav", "mp3", "flac", "ogg"])
    error('AudioExporter:UnsupportedFormat', 'Unsupported format: %s', format);
end

% Check for required toolboxes for compressed formats
if ismember(format, ["mp3", "flac", "ogg"]) && ~license('test', 'Audio_Toolbox')
    error('AudioExporter:MissingToolbox', ...
        'Audio Toolbox required for %s format', format);
end

try
    % Prepare audio data
    exportData = prepareAudioData(audioData, options);

    % Create export parameters
    exportParams = createExportParameters(format, options);

    % Export audio
    audiowrite(filename, exportData, options.SampleRate, exportParams);

    success = true;

catch ME
    error('AudioExporter:ExportError', 'Error exporting audio: %s', ME.message);
end
end

function exportData = prepareAudioData(audioData, options)
% Prepare audio data for export

exportData = audioData;

% Normalize if requested
if options.Normalize
    maxVal = max(abs(exportData(:)));
    if maxVal > 0
        exportData = exportData / maxVal;
    end
end

% Apply dithering if requested
if options.Dither && options.BitsPerSample < 32
    exportData = applyDithering(exportData, options.DitherType, options.BitsPerSample);
end

% Ensure data is in correct range for target bit depth
exportData = clipToBitDepth(exportData, options.BitsPerSample);
end

function exportParams = createExportParameters(format, options)
% Create export parameters based on format

exportParams = struct();

switch format
    case 'wav'
        exportParams.BitsPerSample = options.BitsPerSample;

    case 'mp3'
        exportParams.Quality = options.Quality;

    case 'flac'
        exportParams.CompressionLevel = round(options.Quality / 10); % 0-9

    case 'ogg'
        exportParams.Quality = options.Quality;
end

% Add metadata if provided
if ~isempty(fieldnames(options.Metadata))
    exportParams.Comment = options.Metadata;
end
end

function ditheredData = applyDithering(audioData, ditherType, bitsPerSample)
% Apply dithering to audio data

switch ditherType
    case 'rectangular'
        % Rectangular dither
        noiseLevel = 1 / (2^bitsPerSample);
        ditherNoise = (rand(size(audioData)) - 0.5) * noiseLevel;

    case 'triangular'
        % Triangular dither
        noiseLevel = 1 / (2^bitsPerSample);
        ditherNoise = (rand(size(audioData)) + rand(size(audioData)) - 1) * noiseLevel;

    otherwise
        ditherNoise = 0;
end

ditheredData = audioData + ditherNoise;
end

function clippedData = clipToBitDepth(audioData, bitsPerSample)
% Clip audio data to appropriate range for bit depth

switch bitsPerSample
    case 8
        % 8-bit: -1 to 1 maps to -128 to 127
        clippedData = max(-1, min(1, audioData));

    case 16
        % 16-bit: -1 to 1 maps to -32768 to 32767
        clippedData = max(-1, min(1, audioData));

    case 24
        % 24-bit: -1 to 1 maps to -8388608 to 8388607
        clippedData = max(-1, min(1, audioData));

    case 32
        % 32-bit: -1 to 1 maps to -2147483648 to 2147483647
        clippedData = max(-1, min(1, audioData));

    otherwise
        clippedData = audioData;
end
end
