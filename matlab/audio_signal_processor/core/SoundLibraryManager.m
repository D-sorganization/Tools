function libraryManager = SoundLibraryManager(libraryPath)
%SOUNDLIBRARYMANAGER Manage audio sample library with metadata and search
%
%   LIBRARYMANAGER = SOUNDLIBRARYMANAGER(LIBRARYPATH) creates a library manager
%   for the specified library path.
%
%   LIBRARYMANAGER = SOUNDLIBRARYMANAGER() uses the default library path.
%
%   Input Arguments:
%   ---------------
%   LIBRARYPATH - String specifying the library directory path (optional)
%
%   Properties:
%   ----------
%   LibraryPath - Path to the sample library
%   Catalog - Structure containing library metadata
%   Categories - Available sample categories
%   MATLABSounds - MATLAB built-in sound library integration
%
%   Methods:
%   --------
%   loadSample(category, filename) - Load a sample from the library
%   loadMATLABSound(soundName) - Load MATLAB built-in sound
%   searchSamples(query) - Search samples by metadata
%   getCategories() - Get available categories
%   getMATLABSounds() - Get available MATLAB built-in sounds
%   addSample(filepath, metadata) - Add new sample to library
%   updateCatalog() - Refresh library catalog
%
%   Example:
%   --------
%   % Create library manager
%   libMgr = SoundLibraryManager();
%
%   % Load a drum sample
%   [data, fs] = libMgr.loadSample('drums', 'kick_01.wav');
%
%   % Load MATLAB built-in sound
%   [data, fs] = libMgr.loadMATLABSound('handel');
%
%   % Search for bass samples
%   results = libMgr.searchSamples('bass');
%
%   See also: AudioLoader, MetadataExtractor, load

arguments
    libraryPath (1,1) string = ""
end

% Set default library path if not provided
if libraryPath == ""
    currentDir = fileparts(mfilename('fullpath'));
    libraryPath = fullfile(fileparts(currentDir), 'library');
end

% Initialize library manager
libraryManager = struct();
libraryManager.LibraryPath = libraryPath;
libraryManager.Catalog = struct();
libraryManager.Categories = {};
libraryManager.MATLABSounds = struct();

% Convert to class-like structure with methods
libraryManager.loadSample = @(category, filename) loadSample(libraryManager, category, filename);
libraryManager.loadMATLABSound = @(soundName) loadMATLABSound(libraryManager, soundName);
libraryManager.searchSamples = @(query) searchSamples(libraryManager, query);
libraryManager.getCategories = @() getCategories(libraryManager);
libraryManager.getMATLABSounds = @() getMATLABSounds(libraryManager);
libraryManager.addSample = @(filepath, metadata) addSample(libraryManager, filepath, metadata);
libraryManager.updateCatalog = @() updateCatalog(libraryManager);
libraryManager.initializeMATLABSounds = @() initializeMATLABSounds(libraryManager);

% Initialize catalog and MATLAB sounds
libraryManager.updateCatalog();
libraryManager.initializeMATLABSounds();
end

function [audioData, sampleRate, info] = loadSample(libraryManager, category, filename)
% Load a sample from the specified category

arguments
    libraryManager
    category (1,1) string
    filename (1,1) string
end

% Construct full path
samplePath = fullfile(libraryManager.LibraryPath, 'samples', category, filename);

% Check if file exists
if ~isfile(samplePath)
    error('SoundLibraryManager:FileNotFound', ...
        'Sample "%s" not found in category "%s"', filename, category);
end

% Load using AudioLoader
try
    [audioData, sampleRate, info] = AudioLoader(samplePath);
    info.Category = category;
    info.Filename = filename;
catch ME
    error('SoundLibraryManager:LoadError', ...
        'Error loading sample "%s": %s', filename, ME.message);
end
end

function results = searchSamples(libraryManager, query)
% Search samples by metadata

arguments
    libraryManager
    query (1,1) string
end

results = struct();
results.Query = query;
results.Matches = {};

% Simple text search in catalog
catalog = libraryManager.Catalog;
categories = fieldnames(catalog);

for i = 1:length(categories)
    category = categories{i};
    if isfield(catalog, category)
        samples = catalog.(category);
        sampleNames = fieldnames(samples);

        for j = 1:length(sampleNames)
            sampleName = sampleNames{j};
            sampleInfo = samples.(sampleName);

            % Search in filename and metadata
            if contains(lower(sampleName), lower(query)) || ...
                    (isfield(sampleInfo, 'Tags') && contains(lower(sampleInfo.Tags), lower(query)))

                match = struct();
                match.Category = category;
                match.Filename = sampleName;
                match.Path = fullfile(libraryManager.LibraryPath, 'samples', category, sampleName);
                match.Metadata = sampleInfo;

                results.Matches{end+1} = match;
            end
        end
    end
end

results.Count = length(results.Matches);
end

function categories = getCategories(libraryManager)
% Get available sample categories

samplesDir = fullfile(libraryManager.LibraryPath, 'samples');

if ~isfolder(samplesDir)
    categories = {};
    return;
end

% Get subdirectories
dirInfo = dir(samplesDir);
categories = {};

for i = 1:length(dirInfo)
    if dirInfo(i).isdir && ~strcmp(dirInfo(i).name, '.') && ~strcmp(dirInfo(i).name, '..')
        categories{end+1} = dirInfo(i).name;
    end
end
end

function success = addSample(libraryManager, filepath, metadata)
% Add new sample to library

arguments
    libraryManager
    filepath (1,1) string {mustBeFile}
    metadata (1,1) struct = struct()
end

success = false;

try
    % Determine category
    if isfield(metadata, 'Category')
        category = metadata.Category;
    else
        category = 'user_library';
    end

    % Create category directory if it doesn't exist
    categoryDir = fullfile(libraryManager.LibraryPath, 'samples', category);
    if ~isfolder(categoryDir)
        mkdir(categoryDir);
    end

    % Copy file to library
    [~, filename, ext] = fileparts(filepath);
    destPath = fullfile(categoryDir, [filename, ext]);
    copyfile(filepath, destPath);

    % Update catalog
    libraryManager.updateCatalog();

    success = true;

catch ME
    warning('SoundLibraryManager:AddError', ...
        'Error adding sample: %s', ME.message);
end
end

function [audioData, sampleRate, info] = loadMATLABSound(libraryManager, soundName)
% Load MATLAB built-in sound

arguments
    libraryManager
    soundName (1,1) string
end

% Check if sound exists in MATLAB sounds catalog
if ~isfield(libraryManager.MATLABSounds, soundName)
    error('SoundLibraryManager:MATLABSoundNotFound', ...
        'MATLAB sound "%s" not found', soundName);
end

try
    % Load MATLAB built-in sound using eval to call the sound name as a function
    % MATLAB built-in sounds like 'handel', 'gong', etc. are loaded this way
    soundData = load(char(soundName));

    % Get the field name (usually the same as the sound name, or 'y')
    fieldNames = fieldnames(soundData);
    if ismember('y', fieldNames)
        audioData = soundData.y;
        if isfield(soundData, 'Fs')
            sampleRate = soundData.Fs;
        else
            sampleRate = 8192; % Default for some MATLAB sounds
        end
    else
        % Try the sound name as field
        if isfield(soundData, char(soundName))
            audioData = soundData.(char(soundName));
        else
            audioData = soundData.(fieldNames{1});
        end
        sampleRate = 8192; % Default for MATLAB sounds without Fs
    end

    % Normalize to column vector if needed
    if size(audioData, 2) > size(audioData, 1)
        audioData = audioData';
    end

    % Create info structure
    info = struct();
    info.Filename = soundName;
    info.SampleRate = sampleRate;
    info.Duration = size(audioData, 1) / sampleRate;
    info.Channels = size(audioData, 2);
    info.Source = 'MATLAB Built-in';
    info.Category = 'matlab_sounds';

    % Add MATLAB sound metadata
    matlabSoundInfo = libraryManager.MATLABSounds.(soundName);
    info.Description = matlabSoundInfo.Description;
    info.Tags = matlabSoundInfo.Tags;
    info.LoadedAt = datetime('now');

catch ME
    error('SoundLibraryManager:MATLABLoadError', ...
        'Error loading MATLAB sound "%s": %s', soundName, ME.message);
end
end

function matlabSounds = getMATLABSounds(libraryManager)
% Get available MATLAB built-in sounds

matlabSounds = libraryManager.MATLABSounds;
end

function initializeMATLABSounds(libraryManager)
% Initialize MATLAB built-in sound library catalog

% Core MATLAB sounds (always available)
libraryManager.MATLABSounds.handel = struct();
libraryManager.MATLABSounds.handel.Description = 'Handel''s Hallelujah Chorus';
libraryManager.MATLABSounds.handel.Tags = 'classical music chorus baroque';
libraryManager.MATLABSounds.handel.Category = 'classical';
libraryManager.MATLABSounds.handel.Duration = 'Unknown';

libraryManager.MATLABSounds.gong = struct();
libraryManager.MATLABSounds.gong.Description = 'Gong sound';
libraryManager.MATLABSounds.gong.Tags = 'percussion gong bell';
libraryManager.MATLABSounds.gong.Category = 'percussion';
libraryManager.MATLABSounds.gong.Duration = 'Unknown';

libraryManager.MATLABSounds.laughter = struct();
libraryManager.MATLABSounds.laughter.Description = 'Human laughter';
libraryManager.MATLABSounds.laughter.Tags = 'voice human laughter';
libraryManager.MATLABSounds.laughter.Category = 'voice';
libraryManager.MATLABSounds.laughter.Duration = 'Unknown';

libraryManager.MATLABSounds.splat = struct();
libraryManager.MATLABSounds.splat.Description = 'Splat sound effect';
libraryManager.MATLABSounds.splat.Tags = 'effect splat impact';
libraryManager.MATLABSounds.splat.Category = 'effects';
libraryManager.MATLABSounds.splat.Duration = 'Unknown';

libraryManager.MATLABSounds.train = struct();
libraryManager.MATLABSounds.train.Description = 'Train whistle';
libraryManager.MATLABSounds.train.Tags = 'train whistle transportation';
libraryManager.MATLABSounds.train.Category = 'environmental';
libraryManager.MATLABSounds.train.Duration = 'Unknown';

% Additional sounds that might be available depending on MATLAB version/toolboxes
additionalSounds = {
    'chirp', 'Linear chirp signal';
    'sawtooth', 'Sawtooth wave';
    'square', 'Square wave';
    'sinc', 'Sinc function';
    'diric', 'Dirichlet function';
    'pulstran', 'Pulse train';
    'rectpuls', 'Rectangular pulse';
    'tripuls', 'Triangular pulse';
    'gauspuls', 'Gaussian pulse';
    'vco', 'Voltage controlled oscillator';
    'sweep', 'Linear frequency sweep';
    'noise', 'White noise';
    'pinknoise', 'Pink noise';
    'brownnoise', 'Brown noise'
    };

for i = 1:size(additionalSounds, 1)
    soundName = additionalSounds{i, 1};
    description = additionalSounds{i, 2};

    % Check if function exists
    if exist(soundName, 'file') == 2
        libraryManager.MATLABSounds.(soundName) = struct();
        libraryManager.MATLABSounds.(soundName).Description = description;
        libraryManager.MATLABSounds.(soundName).Tags = 'synthetic signal test';
        libraryManager.MATLABSounds.(soundName).Category = 'synthetic';
        libraryManager.MATLABSounds.(soundName).Duration = 'Variable';
    end
end

% Audio Toolbox sounds (if available)
if license('test', 'Audio_Toolbox')
    audioToolboxSounds = {
        'audioexample', 'Audio example file';
        'speech_dft', 'Speech DFT example';
        'speech_dft_8kHz', 'Speech DFT example 8kHz';
        'speech_dft_16kHz', 'Speech DFT example 16kHz';
        'speech_dft_22kHz', 'Speech DFT example 22kHz';
        'speech_dft_44kHz', 'Speech DFT example 44kHz';
        'speech_dft_48kHz', 'Speech DFT example 48kHz';
        'speech_dft_96kHz', 'Speech DFT example 96kHz';
        'speech_dft_192kHz', 'Speech DFT example 192kHz'
        };

    for i = 1:size(audioToolboxSounds, 1)
        soundName = audioToolboxSounds{i, 1};
        description = audioToolboxSounds{i, 2};

        if exist(soundName, 'file') == 2
            libraryManager.MATLABSounds.(soundName) = struct();
            libraryManager.MATLABSounds.(soundName).Description = description;
            libraryManager.MATLABSounds.(soundName).Tags = 'speech audio example';
            libraryManager.MATLABSounds.(soundName).Category = 'speech';
            libraryManager.MATLABSounds.(soundName).Duration = 'Unknown';
        end
    end
end

% DSP System Toolbox sounds (if available)
if license('test', 'DSP_System_Toolbox')
    dspSounds = {
        'dspblks', 'DSP Blockset examples';
        'dspblksfir', 'DSP Blockset FIR examples';
        'dspblksiir', 'DSP Blockset IIR examples';
        'dspblksadapt', 'DSP Blockset adaptive examples';
        'dspblksmultirate', 'DSP Blockset multirate examples';
        'dspblkswavegen', 'DSP Blockset waveform generator examples'
        };

    for i = 1:size(dspSounds, 1)
        soundName = dspSounds{i, 1};
        description = dspSounds{i, 2};

        if exist(soundName, 'file') == 2
            libraryManager.MATLABSounds.(soundName) = struct();
            libraryManager.MATLABSounds.(soundName).Description = description;
            libraryManager.MATLABSounds.(soundName).Tags = 'dsp example signal processing';
            libraryManager.MATLABSounds.(soundName).Category = 'dsp';
            libraryManager.MATLABSounds.(soundName).Duration = 'Variable';
        end
    end
end
end

function updateCatalog(libraryManager)
% Refresh library catalog by scanning directories

samplesDir = fullfile(libraryManager.LibraryPath, 'samples');
catalog = struct();

if ~isfolder(samplesDir)
    libraryManager.Catalog = catalog;
    return;
end

% Scan each category directory
categories = libraryManager.getCategories();

for i = 1:length(categories)
    category = categories{i};
    categoryDir = fullfile(samplesDir, category);

    % Get all audio files in category
    audioFiles = dir(fullfile(categoryDir, '*.wav'));
    audioFiles = [audioFiles; dir(fullfile(categoryDir, '*.mp3'))];
    audioFiles = [audioFiles; dir(fullfile(categoryDir, '*.flac'))];

    categoryCatalog = struct();

    for j = 1:length(audioFiles)
        filename = audioFiles(j).name;
        filepath = fullfile(categoryDir, filename);

        try
            % Extract metadata
            [~, ~, info] = AudioLoader(filepath, 'Metadata', true);

            % Create sample entry
            sampleName = strrep(filename, '.', '_');
            categoryCatalog.(sampleName) = info;

        catch ME
            warning('SoundLibraryManager:CatalogError', ...
                'Error cataloging %s: %s', filename, ME.message);
        end
    end

    if ~isempty(fieldnames(categoryCatalog))
        catalog.(category) = categoryCatalog;
    end
end

libraryManager.Catalog = catalog;
end
