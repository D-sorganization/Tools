function effectsLibrary = InstrumentEffectsLibrary(libraryPath)
%INSTRUMENTEFFECTSLIBRARY Manage instrument-specific effect presets and chains
%
%   EFFECTSLIBRARY = INSTRUMENTEFFECTSLIBRARY(LIBRARYPATH) creates an effects
%   library manager for the specified library path.
%
%   EFFECTSLIBRARY = INSTRUMENTEFFECTSLIBRARY() uses the default library path.
%
%   Input Arguments:
%   ---------------
%   LIBRARYPATH - String specifying the library directory path (optional)
%
%   Properties:
%   ----------
%   LibraryPath - Path to the effects library
%   Presets - Structure containing effect presets by instrument
%   Categories - Available instrument categories
%
%   Methods:
%   --------
%   getPreset(category, presetName) - Get effect preset
%   getPresets(category) - Get all presets for category
%   getCategories() - Get available categories
%   addPreset(category, presetName, effects) - Add new preset
%   loadPresets() - Load presets from JSON files
%   savePresets() - Save presets to JSON files
%
%   Example:
%   --------
%   % Create effects library
%   effectsLib = InstrumentEffectsLibrary();
%
%   % Get guitar preset
%   guitarPreset = effectsLib.getPreset('guitar', 'rock_overdrive');
%
%   % Get all vocal presets
%   vocalPresets = effectsLib.getPresets('vocal');
%
%   See also: AudioEffects, MixerCore

arguments
    libraryPath (1,1) string = ""
end

% Set default library path if not provided
if libraryPath == ""
    currentDir = fileparts(mfilename('fullpath'));
    libraryPath = fullfile(fileparts(currentDir), 'library');
end

% Initialize effects library
effectsLibrary = struct();
effectsLibrary.LibraryPath = libraryPath;
effectsLibrary.Presets = struct();
effectsLibrary.Categories = {};

% Convert to class-like structure with methods
effectsLibrary.getPreset = @(category, presetName) getPreset(effectsLibrary, category, presetName);
effectsLibrary.getPresets = @(category) getPresets(effectsLibrary, category);
effectsLibrary.getCategories = @() getCategories(effectsLibrary);
effectsLibrary.addPreset = @(category, presetName, effects) addPreset(effectsLibrary, category, presetName, effects);
effectsLibrary.loadPresets = @() loadPresets(effectsLibrary);
effectsLibrary.savePresets = @() savePresets(effectsLibrary);

% Initialize with default presets
effectsLibrary.loadPresets();
end

function preset = getPreset(effectsLibrary, category, presetName)
% Get specific effect preset

arguments
    effectsLibrary
    category (1,1) string
    presetName (1,1) string
end

if isfield(effectsLibrary.Presets, category) && ...
        isfield(effectsLibrary.Presets.(category), presetName)
    preset = effectsLibrary.Presets.(category).(presetName);
else
    error('InstrumentEffectsLibrary:PresetNotFound', ...
        'Preset "%s" not found in category "%s"', presetName, category);
end
end

function presets = getPresets(effectsLibrary, category)
% Get all presets for a category

arguments
    effectsLibrary
    category (1,1) string
end

if isfield(effectsLibrary.Presets, category)
    presets = effectsLibrary.Presets.(category);
else
    presets = struct();
end
end

function categories = getCategories(effectsLibrary)
% Get available instrument categories

categories = fieldnames(effectsLibrary.Presets);
end

function success = addPreset(effectsLibrary, category, presetName, effects)
% Add new effect preset

arguments
    effectsLibrary
    category (1,1) string
    presetName (1,1) string
    effects (1,1) struct
end

success = false;

try
    % Create category if it doesn't exist
    if ~isfield(effectsLibrary.Presets, category)
        effectsLibrary.Presets.(category) = struct();
    end

    % Add preset
    effectsLibrary.Presets.(category).(presetName) = effects;
    success = true;

catch ME
    warning('InstrumentEffectsLibrary:AddError', ...
        'Error adding preset: %s', ME.message);
end
end

function loadPresets(effectsLibrary)
% Load presets from JSON files and create default presets

% Initialize with default presets
effectsLibrary.Presets = createDefaultPresets();

% Try to load from JSON files
instrumentEffectsDir = fullfile(effectsLibrary.LibraryPath, 'instrument_effects');

if isfolder(instrumentEffectsDir)
    categories = dir(instrumentEffectsDir);

    for i = 1:length(categories)
        if categories(i).isdir && ~strcmp(categories(i).name, '.') && ~strcmp(categories(i).name, '..')
            category = categories(i).name;
            categoryDir = fullfile(instrumentEffectsDir, category);

            % Load JSON files in category directory
            jsonFiles = dir(fullfile(categoryDir, '*.json'));

            for j = 1:length(jsonFiles)
                jsonFile = fullfile(categoryDir, jsonFiles(j).name);
                try
                    jsonData = jsondecode(fileread(jsonFile));
                    [~, presetName, ~] = fileparts(jsonFiles(j).name);

                    % Create category if it doesn't exist
                    if ~isfield(effectsLibrary.Presets, category)
                        effectsLibrary.Presets.(category) = struct();
                    end

                    effectsLibrary.Presets.(category).(presetName) = jsonData;
                catch ME
                    warning('InstrumentEffectsLibrary:LoadError', ...
                        'Error loading preset %s: %s', jsonFile, ME.message);
                end
            end
        end
    end
end
end

function savePresets(effectsLibrary)
% Save presets to JSON files

instrumentEffectsDir = fullfile(effectsLibrary.LibraryPath, 'instrument_effects');

if ~isfolder(instrumentEffectsDir)
    mkdir(instrumentEffectsDir);
end

categories = fieldnames(effectsLibrary.Presets);

for i = 1:length(categories)
    category = categories{i};
    categoryDir = fullfile(instrumentEffectsDir, category);

    if ~isfolder(categoryDir)
        mkdir(categoryDir);
    end

    presets = effectsLibrary.Presets.(category);
    presetNames = fieldnames(presets);

    for j = 1:length(presetNames)
        presetName = presetNames{j};
        presetData = presets.(presetName);

        jsonFile = fullfile(categoryDir, [presetName, '.json']);
        try
            jsonString = jsonencode(presetData, 'PrettyPrint', true);
            fid = fopen(jsonFile, 'w');
            fprintf(fid, '%s', jsonString);
            fclose(fid);
        catch ME
            warning('InstrumentEffectsLibrary:SaveError', ...
                'Error saving preset %s: %s', jsonFile, ME.message);
        end
    end
end
end

function defaultPresets = createDefaultPresets()
% Create default effect presets for different instruments

defaultPresets = struct();

% Guitar presets
defaultPresets.guitar = struct();

% Rock Overdrive preset
defaultPresets.guitar.rock_overdrive = struct();
defaultPresets.guitar.rock_overdrive.Effects = {
    struct('Type', 'Distortion', 'Parameters', struct('Drive', 0.7, 'Tone', 0.6, 'Level', 0.8));
    struct('Type', 'EQ', 'Parameters', struct('LowGain', 2, 'MidGain', -1, 'HighGain', 3, 'LowFreq', 200, 'HighFreq', 3000));
    struct('Type', 'Reverb', 'Parameters', struct('RoomSize', 0.3, 'DecayTime', 1.5, 'Damping', 0.7));
    };
defaultPresets.guitar.rock_overdrive.Description = 'Classic rock overdrive with EQ and reverb';

% Clean Guitar preset
defaultPresets.guitar.clean = struct();
defaultPresets.guitar.clean.Effects = {
    struct('Type', 'EQ', 'Parameters', struct('LowGain', 1, 'MidGain', 0, 'HighGain', 2, 'LowFreq', 250, 'HighFreq', 4000));
    struct('Type', 'Compression', 'Parameters', struct('Threshold', -8, 'Ratio', 3, 'Attack', 5, 'Release', 50));
    struct('Type', 'Reverb', 'Parameters', struct('RoomSize', 0.4, 'DecayTime', 2.0, 'Damping', 0.6));
    };
defaultPresets.guitar.clean.Description = 'Clean guitar with subtle compression and reverb';

% Vocal presets
defaultPresets.vocal = struct();

% Pop Vocal preset
defaultPresets.vocal.pop = struct();
defaultPresets.vocal.pop.Effects = {
    struct('Type', 'Compression', 'Parameters', struct('Threshold', -6, 'Ratio', 4, 'Attack', 3, 'Release', 100));
    struct('Type', 'EQ', 'Parameters', struct('LowGain', -2, 'MidGain', 1, 'HighGain', 2, 'LowFreq', 200, 'HighFreq', 5000));
    struct('Type', 'Reverb', 'Parameters', struct('RoomSize', 0.5, 'DecayTime', 2.5, 'Damping', 0.5));
    };
defaultPresets.vocal.pop.Description = 'Pop vocal with compression, EQ, and reverb';

% Radio Vocal preset
defaultPresets.vocal.radio = struct();
defaultPresets.vocal.radio.Effects = {
    struct('Type', 'Compression', 'Parameters', struct('Threshold', -4, 'Ratio', 6, 'Attack', 2, 'Release', 80));
    struct('Type', 'EQ', 'Parameters', struct('LowGain', -3, 'MidGain', 2, 'HighGain', 3, 'LowFreq', 300, 'HighFreq', 6000));
    struct('Type', 'Limiting', 'Parameters', struct('Limit', -0.3));
    };
defaultPresets.vocal.radio.Description = 'Radio-ready vocal with heavy compression and limiting';

% Synth presets
defaultPresets.synth = struct();

% Pad Synth preset
defaultPresets.synth.pad = struct();
defaultPresets.synth.pad.Effects = {
    struct('Type', 'Chorus', 'Parameters', struct('Rate', 0.3, 'Depth', 0.4, 'Feedback', 0.2));
    struct('Type', 'Delay', 'Parameters', struct('DelayTime', 0.5, 'Feedback', 0.3, 'TempoSync', true, 'Tempo', 120));
    struct('Type', 'Reverb', 'Parameters', struct('RoomSize', 0.8, 'DecayTime', 4.0, 'Damping', 0.3));
    };
defaultPresets.synth.pad.Description = 'Atmospheric pad with chorus, delay, and reverb';

% Lead Synth preset
defaultPresets.synth.lead = struct();
defaultPresets.synth.lead.Effects = {
    struct('Type', 'Distortion', 'Parameters', struct('Drive', 0.4, 'Tone', 0.7, 'Level', 0.9));
    struct('Type', 'Chorus', 'Parameters', struct('Rate', 0.8, 'Depth', 0.2, 'Feedback', 0.1));
    struct('Type', 'Delay', 'Parameters', struct('DelayTime', 0.25, 'Feedback', 0.2));
    };
defaultPresets.synth.lead.Description = 'Lead synth with distortion, chorus, and delay';

% Master presets
defaultPresets.master = struct();

% Mastering preset
defaultPresets.master.mastering = struct();
defaultPresets.master.mastering.Effects = {
    struct('Type', 'EQ', 'Parameters', struct('LowGain', 1, 'MidGain', 0, 'HighGain', 1, 'LowFreq', 100, 'HighFreq', 8000));
    struct('Type', 'Compression', 'Parameters', struct('Threshold', -3, 'Ratio', 2, 'Attack', 10, 'Release', 200));
    struct('Type', 'Limiting', 'Parameters', struct('Limit', -0.1));
    };
defaultPresets.master.mastering.Description = 'Mastering chain with EQ, compression, and limiting';

% Live preset
defaultPresets.master.live = struct();
defaultPresets.master.live.Effects = {
    struct('Type', 'EQ', 'Parameters', struct('LowGain', 0, 'MidGain', 0, 'HighGain', 0, 'LowFreq', 200, 'HighFreq', 4000));
    struct('Type', 'Limiting', 'Parameters', struct('Limit', -0.5));
    };
defaultPresets.master.live.Description = 'Live performance mastering with EQ and limiting';
end
