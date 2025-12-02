function launch_audio_processor()
%LAUNCH_AUDIO_PROCESSOR Main entry point for the Audio Signal Processor
%
%   This function initializes the MATLAB Audio Signal Processor application,
%   checks for required toolboxes, sets up the path, and launches the GUI.
%
%   Requirements:
%   - MATLAB R2020b or later
%   - Signal Processing Toolbox (required)
%   - Audio Toolbox (recommended)
%   - DSP System Toolbox (optional, for advanced features)
%
%   Example:
%       launch_audio_processor
%
%   See also: MainWindow

arguments
end

% Add current directory and subdirectories to path
currentDir = fileparts(mfilename('fullpath'));
addpath(currentDir);
addpath(fullfile(currentDir, 'core'));
addpath(fullfile(currentDir, 'gui'));
addpath(fullfile(currentDir, 'utils'));
addpath(fullfile(currentDir, 'tests'));
addpath(fullfile(currentDir, 'examples'));

% Check for required toolboxes
fprintf('MATLAB Audio Signal Processor\n');
fprintf('============================\n');

% Check Signal Processing Toolbox (required)
if ~license('test', 'Signal_Toolbox')
    error('AudioProcessor:MissingToolbox', ...
        'Signal Processing Toolbox is required but not available.');
else
    fprintf('✓ Signal Processing Toolbox: Available\n');
end

% Check Audio Toolbox (recommended)
v = ver('audio');
if ~isempty(v)
    fprintf('✓ Audio Toolbox: Available\n');
else
    warning('AudioProcessor:MissingToolbox', ...
        'Audio Toolbox is recommended but not available. Some features may be limited.');
end

% Check DSP System Toolbox (optional)
v = ver('dsp');
if ~isempty(v)
    fprintf('✓ DSP System Toolbox: Available\n');
else
    fprintf('⚠ DSP System Toolbox: Not available (optional)\n');
end

% Check MATLAB version
matlabVersion = version('-release');
fprintf('✓ MATLAB Version: %s\n', matlabVersion);

% Initialize application
try
    fprintf('\nInitializing Audio Signal Processor...\n');

    % Create main window
    mainWindow = MainWindow();

    fprintf('✓ Application launched successfully\n');
    fprintf('\nReady to process audio!\n');

catch ME
    fprintf('✗ Error launching application: %s\n', ME.message);

    % Clean up any open figures
    try
        openFigs = findall(0, 'Type', 'figure');
        delete(openFigs);
    catch
        % Ignore cleanup errors
    end

    % Display error details
    fprintf('\nError details:\n');
    fprintf('  Identifier: %s\n', ME.identifier);
    if ~isempty(ME.stack)
        fprintf('  File: %s\n', ME.stack(1).file);
        fprintf('  Line: %d\n', ME.stack(1).line);
    else
        fprintf('  File: [stack unavailable]\n');
        fprintf('  Line: [stack unavailable]\n');
    end
    fprintf('\nPlease fix the error and try again.\n');

    % Don't rethrow - just return so MATLAB prompt is available
    return;
end
end
