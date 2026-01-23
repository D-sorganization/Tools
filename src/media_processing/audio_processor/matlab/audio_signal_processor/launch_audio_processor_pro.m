function mainWindow = launch_audio_processor_pro()
%LAUNCH_AUDIO_PROCESSOR_PRO Launch Audio Signal Processor - Professional Edition
%
%   MAINWINDOW = LAUNCH_AUDIO_PROCESSOR_PRO() launches the complete audio
%   processing suite with all 9 tabs and 100% backend feature exposure.
%
%   Features:
%   ---------
%   - Professional audio editing (trim, cut, fade, 50-level undo)
%   - 11 audio effects including convolution reverb
%   - Enhanced multi-track mixer with time offsets and fades
%   - Music production tools (autotune, key/tempo detection)
%   - Research-grade analysis (wavelets, features, anti-aliasing)
%   - Comprehensive sample library
%   - User preferences and settings
%
%   Tabs:
%   -----
%   1. Waveform - View and navigate audio
%   2. Edit - Audio editing with undo/redo
%   3. Effects - Effect chain management
%   4. Mixer - Multi-track mixing with timeline
%   5. Production - Music production tools
%   6. Analysis - Real-time audio analysis
%   7. Research - Advanced analysis tools
%   8. Library - Sample browser and management
%   9. Settings - Application preferences
%
%   Example:
%   --------
%   mainWindow = launch_audio_processor_pro();
%
%   See also: MainWindow, COMPLETE_IMPLEMENTATION_GUIDE

fprintf('\n');
fprintf('========================================\n');
fprintf('  Audio Signal Processor - Pro Edition  \n');
fprintf('========================================\n');
fprintf('\n');

% Check MATLAB version
if verLessThan('matlab', '9.9')  % R2020b
    warning('This application requires MATLAB R2020b or later for best results.');
end

% Add paths
fprintf('Adding paths...\n');
currentDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(currentDir, 'core')));
addpath(genpath(fullfile(currentDir, 'gui')));
addpath(genpath(fullfile(currentDir, 'utils')));

% Load all callback functions
fprintf('Loading GUI components...\n');

try
    % Check if callback files exist
    callbackFiles = {
        fullfile(currentDir, 'gui', 'MainWindowCallbacks.m');
        fullfile(currentDir, 'gui', 'MainWindowCallbacks_Part2.m');
        fullfile(currentDir, 'gui', 'MainWindowCallbacks_Filters.m')
    };

    for i = 1:length(callbackFiles)
        if exist(callbackFiles{i}, 'file')
            run(callbackFiles{i});
        else
            warning('Callback file not found: %s', callbackFiles{i});
        end
    end

    % Create main window
    fprintf('Creating main window...\n');
    mainWindow = MainWindow();

    % Success message
    fprintf('\n');
    fprintf('✓ Launch successful!\n');
    fprintf('\n');
    fprintf('Quick Start:\n');
    fprintf('  1. Load audio: File → Load Audio (Ctrl+O)\n');
    fprintf('  2. Edit: Use Edit tab for trim, fade, normalize\n');
    fprintf('  3. Effects: Add effects in Effects tab\n');
    fprintf('  4. Mix: Load tracks in Mixer tab with time offsets\n');
    fprintf('  5. Production: Use autotune and music tools\n');
    fprintf('\n');
    fprintf('Documentation:\n');
    fprintf('  - COMPLETE_IMPLEMENTATION_GUIDE.md - Full guide\n');
    fprintf('  - README_COMPREHENSIVE.md - Feature reference\n');
    fprintf('  - CONVOLUTION_REVERB_GUIDE.md - Reverb guide\n');
    fprintf('\n');
    fprintf('Enjoy your professional audio processing suite!\n');
    fprintf('========================================\n');
    fprintf('\n');

catch ME
    fprintf('\n');
    fprintf('✗ Error during launch: %s\n', ME.message);
    fprintf('\n');
    fprintf('Troubleshooting:\n');
    fprintf('  1. Ensure you are in the audio_signal_processor directory\n');
    fprintf('  2. Check that all core/ and gui/ files are present\n');
    fprintf('  3. Review COMPLETE_IMPLEMENTATION_GUIDE.md for setup\n');
    fprintf('\n');
    rethrow(ME);
end
end
