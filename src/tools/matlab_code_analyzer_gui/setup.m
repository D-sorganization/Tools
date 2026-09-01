function setup()
%SETUP Add Code Analysis GUI to MATLAB path and verify installation.
%
%   SETUP() adds the Code Analysis GUI folder to the MATLAB path and
%   runs a quick verification to ensure all components are working.
%
%   This function should be run once after downloading or installing
%   the Code Analysis GUI package.
%
%   Example:
%       setup();

fprintf('Setting up MATLAB Code Analysis GUI...\n');

% Get the directory containing this setup script
thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);

% Add to path
addpath(thisDir);
fprintf('Added to MATLAB path: %s\n', thisDir);
utilitiesQualityDir = fullfile(thisDir, '..', 'matlab_utilities', 'quality');
if exist(utilitiesQualityDir, 'dir') == 7
    addpath(utilitiesQualityDir);
    fprintf('Added to MATLAB path: %s\n', utilitiesQualityDir);
end

% Verify components
fprintf('\nVerifying installation...\n');

% Check core functions exist
requiredFiles = {'exportCodeIssues.m', 'codeIssuesGUI.m', 'launchCodeAnalyzer.m'};
allFound = true;

for i = 1:length(requiredFiles)
    if exist(fullfile(thisDir, requiredFiles{i}), 'file') == 2
        fprintf('✓ Found: %s\n', requiredFiles{i});
    else
        fprintf('✗ Missing: %s\n', requiredFiles{i});
        allFound = false;
    end
end

% Test basic functionality
if allFound
    try
        % Test that we can call the main functions without error
        help('exportCodeIssues');
        help('codeIssuesGUI');
        help('launchCodeAnalyzer');
        fprintf('\n✓ All functions accessible\n');

        % Test that we can create an empty results table
        % Use current directory (folder) instead of non-existent file
        testResults = exportCodeIssues(pwd, 'Quiet', true);
        if istable(testResults)
            fprintf('✓ Core functionality verified\n');
        else
            fprintf('✗ Core functionality test failed\n');
            allFound = false;
        end

    catch ME
        fprintf('✗ Function verification failed: %s\n', ME.message);
        allFound = false;
    end
end

% Final status
if allFound
    fprintf('\n🎉 Setup complete! Code Analysis GUI is ready to use.\n');
    fprintf('\nTo launch the GUI, run:\n');
    fprintf('   launchCodeAnalyzer()\n');
    fprintf('\nTo see help for any function, use:\n');
    fprintf('   help codeIssuesGUI\n');
    fprintf('   help exportCodeIssues\n');

    % Save path for future sessions (optional)
    try
        savepath();
        fprintf('\n✓ Path saved for future MATLAB sessions\n');
    catch
        fprintf('\n⚠ Could not save path - you may need to run setup() again in future sessions\n');
    end
else
    fprintf('\n❌ Setup incomplete - please check for missing files\n');
end

end
