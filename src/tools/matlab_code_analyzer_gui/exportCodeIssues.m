function T = exportCodeIssues(targetPath, varargin)
%EXPORTCODEISSUES Forwarding shim to canonical exportCodeIssues in matlab_utilities.
%
%   T = EXPORTCODEISSUES(targetPath) runs Code Analyzer on a single file or on
%   all .m files under a folder (recursively) and returns a table of issues by
%   delegating to the canonical implementation in tools/matlab_utilities/quality.
%
%   T = EXPORTCODEISSUES(targetPath, 'Name', Value, ...) accepts options:
%       'Output'        - Output file path (.csv, .xlsx, .json, .md)
%       'Recursive'     - true|false (default true)
%       'IncludeExt'    - Cellstr of file extensions. Default {'.m'}
%       'ExcludeDirs'   - Cellstr of directory names to skip
%       'ExcludeFiles'  - Cellstr of wildcard file patterns to skip
%       'Root'          - Root folder for computing relative paths
%       'OnError'       - 'record' (default) | 'rethrow'
%       'Quiet'         - true|false (default true)
%
%   See also: run_quality_checks, codeIssuesGUI, CHECKCODE

% Determine directory of this shim and locate canonical utility folder
thisDir = fileparts(mfilename('fullpath'));
canonicalDir = fullfile(thisDir, '..', 'matlab_utilities', 'quality');
canonicalFile = fullfile(canonicalDir, 'exportCodeIssues.m');

if exist(canonicalFile, 'file') ~= 2
    error('exportCodeIssues:MissingCanonical', ...
        'Canonical exportCodeIssues.m not found at: %s', canonicalDir);
end

% Ensure canonical directory is in MATLAB path
if ~contains(path, canonicalDir)
    addpath(canonicalDir);
end

% Delegate directly by evaluating in canonical directory
origDir = pwd;
cd(canonicalDir);
restoreDir = onCleanup(@() cd(origDir));

canonicalFunc = str2func('exportCodeIssues');
T = canonicalFunc(targetPath, varargin{:});

end
