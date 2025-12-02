function run_all()
%RUN_ALL Recreates key results end-to-end.
%
%   RUN_ALL() executes a complete end-to-end workflow to recreate key
%   results, including configuration, output directory preparation, and
%   metadata saving.
%
%   Output:
%   -------
%   Creates output directory structure and saves metadata JSON file.
%
%   Example:
%   --------
%   run_all();
%
%   See also: datetime, datestr, jsonencode

arguments
end

% Standard reproducibility seed (commonly used in scientific computing)
% Value: 42 [dimensionless] Standard seed for reproducibility (commonly used in scientific computing)
REPRODUCIBILITY_SEED = 42;

% 1) Configure reproducibility
rng(REPRODUCIBILITY_SEED);

% 2) Prepare output directory
outdir = fullfile('output', datestr(datetime('now'),'yyyy-mm-dd'), 'baseline');
try
    if ~isfolder(outdir)
        mkdir(outdir);
    end
catch ME
    error('Failed to create output directory: %s', ME.message);
end

% 3) Save metadata
meta.date = datestr(datetime('now'));
meta.matlab_version = version;
meta.commit_sha = 'Commit SHA to be injected via CI pipeline';
meta.description = 'Baseline run_all template';
fid = fopen(fullfile(outdir, 'metadata.json'),'w');
fprintf(fid, '%s', jsonencode(meta));
fclose(fid);

% 4) Placeholder for simulations and plots
fprintf('run_all completed. Outputs in %s\n', outdir);
end
