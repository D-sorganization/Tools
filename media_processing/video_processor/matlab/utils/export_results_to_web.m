function export_results_to_web(results, output_file)
    % EXPORT_RESULTS_TO_WEB Export MATLAB results to JSON for web platform
    %
    % This function converts MATLAB analysis results to JSON format
    % that can be imported by the web platform.
    %
    % Inputs:
    %   results     - Struct with analysis results
    %                 Required fields:
    %                 .trajectory - Joint angles over time
    %                 .forces    - Forces at joints
    %                 .energy    - Energy components
    %                 .fit_error - Model fit error
    %   output_file - String path to output JSON file
    %
    % Example:
    %   results = pendulum_model(pose_data);
    %   export_results_to_web(results, 'results.json');
    %
    % See also: pendulum_model, import_matlab_results

    arguments
        results (1,1) struct
        output_file (1,1) string
    end

    % Convert MATLAB struct to JSON-compatible format
    json_data = struct();

    % Trajectory: Convert to row vectors (JSON array of arrays)
    if isfield(results, 'trajectory') && ~isempty(results.trajectory)
        json_data.trajectory = results.trajectory';
    end

    % Forces: Convert to row vectors
    if isfield(results, 'forces') && ~isempty(results.forces)
        json_data.forces = results.forces';
    end

    % Energy: Convert nested struct
    if isfield(results, 'energy')
        json_data.energy = struct();
        if isfield(results.energy, 'kinetic')
            json_data.energy.kinetic = results.energy.kinetic;
        end
        if isfield(results.energy, 'potential')
            json_data.energy.potential = results.energy.potential;
        end
        if isfield(results.energy, 'total')
            json_data.energy.total = results.energy.total;
        end
    end

    % Fit error
    if isfield(results, 'fit_error')
        json_data.fit_error = results.fit_error;
    end

    % Timestamp
    json_data.timestamp = datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss');

    % Metadata
    json_data.matlab_version = version;
    json_data.model_version = '1.0.0';

    % Write JSON file
    try
        json_text = jsonencode(json_data);
        fid = fopen(output_file, 'w');
        if fid == -1
            error('EXPORT_RESULTS_TO_WEB:FileError', ...
                'Cannot open file for writing: %s', output_file);
        end
        fprintf(fid, '%s', json_text);
        fclose(fid);

        fprintf('Results exported to: %s\n', output_file);
    catch ME
        error('EXPORT_RESULTS_TO_WEB:ExportError', ...
            'Error exporting results: %s', ME.message);
    end
end
