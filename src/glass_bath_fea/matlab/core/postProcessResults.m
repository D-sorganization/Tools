function [processedResults, figures] = postProcessResults(model, results, outputPath)
%POSTPROCESSRESULTS Post-process and visualize FEA results
%
%   [processedResults, figures] = postProcessResults(model, results, outputPath)
%   creates visualizations and exports results to files.
%
%   Inputs:
%       model - PDE model
%       results - Solution results from solveElectricalHeating
%       outputPath - Directory for output files
%
%   Outputs:
%       processedResults - Structure with processed data
%       figures - Cell array of figure handles

    figures = {};

    % Create output directory if needed
    if ~exist(outputPath, 'dir')
        mkdir(outputPath);
    end

    % Extract mesh data
    nodes = model.Mesh.Nodes;
    elements = model.Mesh.Elements;

    %% Figure 1: Voltage Distribution
    fig1 = figure('Name', 'Voltage Distribution', 'Position', [100, 100, 800, 600]);

    % 3D scatter plot of voltage
    subplot(1, 2, 1);
    scatter3(nodes(1,:), nodes(2,:), nodes(3,:), 10, results.Voltage, 'filled');
    colorbar;
    colormap(jet);
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('Electric Potential (V)');
    axis equal;
    view(45, 30);

    % Cross-section at mid-height
    subplot(1, 2, 2);
    z_mid = mean(nodes(3,:));
    idx_slice = abs(nodes(3,:) - z_mid) < 0.05 * (max(nodes(3,:)) - min(nodes(3,:)));
    scatter(nodes(1,idx_slice), nodes(2,idx_slice), 20, results.Voltage(idx_slice), 'filled');
    colorbar;
    colormap(jet);
    xlabel('X (m)');
    ylabel('Y (m)');
    title(sprintf('Voltage at Z = %.3f m', z_mid));
    axis equal;

    saveas(fig1, fullfile(outputPath, 'voltage_distribution.png'));
    figures{end+1} = fig1;

    %% Figure 2: Current Density
    fig2 = figure('Name', 'Current Density', 'Position', [150, 150, 800, 600]);

    J_mag = results.CurrentDensity.magnitude;

    subplot(1, 2, 1);
    scatter3(nodes(1,:), nodes(2,:), nodes(3,:), 10, J_mag, 'filled');
    colorbar;
    colormap(hot);
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('Current Density Magnitude (A/m^2)');
    axis equal;
    view(45, 30);

    % Histogram of current density
    subplot(1, 2, 2);
    histogram(J_mag, 50);
    xlabel('Current Density (A/m^2)');
    ylabel('Count');
    title('Current Density Distribution');
    grid on;

    saveas(fig2, fullfile(outputPath, 'current_density.png'));
    figures{end+1} = fig2;

    %% Figure 3: Power Dissipation
    fig3 = figure('Name', 'Power Density', 'Position', [200, 200, 800, 600]);

    P = results.PowerDensity;

    subplot(1, 2, 1);
    scatter3(nodes(1,:), nodes(2,:), nodes(3,:), 10, P, 'filled');
    colorbar;
    colormap(hot);
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('Power Density (W/m^3)');
    axis equal;
    view(45, 30);

    % Cross-section
    subplot(1, 2, 2);
    scatter(nodes(1,idx_slice), nodes(2,idx_slice), 20, P(idx_slice), 'filled');
    colorbar;
    colormap(hot);
    xlabel('X (m)');
    ylabel('Y (m)');
    title(sprintf('Power Density at Z = %.3f m', z_mid));
    axis equal;

    saveas(fig3, fullfile(outputPath, 'power_density.png'));
    figures{end+1} = fig3;

    %% Figure 4: Conductivity Distribution
    fig4 = figure('Name', 'Conductivity', 'Position', [250, 250, 600, 500]);

    sigma = results.Conductivity;
    scatter3(nodes(1,:), nodes(2,:), nodes(3,:), 10, log10(sigma), 'filled');
    colorbar;
    colormap(parula);
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('log_{10}(Conductivity) (S/m)');
    axis equal;
    view(45, 30);

    saveas(fig4, fullfile(outputPath, 'conductivity.png'));
    figures{end+1} = fig4;

    %% Calculate Summary Statistics
    processedResults = struct();

    % Total power (W)
    processedResults.totalPower = results.TotalPower;

    % Maximum values
    processedResults.maxVoltage = max(results.Voltage);
    processedResults.minVoltage = min(results.Voltage);
    processedResults.maxCurrentDensity = results.MaxCurrentDensity;
    processedResults.maxPowerDensity = results.MaxPowerDensity;

    % Average values
    processedResults.avgVoltage = mean(results.Voltage);
    processedResults.avgCurrentDensity = mean(J_mag);
    processedResults.avgPowerDensity = mean(P);

    % Spatial statistics
    processedResults.numNodes = size(nodes, 2);
    processedResults.numElements = size(elements, 2);

    % Operating temperature (assumed uniform)
    if isfield(model.UserData, 'materials')
        if isfield(model.UserData.materials, 'operating_temperature')
            processedResults.avgTemperature = model.UserData.materials.operating_temperature;
        else
            processedResults.avgTemperature = 1350;
        end
    else
        processedResults.avgTemperature = 1350;
    end

    %% Save Results to MAT File
    save(fullfile(outputPath, 'fea_results.mat'), ...
        'results', 'processedResults', '-v7.3');

    %% Create Summary Report
    reportFile = fullfile(outputPath, 'analysis_report.txt');
    fid = fopen(reportFile, 'w');
    fprintf(fid, 'Glass Bath FEA - Analysis Report\n');
    fprintf(fid, '================================\n\n');
    fprintf(fid, 'Date: %s\n\n', datestr(now));
    fprintf(fid, 'Mesh Statistics:\n');
    fprintf(fid, '  Nodes: %d\n', processedResults.numNodes);
    fprintf(fid, '  Elements: %d\n\n', processedResults.numElements);
    fprintf(fid, 'Electrical Results:\n');
    fprintf(fid, '  Voltage range: %.2f to %.2f V\n', ...
        processedResults.minVoltage, processedResults.maxVoltage);
    fprintf(fid, '  Max current density: %.2f A/m^2\n', processedResults.maxCurrentDensity);
    fprintf(fid, '  Total power: %.2f W (%.2f kW)\n', ...
        processedResults.totalPower, processedResults.totalPower/1000);
    fprintf(fid, '\nOperating Temperature: %.1f C\n', processedResults.avgTemperature);
    fclose(fid);

    fprintf('    Report saved: %s\n', reportFile);
end
