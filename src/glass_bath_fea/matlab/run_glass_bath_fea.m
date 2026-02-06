%% Glass Bath FEA - Master Workflow Script
% Finite Element Analysis of molten glass electrical heating
% using MATLAB PDE Toolbox
%
% This script coordinates the complete FEA workflow:
%   1. Load mesh and material data from Python exports
%   2. Setup PDE model with electromagnetic conduction
%   3. Apply boundary conditions (electrode voltages)
%   4. Solve for current density and power dissipation
%   5. Post-process and visualize results
%
% Requires: PDE Toolbox (R2020b or later)
%
% Author: Glass Bath FEA Team
% Date: 2025

clear; clc; close all;

%% Configuration
% Path to exported data from Python
dataPath = './fea_export/';  % Adjust as needed

% Output path for results
outputPath = './results/';
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

fprintf('Glass Bath FEA - Starting Analysis\n');
fprintf('==================================\n\n');

%% Step 1: Load Data
fprintf('Step 1: Loading mesh and material data...\n');

% Load mesh
meshFile = fullfile(dataPath, 'mesh.mat');
if exist(meshFile, 'file')
    meshData = load(meshFile);
    fprintf('  - Loaded mesh: %d nodes, %d elements\n', ...
        size(meshData.p, 2), size(meshData.t, 2));
else
    error('Mesh file not found: %s', meshFile);
end

% Load material properties
materialFile = fullfile(dataPath, 'material_properties.mat');
if exist(materialFile, 'file')
    materials = loadMaterialData(materialFile);
    fprintf('  - Loaded material properties\n');
else
    error('Material file not found: %s', materialFile);
end

% Load boundary conditions
bcFile = fullfile(dataPath, 'boundary_conditions.mat');
if exist(bcFile, 'file')
    bcData = load(bcFile);
    fprintf('  - Loaded boundary conditions\n');
else
    error('Boundary conditions file not found: %s', bcFile);
end

% Load configuration
configFile = fullfile(dataPath, 'config.mat');
if exist(configFile, 'file')
    config = load(configFile);
    fprintf('  - Loaded configuration\n');
else
    warning('Configuration file not found, using defaults');
    config = struct();
end

%% Step 2: Setup PDE Model
fprintf('\nStep 2: Setting up PDE model...\n');

model = setupPDEModel(meshData, materials);
fprintf('  - PDE model created\n');

%% Step 3: Apply Boundary Conditions
fprintf('\nStep 3: Applying boundary conditions...\n');

model = applyBoundaryConditions(model, bcData, meshData);
fprintf('  - Electrode voltages applied\n');

%% Step 4: Solve
fprintf('\nStep 4: Solving electrical heating problem...\n');

tic;
results = solveElectricalHeating(model, materials);
solveTime = toc;
fprintf('  - Solution completed in %.2f seconds\n', solveTime);

%% Step 5: Post-Process
fprintf('\nStep 5: Post-processing results...\n');

[processedResults, figures] = postProcessResults(model, results, outputPath);
fprintf('  - Results saved to: %s\n', outputPath);

%% Summary
fprintf('\n==================================\n');
fprintf('Analysis Complete!\n\n');
fprintf('Key Results:\n');
if isfield(processedResults, 'totalPower')
    fprintf('  - Total power dissipation: %.2f kW\n', ...
        processedResults.totalPower / 1000);
end
if isfield(processedResults, 'maxCurrentDensity')
    fprintf('  - Max current density: %.2f A/m^2\n', ...
        processedResults.maxCurrentDensity);
end
if isfield(processedResults, 'avgTemperature')
    fprintf('  - Average temperature: %.1f C\n', ...
        processedResults.avgTemperature);
end

fprintf('\nFigures saved to: %s\n', outputPath);
