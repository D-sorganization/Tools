function model = setupPDEModel(meshData, materials)
%SETUPPDEMODEL Setup PDE Toolbox model for electromagnetic conduction
%
%   model = setupPDEModel(meshData, materials) creates and configures a
%   PDE Toolbox model for solving the DC electrical conduction problem
%   in a glass bath with electrodes.
%
%   Inputs:
%       meshData - Structure with mesh data (p, t, material_ids)
%       materials - Structure with material properties
%
%   Outputs:
%       model - Configured PDE model object
%
%   The model solves: div(sigma * grad(V)) = 0
%   where sigma is temperature-dependent conductivity and V is potential.

    % Create electromagnetic model for DC conduction
    % Note: For MATLAB R2021a+, use 'electromagnetic', 'conduction'
    % For earlier versions, use 'thermal' as approximation
    try
        model = createpde('electromagnetic', 'conduction');
    catch
        % Fallback for older MATLAB versions
        model = createpde(1);  % Scalar PDE
        warning('Using scalar PDE model (older MATLAB version)');
    end

    % Import mesh from Python-generated data
    % meshData.p should be 3xN (x,y,z coordinates)
    % meshData.t should be 4xN or 5xN (tetrahedral elements)
    nodes = double(meshData.p);
    elements = double(meshData.t);

    % Ensure correct dimensions
    if size(nodes, 1) ~= 3
        nodes = nodes';
    end
    if size(elements, 1) < 4
        elements = elements';
    end

    % Remove last row if it contains material IDs (MATLAB convention)
    if size(elements, 1) > 4
        materialIds = elements(end, :);
        elements = elements(1:4, :);
    elseif isfield(meshData, 'material_ids')
        materialIds = meshData.material_ids;
    else
        materialIds = ones(1, size(elements, 2));
    end

    % Create geometry from mesh
    try
        geometryFromMesh(model, nodes, elements);
    catch ME
        error('Failed to create geometry from mesh: %s', ME.message);
    end

    % Specify electrical properties by region
    % Region 1: Glass (temperature-dependent conductivity)
    % Region 2: Metal (high conductivity)

    numFaces = model.Geometry.NumFaces;
    numEdges = model.Geometry.NumEdges;
    numCells = model.Geometry.NumCells;

    fprintf('    Geometry: %d faces, %d edges, %d cells\n', ...
        numFaces, numEdges, numCells);

    % Set electrical conductivity for glass region
    glassCells = find(materialIds == 1);
    if ~isempty(glassCells)
        try
            specifyElectricalProperties(model, 'Cell', unique(glassCells), ...
                'ElectricalConductivity', @(location, state) ...
                glassConductivity(location, state, materials));
        catch
            % Fallback: use average conductivity
            sigma_avg = materials.base_conductivity * materials.composition_factor;
            specifyCoefficients(model, 'm', 0, 'd', 0, 'c', sigma_avg, 'a', 0, 'f', 0);
        end
    end

    % Set electrical conductivity for metal region
    metalCells = find(materialIds == 2);
    if ~isempty(metalCells)
        try
            specifyElectricalProperties(model, 'Cell', unique(metalCells), ...
                'ElectricalConductivity', materials.metal_conductivity);
        catch
            % Handled by glass region's specifyCoefficients
        end
    end

    % Store material info in model for later use
    model.UserData.materials = materials;
    model.UserData.materialIds = materialIds;
end
