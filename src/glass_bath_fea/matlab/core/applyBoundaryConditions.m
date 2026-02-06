function model = applyBoundaryConditions(model, bcData, meshData)
%APPLYBOUNDARYCONDITIONS Apply electrode voltage boundary conditions
%
%   model = applyBoundaryConditions(model, bcData, meshData) applies
%   Dirichlet boundary conditions (voltage) at electrode faces and
%   Neumann conditions (zero current) at outer boundaries.
%
%   Inputs:
%       model - PDE model from setupPDEModel
%       bcData - Boundary condition data from Python
%       meshData - Mesh data for geometry reference
%
%   Outputs:
%       model - Model with boundary conditions applied
%
%   Three-phase delta configuration:
%       Electrode 1: V_a = V * sin(0)
%       Electrode 2: V_b = V * sin(2*pi/3)
%       Electrode 3: V_c = V * sin(4*pi/3)

    % Get electrode voltages from boundary condition data
    if isfield(bcData, 'electrode_voltages')
        voltages = bcData.electrode_voltages;
    elseif isfield(bcData, 'phase_voltages')
        voltages = bcData.phase_voltages;
    else
        % Default: 100V per phase
        voltages = [100, 100, 100];
    end

    % Ensure voltages is a row vector
    voltages = voltages(:)';

    % Get electrode positions
    if isfield(bcData, 'tip_positions')
        tipPositions = bcData.tip_positions;
    else
        error('Electrode tip positions not found in boundary condition data');
    end

    % Get number of electrodes
    numElectrodes = length(voltages);
    fprintf('    Applying voltages to %d electrodes\n', numElectrodes);

    % Get electrode angles for phase assignment
    if isfield(bcData, 'electrode_angles')
        angles = bcData.electrode_angles;
    else
        angles = (0:numElectrodes-1) * 2 * pi / numElectrodes;
    end

    % Three-phase voltage calculation
    % V(t) = V_peak * sin(omega*t + phase_angle)
    % For DC steady-state analysis, use RMS values
    omega_t = 0;  % Phase reference
    phaseAngles = [0, 2*pi/3, 4*pi/3];  % 120° apart

    % Apply voltage boundary conditions at electrode faces
    % This requires identifying which faces correspond to electrodes
    numFaces = model.Geometry.NumFaces;

    % Method 1: Apply based on geometric proximity to electrode positions
    % (Used when electrode faces are not explicitly tagged)
    try
        for i = 1:numElectrodes
            % Calculate phase voltage (RMS)
            if numel(voltages) >= i
                V_rms = voltages(i);
            else
                V_rms = voltages(1);
            end

            % Apply three-phase pattern
            V_phase = V_rms * sin(omega_t + phaseAngles(i));

            % Find faces near electrode tip
            % This is an approximation - proper implementation would
            % tag faces during mesh generation
            tipPos = tipPositions(i, :);

            % Apply Dirichlet BC at electrode region
            % Using electromagnetic BC for conduction model
            try
                % Find face closest to electrode tip
                faceIdx = findElectrodeFace(model, tipPos);
                if ~isempty(faceIdx)
                    electromagneticBC(model, 'Face', faceIdx, 'Voltage', V_phase);
                    fprintf('    Electrode %d: V = %.1f V (Face %d)\n', ...
                        i, V_phase, faceIdx);
                end
            catch
                % Fallback: apply boundary condition by region
                applyBoundaryCondition(model, 'dirichlet', ...
                    'Edge', i, 'u', V_phase);
            end
        end
    catch ME
        warning('Could not apply electrode BCs: %s', ME.message);
        % Apply simplified boundary conditions
        applyBoundaryCondition(model, 'dirichlet', 'Face', 1, 'u', voltages(1));
    end

    % Apply insulating boundary condition on outer walls
    % (Zero normal current density: J·n = 0)
    try
        % Get all boundary faces
        allFaces = 1:numFaces;

        % Exclude electrode faces (approximation)
        wallFaces = setdiff(allFaces, 1:numElectrodes);

        if ~isempty(wallFaces)
            electromagneticBC(model, 'Face', wallFaces, 'CurrentDensity', 0);
        end
    catch
        % Neumann BC is default (natural BC)
    end

    % Store boundary condition info
    model.UserData.voltages = voltages;
    model.UserData.numElectrodes = numElectrodes;
end


function faceIdx = findElectrodeFace(model, tipPos)
%FINDELECTRODEFACE Find face closest to electrode tip position
%
%   Helper function to identify which mesh face corresponds to
%   an electrode based on geometric proximity.

    faceIdx = [];

    % Get face centroids
    numFaces = model.Geometry.NumFaces;

    if numFaces < 1
        return;
    end

    % Default: return first few faces as electrodes
    % (This is a placeholder - proper implementation would use
    % mesh tagging during generation)
    faceIdx = 1;

    % TODO: Implement proper face identification based on geometry
    % This would involve:
    % 1. Getting face vertex coordinates
    % 2. Computing face centroids
    % 3. Finding face closest to tipPos
end
