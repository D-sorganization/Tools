function materials = loadMaterialData(filepath)
%LOADMATERIALDATA Load material properties from Python-exported MAT file
%
%   materials = loadMaterialData(filepath) reads material property data
%   from a .mat file created by the Python glass_bath_fea module.
%
%   Inputs:
%       filepath - Path to material_properties.mat file
%
%   Outputs:
%       materials - Structure with fields:
%           .base_conductivity   - Base conductivity at reference temp (S/m)
%           .activation_energy   - Activation energy for conduction (J/mol)
%           .reference_temp      - Reference temperature (K)
%           .composition_factor  - Composition correction factor
%           .gas_constant        - Gas constant R (J/(mol·K))
%           .metal_conductivity  - Metal layer conductivity (S/m)
%           .fulcher_A, B, T0    - Fulcher equation parameters for viscosity

    % Load data from file
    data = load(filepath);

    % Initialize output structure with defaults
    materials = struct();

    % Arrhenius parameters
    if isfield(data, 'base_conductivity')
        materials.base_conductivity = double(data.base_conductivity(1));
    else
        materials.base_conductivity = 1.0;  % Default
    end

    if isfield(data, 'activation_energy')
        materials.activation_energy = double(data.activation_energy(1));
    else
        materials.activation_energy = 80000;  % Default (J/mol)
    end

    if isfield(data, 'reference_temp')
        materials.reference_temp = double(data.reference_temp(1));
    else
        materials.reference_temp = 1473.15;  % Default: 1200°C in K
    end

    if isfield(data, 'composition_factor')
        materials.composition_factor = double(data.composition_factor(1));
    else
        materials.composition_factor = 1.0;  % Default
    end

    if isfield(data, 'gas_constant')
        materials.gas_constant = double(data.gas_constant(1));
    else
        materials.gas_constant = 8.314;  % J/(mol·K)
    end

    % Metal conductivity
    if isfield(data, 'metal_conductivity')
        materials.metal_conductivity = double(data.metal_conductivity(1));
    else
        materials.metal_conductivity = 10000.0;  % Default (S/m)
    end

    % Fulcher equation parameters for viscosity
    if isfield(data, 'fulcher_A')
        materials.fulcher_A = double(data.fulcher_A(1));
    else
        materials.fulcher_A = -2.0;
    end

    if isfield(data, 'fulcher_B')
        materials.fulcher_B = double(data.fulcher_B(1));
    else
        materials.fulcher_B = 4500.0;  % K
    end

    if isfield(data, 'fulcher_T0')
        materials.fulcher_T0 = double(data.fulcher_T0(1));
    else
        materials.fulcher_T0 = 250.0;  % K
    end

    % Operating temperature (if available)
    if isfield(data, 'operating_temperature')
        materials.operating_temperature = double(data.operating_temperature(1));
    else
        materials.operating_temperature = 1350;  % Default (°C)
    end

    % Print loaded values
    fprintf('    Loaded material properties:\n');
    fprintf('      Base conductivity: %.2f S/m\n', materials.base_conductivity);
    fprintf('      Activation energy: %.0f J/mol\n', materials.activation_energy);
    fprintf('      Reference temp: %.2f K\n', materials.reference_temp);
    fprintf('      Composition factor: %.3f\n', materials.composition_factor);
    fprintf('      Metal conductivity: %.0f S/m\n', materials.metal_conductivity);
end
