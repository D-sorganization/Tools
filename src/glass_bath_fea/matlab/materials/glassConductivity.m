function sigma = glassConductivity(location, state, materials)
%GLASSCONDUCTIVITY Temperature-dependent electrical conductivity of glass
%
%   sigma = glassConductivity(location, state, materials) calculates the
%   electrical conductivity using the Arrhenius equation with composition
%   correction factors.
%
%   Arrhenius equation:
%       sigma(T) = sigma_0 * C_comp * exp(-Ea/R * (1/T - 1/T_ref))
%
%   Inputs:
%       location - Structure with x, y, z coordinates
%       state - Current solution state (for nonlinear problems)
%       materials - Material property structure
%
%   Outputs:
%       sigma - Electrical conductivity (S/m)
%
%   The composition factor accounts for:
%       - Na2O content (increases ionic mobility)
%       - Fe2O3 content (increases electronic conduction)

    % Extract Arrhenius parameters
    sigma_0 = materials.base_conductivity;      % Base conductivity (S/m)
    Ea = materials.activation_energy;           % Activation energy (J/mol)
    R = materials.gas_constant;                 % Gas constant (J/(mol·K))
    T_ref = materials.reference_temp;           % Reference temperature (K)
    C_comp = materials.composition_factor;      % Composition correction

    % Get temperature
    % For coupled thermal-electrical, temperature would come from state
    % For now, use operating temperature
    if isfield(state, 'u') && ~isempty(state.u)
        % If solving coupled problem, u might contain temperature
        T_celsius = state.u;
    else
        % Use constant operating temperature
        if isfield(materials, 'operating_temperature')
            T_celsius = materials.operating_temperature;
        else
            T_celsius = 1350;  % Default
        end
    end

    % Convert to Kelvin
    T_K = T_celsius + 273.15;

    % Calculate conductivity using Arrhenius equation
    % sigma(T) = sigma_0 * C_comp * exp(-Ea/R * (1/T - 1/T_ref))
    arrhenius_term = exp(-Ea / R * (1./T_K - 1/T_ref));
    sigma = sigma_0 * C_comp * arrhenius_term;

    % Ensure output is same size as input locations
    if isstruct(location)
        numPoints = numel(location.x);
        if isscalar(sigma)
            sigma = sigma * ones(1, numPoints);
        end
    end
end
