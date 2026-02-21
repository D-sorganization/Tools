function results = temperature_sweep(T_start, T_end, n_points, P, feed, varargin)
%TEMPERATURE_SWEEP Run equilibrium across temperature range.
%
%   results = temperature_sweep(T_start, T_end, n_points, P, feed)
%   results = temperature_sweep(..., 'steam_carbon', 1.0)
%
%   Inputs:
%     T_start  - Start temperature [K]
%     T_end    - End temperature [K]
%     n_points - Number of temperature points
%     P        - Pressure [Pa]
%     feed     - Element feed struct
%
%   Output:
%     results - Struct with arrays:
%       .temperatures  - Temperature array [K]
%       .mole_fracs    - (n_sp x n_points) mole fraction matrix
%       .h2_co_ratio   - H2/CO ratio array
%       .carbon_conv   - Carbon conversion array
%       .species       - Cell array of species keys
%       .converged     - Boolean convergence array
%
%   Uses warm-starting for efficient sequential computation.
%   Compatible with MATLAB and GNU Octave.

    assert(T_start > 0, 'T_start must be positive');
    assert(T_end > T_start, 'T_end must be greater than T_start');
    assert(n_points >= 2, 'Need at least 2 points');

    temps = linspace(T_start, T_end, n_points);
    db = thermo_data();
    n_sp = db.n_species;

    % Preallocate
    mole_fracs = zeros(n_sp, n_points);
    h2_co = zeros(1, n_points);
    c_conv = zeros(1, n_points);
    conv_flags = false(1, n_points);

    warm = [];

    for k = 1:n_points
        r = gasification_equilibrium(temps(k), P, feed, varargin{:}, 'warm_start', warm);

        mole_fracs(:, k) = r.mole_frac;
        h2_co(k) = r.h2_co_ratio;
        c_conv(k) = r.carbon_conv;
        conv_flags(k) = r.converged;

        if r.converged
            warm = r.moles;
        end

        % Progress indicator
        if mod(k, 10) == 0
            fprintf('  Sweep: %d/%d (T=%.0f K)\n', k, n_points, temps(k));
        end
    end

    results.temperatures = temps;
    results.mole_fracs = mole_fracs;
    results.h2_co_ratio = h2_co;
    results.carbon_conv = c_conv;
    results.species = {db.species.key};
    results.converged = conv_flags;
    results.n_converged = sum(conv_flags);

    fprintf('Sweep complete: %d/%d points converged\n', results.n_converged, n_points);
end
