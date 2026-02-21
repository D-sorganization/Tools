function data = surface_sweep(T_range, param_name, param_range, P, feed, n_T, n_param, varargin)
%SURFACE_SWEEP Run 2D parameter sweep for surface plots.
%
%   data = surface_sweep(T_range, param_name, param_range, P, feed, n_T, n_param)
%
%   Inputs:
%     T_range     - [T_start, T_end] in Kelvin
%     param_name  - 'steam_carbon', 'oxygen_carbon', 'pressure',
%                   'ch4_flow', 'c3h8_flow', 'n2_purge', 'ng_flow'
%     param_range - [param_start, param_end]
%     P           - Base pressure [Pa] (overridden if param_name='pressure')
%     feed        - Element feed struct
%     n_T         - Number of temperature points
%     n_param     - Number of parameter points
%
%   Output:
%     data - Struct with:
%       .T_grid, .P_grid  - Meshgrid arrays
%       .compositions     - (n_T x n_param x n_species) array
%       .h2_co_ratio      - (n_T x n_param) array
%       .carbon_conv      - (n_T x n_param) array
%       .cge              - (n_T x n_param) array
%       .temperatures     - Temperature vector
%       .param_values     - Parameter vector

    temps = linspace(T_range(1), T_range(2), n_T);
    params = linspace(param_range(1), param_range(2), n_param);

    db = thermo_data();
    n_sp = db.n_species;

    compositions = zeros(n_T, n_param, n_sp);
    h2_co = zeros(n_T, n_param);
    c_conv = zeros(n_T, n_param);
    cge = zeros(n_T, n_param);

    total_pts = n_T * n_param;
    count = 0;

    for j = 1:n_param
        warm = [];
        for i = 1:n_T
            extra_args = varargin;

            switch param_name
                case 'steam_carbon'
                    extra_args = [extra_args, {'steam_carbon', params(j)}];
                    r = gasification_equilibrium(temps(i), P, feed, extra_args{:}, 'warm_start', warm);
                case 'oxygen_carbon'
                    extra_args = [extra_args, {'oxygen_carbon', params(j)}];
                    r = gasification_equilibrium(temps(i), P, feed, extra_args{:}, 'warm_start', warm);
                case 'pressure'
                    r = gasification_equilibrium(temps(i), params(j), feed, extra_args{:}, 'warm_start', warm);
                case 'ch4_flow'
                    extra_args = [extra_args, {'ch4_flow', params(j)}];
                    r = gasification_equilibrium(temps(i), P, feed, extra_args{:}, 'warm_start', warm);
                case 'c3h8_flow'
                    extra_args = [extra_args, {'c3h8_flow', params(j)}];
                    r = gasification_equilibrium(temps(i), P, feed, extra_args{:}, 'warm_start', warm);
                case 'n2_purge'
                    extra_args = [extra_args, {'n2_purge', params(j)}];
                    r = gasification_equilibrium(temps(i), P, feed, extra_args{:}, 'warm_start', warm);
                case 'ng_flow'
                    extra_args = [extra_args, {'ng_flow', params(j)}];
                    r = gasification_equilibrium(temps(i), P, feed, extra_args{:}, 'warm_start', warm);
                otherwise
                    error('Unknown parameter: %s', param_name);
            end

            compositions(i, j, :) = r.mole_frac;
            h2_co(i, j) = r.h2_co_ratio;
            c_conv(i, j) = r.carbon_conv;
            cge(i, j) = r.cold_gas_efficiency;

            if r.converged
                warm = r.moles;
            end

            count = count + 1;
            if mod(count, 50) == 0
                fprintf('  Surface: %d/%d (%.0f%%)\n', count, total_pts, 100*count/total_pts);
            end
        end
    end

    [data.T_grid, data.P_grid] = meshgrid(temps - 273.15, params);
    data.T_grid = data.T_grid';
    data.P_grid = data.P_grid';
    data.compositions = compositions;
    data.h2_co_ratio = h2_co;
    data.carbon_conv = c_conv;
    data.cge = cge;
    data.temperatures = temps;
    data.param_values = params;
    data.param_name = param_name;
    data.species = {db.species.key};

    fprintf('Surface sweep complete: %d points computed\n', total_pts);
end
