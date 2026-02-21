function result = gasification_equilibrium(T, P, feed, varargin)
%GASIFICATION_EQUILIBRIUM Solve chemical equilibrium via Gibbs minimization.
%
%   result = gasification_equilibrium(T, P, feed)
%   result = gasification_equilibrium(T, P, feed, 'steam_carbon', 1.0)
%
%   Inputs:
%     T    - Temperature [K], scalar > 0
%     P    - Pressure [Pa], scalar > 0
%     feed - Struct with element fields, e.g. struct('C',1,'H',1,'O',0.5)
%
%   Optional Name-Value Pairs:
%     'steam_carbon'  - Steam-to-carbon molar ratio (default: 0)
%     'oxygen_carbon' - O2-to-carbon molar ratio (default: 0)
%     'steam_flow'    - Direct steam injection [mol] (default: 0)
%     'o2_flow'       - Direct O2 injection [mol] (default: 0)
%     'use_air'       - Use air instead of pure O2 (default: false)
%     'n2_purge'      - N2 purge flow [mol] (default: 0)
%     'ch4_flow'      - CH4 injection [mol] (default: 0)
%     'c3h8_flow'     - C3H8 injection [mol] (default: 0)
%     'ng_flow'       - Natural gas injection [mol] (default: 0)
%     'warm_start'    - Initial moles vector from previous solution
%
%   Output:
%     result - Struct with fields:
%       .species      - Cell array of species keys
%       .mole_frac    - Mole fractions (gas-phase)
%       .moles        - Absolute moles per mol feed
%       .converged    - Boolean convergence flag
%       .h2_co_ratio  - H2/CO molar ratio
%       .carbon_conv  - Carbon conversion fraction
%       .cold_gas_efficiency - CGE on HHV basis
%       .gibbs_energy - Total dimensionless Gibbs energy
%       .feed_elements - Element balance used
%
%   Design by Contract:
%     Precondition:  T > 0, P > 0, feed has at least one element
%     Postcondition: Element balance error < 1e-6 if converged
%
%   Compatible with MATLAB and GNU Octave.

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'steam_carbon', 0);
    addParameter(p, 'oxygen_carbon', 0);
    addParameter(p, 'steam_flow', 0);
    addParameter(p, 'o2_flow', 0);
    addParameter(p, 'use_air', false);
    addParameter(p, 'n2_purge', 0);
    addParameter(p, 'ch4_flow', 0);
    addParameter(p, 'c3h8_flow', 0);
    addParameter(p, 'ng_flow', 0);
    addParameter(p, 'warm_start', []);
    parse(p, varargin{:});
    opts = p.Results;

    assert(T > 0, 'Temperature must be positive');
    assert(P > 0, 'Pressure must be positive');

    % Load thermodynamic database
    db = thermo_data();
    n_sp = db.n_species;

    % Build element list and composition matrix
    elements = db.elements;
    n_elem = db.n_elements;

    % Element-species matrix A(j,i) = atoms of element j in species i
    A = zeros(n_elem, n_sp);
    for i = 1:n_sp
        sp = db.species(i);
        for j = 1:n_elem
            if isfield(sp.elements, elements{j})
                A(j, i) = sp.elements.(elements{j});
            end
        end
    end

    % Build element target vector from feed
    b = zeros(n_elem, 1);
    for j = 1:n_elem
        if isfield(feed, elements{j})
            b(j) = feed.(elements{j});
        end
    end

    C_moles = b(1);  % Carbon is first element
    H_idx = 2; O_idx = 3; N_idx = 4;

    % Legacy ratio-based injections
    if opts.steam_carbon > 0 && C_moles > 0
        b(H_idx) = b(H_idx) + opts.steam_carbon * C_moles * 2;
        b(O_idx) = b(O_idx) + opts.steam_carbon * C_moles * 1;
    end
    if opts.oxygen_carbon > 0 && C_moles > 0
        b(O_idx) = b(O_idx) + opts.oxygen_carbon * C_moles * 2;
    end

    % Direct process injections
    if opts.steam_flow > 0
        b(H_idx) = b(H_idx) + opts.steam_flow * 2;
        b(O_idx) = b(O_idx) + opts.steam_flow * 1;
    end

    if opts.o2_flow > 0
        b(O_idx) = b(O_idx) + opts.o2_flow * 2;
        if opts.use_air
            air_moles = opts.o2_flow / 0.2095;
            b(N_idx) = b(N_idx) + air_moles * 0.7808 * 2;
        end
    end

    if opts.n2_purge > 0
        b(N_idx) = b(N_idx) + opts.n2_purge * 2;
    end

    if opts.ch4_flow > 0
        b(1) = b(1) + opts.ch4_flow * 1;
        b(H_idx) = b(H_idx) + opts.ch4_flow * 4;
    end

    if opts.c3h8_flow > 0
        b(1) = b(1) + opts.c3h8_flow * 3;
        b(H_idx) = b(H_idx) + opts.c3h8_flow * 8;
    end

    if opts.ng_flow > 0
        b(1) = b(1) + opts.ng_flow * 1.05;
        b(H_idx) = b(H_idx) + opts.ng_flow * 4.16;
        b(N_idx) = b(N_idx) + opts.ng_flow * 0.04;
    end

    % Store feed elements for output
    feed_elements = struct();
    for j = 1:n_elem
        if b(j) > 0
            feed_elements.(elements{j}) = b(j);
        end
    end

    % Remove unused elements (zero target, zero in all species)
    active = b > 0 | any(A > 0, 2);
    A_active = A(active, :);
    b_active = b(active);

    % Initial guess
    MIN_MOLES = 1e-15;
    if ~isempty(opts.warm_start) && length(opts.warm_start) == n_sp
        n0 = max(opts.warm_start, MIN_MOLES);
    else
        n0 = ones(n_sp, 1) * max(sum(b_active), 1) / n_sp;
    end

    % Gibbs energy function
    P_ref = db.P_ref;
    function G = total_gibbs(n)
        n_safe = max(n, MIN_MOLES);
        gas_total_inner = 0;
        for ii = 1:n_sp
            if strcmp(db.species(ii).phase, 'gas')
                gas_total_inner = gas_total_inner + n_safe(ii);
            end
        end
        gas_total_inner = max(gas_total_inner, MIN_MOLES);

        G = 0;
        ln_P_ratio = log(P / P_ref);
        for ii = 1:n_sp
            sp_inner = db.species(ii);
            if T <= sp_inner.T_mid
                coeffs = sp_inner.coeff_low;
            else
                coeffs = sp_inner.coeff_high;
            end
            g_std = gibbs_dimensionless(coeffs, T);

            if strcmp(sp_inner.phase, 'gas')
                x_i = n_safe(ii) / gas_total_inner;
                G = G + n_safe(ii) * (g_std + ln_P_ratio + log(max(x_i, MIN_MOLES)));
            else
                G = G + n_safe(ii) * g_std;
            end
        end
    end

    % Pin unconstrained species (all-zero columns) to MIN_MOLES
    col_sums = sum(abs(A_active), 1);
    lb = ones(n_sp, 1) * MIN_MOLES;
    ub = ones(n_sp, 1) * Inf;
    for ii = 1:n_sp
        if col_sums(ii) < 1e-15
            lb(ii) = MIN_MOLES;
            ub(ii) = MIN_MOLES;
            n0(ii) = MIN_MOLES;
        end
    end

    % Solve using fmincon (MATLAB) or sqp (Octave)
    converged = false;
    n_iter = 0;

    if exist('OCTAVE_VERSION', 'builtin')
        try
            [n_eq, ~, info] = sqp(n0, @total_gibbs, ...
                @(n) A_active * n - b_active, [], lb, ub, 2000, 1e-10);
            converged = (info == 101 || info == 0);
        catch
            n_eq = n0;
        end
    else
        options = optimoptions('fmincon', 'Algorithm', 'sqp', ...
            'Display', 'off', 'MaxIterations', 2000, ...
            'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-12);
        try
            [n_eq, ~, exitflag, output] = fmincon(@total_gibbs, n0, ...
                [], [], A_active, b_active, lb, ub, [], options);
            converged = (exitflag > 0);
            n_iter = output.iterations;
        catch
            n_eq = n0;
        end
    end

    n_eq = max(n_eq, 0);

    % Compute mole fractions (gas phase)
    gas_total = 0;
    for i = 1:n_sp
        if strcmp(db.species(i).phase, 'gas')
            gas_total = gas_total + n_eq(i);
        end
    end
    gas_total = max(gas_total, MIN_MOLES);

    mole_frac = zeros(n_sp, 1);
    for i = 1:n_sp
        if strcmp(db.species(i).phase, 'gas')
            mole_frac(i) = n_eq(i) / gas_total;
        end
    end

    % Element balance check
    balance_err = max(abs(A_active * n_eq - b_active));
    if balance_err < 1e-6
        converged = true;
    elseif balance_err > 1e-6
        converged = false;
    end

    % Species lookup
    species_keys = cell(n_sp, 1);
    for i = 1:n_sp
        species_keys{i} = db.species(i).key;
    end

    % H2/CO ratio
    h2_idx_sp = find(strcmp(species_keys, 'H2'));
    co_idx_sp = find(strcmp(species_keys, 'CO'));
    cs_idx_sp = find(strcmp(species_keys, 'C_solid'));

    h2_co = 0;
    if ~isempty(h2_idx_sp) && ~isempty(co_idx_sp) && n_eq(co_idx_sp) > 1e-12
        h2_co = n_eq(h2_idx_sp) / n_eq(co_idx_sp);
    end

    % Carbon conversion
    carbon_conv = 1.0;
    if ~isempty(cs_idx_sp) && C_moles > 1e-12
        carbon_conv = 1.0 - n_eq(cs_idx_sp) / C_moles;
        carbon_conv = max(0, min(1, carbon_conv));
    end

    % Cold gas efficiency (HHV basis)
    cge = 0;
    syngas_energy = 0;
    hhv_keys = {'H2', 'CO', 'CH4', 'C2H4', 'C2H6'};
    hhv_vals = [285.8, 283.0, 890.8, 1411.0, 1560.7];
    for k = 1:length(hhv_keys)
        sp_idx = find(strcmp(species_keys, hhv_keys{k}));
        if ~isempty(sp_idx)
            syngas_energy = syngas_energy + mole_frac(sp_idx) * gas_total * hhv_vals(k);
        end
    end
    feed_energy = b(1) * 393.5 + b(H_idx) * 0.5 * 285.8;
    if feed_energy > 0
        cge = min(syngas_energy / feed_energy, 2.0);
    end

    % Pack results
    result.species = species_keys;
    result.mole_frac = mole_frac;
    result.moles = n_eq;
    result.converged = converged;
    result.h2_co_ratio = h2_co;
    result.carbon_conv = carbon_conv;
    result.cold_gas_efficiency = cge;
    result.gibbs_energy = total_gibbs(n_eq);
    result.temperature = T;
    result.pressure = P;
    result.balance_error = balance_err;
    result.iterations = n_iter;
    result.gas_total = gas_total;
    result.feed_elements = feed_elements;
end
