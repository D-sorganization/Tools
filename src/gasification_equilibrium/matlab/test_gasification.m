function test_gasification()
%TEST_GASIFICATION Run all tests for the gasification equilibrium module.
%
%   test_gasification()
%
%   Runs comprehensive tests of:
%     - Thermodynamic data integrity (14 species incl. C3H8)
%     - NASA polynomial evaluations
%     - Equilibrium solver convergence
%     - Known chemical equilibria
%     - Process injection features
%     - Temperature sweep functionality
%
%   Compatible with MATLAB and GNU Octave.

    passed = 0;
    failed = 0;
    total = 0;

    fprintf('\n======================================================\n');
    fprintf('  Gasification Equilibrium - Test Suite (Phase 2)     \n');
    fprintf('======================================================\n\n');

    % --- Thermodynamic Data Tests ---
    fprintf('--- Thermodynamic Data ---\n');
    db = thermo_data();

    [passed, failed, total] = check(db.n_species >= 14, 'At least 14 species in database', passed, failed, total);
    [passed, failed, total] = check(abs(db.R - 8.314) < 0.01, 'Gas constant R = 8.314', passed, failed, total);
    [passed, failed, total] = check(db.P_ref == 101325, 'Reference pressure = 101325 Pa', passed, failed, total);

    % Check all species have valid coefficients and T ranges
    for i = 1:db.n_species
        sp = db.species(i);
        ok = length(sp.coeff_low) == 7 && length(sp.coeff_high) == 7;
        [passed, failed, total] = check(ok, sprintf('%s: 7 coefficients each', sp.key), passed, failed, total);
        ok2 = sp.T_low < sp.T_mid && sp.T_mid < sp.T_high;
        [passed, failed, total] = check(ok2, sprintf('%s: valid T range', sp.key), passed, failed, total);
    end

    % Check new species are present
    keys = {db.species.key};
    [passed, failed, total] = check(any(strcmp(keys, 'C3H8')), 'Propane (C3H8) in database', passed, failed, total);
    [passed, failed, total] = check(any(strcmp(keys, 'O2')), 'Oxygen (O2) in database', passed, failed, total);
    [passed, failed, total] = check(any(strcmp(keys, 'C2H4')), 'Ethylene (C2H4) in database', passed, failed, total);
    [passed, failed, total] = check(any(strcmp(keys, 'C2H6')), 'Ethane (C2H6) in database', passed, failed, total);
    [passed, failed, total] = check(any(strcmp(keys, 'H2S')), 'H2S in database', passed, failed, total);
    [passed, failed, total] = check(any(strcmp(keys, 'NH3')), 'NH3 in database', passed, failed, total);
    [passed, failed, total] = check(any(strcmp(keys, 'SO2')), 'SO2 in database', passed, failed, total);

    % --- Gibbs Function Tests ---
    fprintf('\n--- Gibbs Polynomial ---\n');

    for T = [300, 500, 800, 1000, 1500]
        for i = 1:db.n_species
            sp = db.species(i);
            if T <= sp.T_mid
                coeffs = sp.coeff_low;
            else
                coeffs = sp.coeff_high;
            end
            g = gibbs_dimensionless(coeffs, T);
            ok = isfinite(g);
            [passed, failed, total] = check(ok, sprintf('%s at %dK: G/RT is finite', sp.key, T), passed, failed, total);
        end
    end

    % --- Single Point Equilibrium ---
    fprintf('\n--- Single Point Equilibrium ---\n');

    feed = struct('C', 1.0, 'H', 1.0, 'O', 1.0);
    r = gasification_equilibrium(1000, 101325, feed);
    [passed, failed, total] = check(r.converged, '1000K convergence', passed, failed, total);
    [passed, failed, total] = check(r.balance_error < 1e-4, '1000K element balance', passed, failed, total);

    mf_sum = sum(r.mole_frac);
    [passed, failed, total] = check(abs(mf_sum - 1.0) < 0.05, 'Mole fractions sum to ~1', passed, failed, total);
    [passed, failed, total] = check(all(r.mole_frac >= 0), 'All mole fractions >= 0', passed, failed, total);

    r500 = gasification_equilibrium(500, 101325, feed);
    [passed, failed, total] = check(r500.converged, '500K convergence', passed, failed, total);

    r1500 = gasification_equilibrium(1500, 101325, feed);
    [passed, failed, total] = check(r1500.converged, '1500K convergence', passed, failed, total);

    % --- Known Equilibria ---
    fprintf('\n--- Known Chemical Equilibria ---\n');

    % Boudouard: CO > CO2 at high T
    r_boud = gasification_equilibrium(1200, 101325, struct('C', 2, 'O', 2));
    co_idx = find(strcmp(r_boud.species, 'CO'));
    co2_idx = find(strcmp(r_boud.species, 'CO2'));
    [passed, failed, total] = check(r_boud.mole_frac(co_idx) > r_boud.mole_frac(co2_idx), ...
        'Boudouard: CO > CO2 at 1200K', passed, failed, total);

    % Boudouard: CO2 > CO at low T
    r_boud_low = gasification_equilibrium(400, 101325, struct('C', 2, 'O', 2));
    [passed, failed, total] = check(r_boud_low.mole_frac(co2_idx) > r_boud_low.mole_frac(co_idx), ...
        'Boudouard: CO2 > CO at 400K', passed, failed, total);

    % Methanation: more CH4 at low T
    ch4_idx = find(strcmp(r_boud.species, 'CH4'));
    r_met_low = gasification_equilibrium(500, 101325, struct('C', 1, 'H', 4));
    r_met_high = gasification_equilibrium(1200, 101325, struct('C', 1, 'H', 4));
    [passed, failed, total] = check(r_met_low.mole_frac(ch4_idx) > r_met_high.mole_frac(ch4_idx), ...
        'Methanation: CH4 at 500K > 1200K', passed, failed, total);

    % --- Pressure Effects ---
    fprintf('\n--- Pressure Effects ---\n');

    r_1atm = gasification_equilibrium(800, 101325, struct('C', 1, 'H', 4));
    r_30atm = gasification_equilibrium(800, 101325*30, struct('C', 1, 'H', 4));
    [passed, failed, total] = check(r_30atm.mole_frac(ch4_idx) > r_1atm.mole_frac(ch4_idx), ...
        'Le Chatelier: more CH4 at high P', passed, failed, total);

    % --- Process Injection Tests ---
    fprintf('\n--- Process Injections ---\n');

    % Steam injection via ratio
    r_no_steam = gasification_equilibrium(1000, 101325, struct('C', 1, 'O', 0.5));
    r_with_steam = gasification_equilibrium(1000, 101325, struct('C', 1, 'O', 0.5), ...
        'steam_carbon', 1.0);
    h2_idx = find(strcmp(r_no_steam.species, 'H2'));
    [passed, failed, total] = check(r_with_steam.mole_frac(h2_idx) > r_no_steam.mole_frac(h2_idx), ...
        'Steam ratio increases H2', passed, failed, total);

    % Direct steam flow
    r_direct_steam = gasification_equilibrium(1000, 101325, struct('C', 1, 'O', 0.5), ...
        'steam_flow', 1.0);
    [passed, failed, total] = check(r_direct_steam.converged, ...
        'Direct steam flow converges', passed, failed, total);

    % O2 injection
    r_o2 = gasification_equilibrium(1000, 101325, struct('C', 1, 'H', 1), ...
        'o2_flow', 0.5);
    [passed, failed, total] = check(r_o2.converged, 'O2 injection converges', passed, failed, total);

    % Air mode
    r_air = gasification_equilibrium(1000, 101325, struct('C', 1, 'H', 1), ...
        'o2_flow', 0.5, 'use_air', true);
    [passed, failed, total] = check(r_air.converged, 'Air mode converges', passed, failed, total);
    [passed, failed, total] = check(isfield(r_air.feed_elements, 'N'), ...
        'Air mode adds nitrogen', passed, failed, total);

    % N2 purge
    r_n2 = gasification_equilibrium(1000, 101325, struct('C', 1, 'H', 1, 'O', 0.5), ...
        'n2_purge', 1.0);
    [passed, failed, total] = check(r_n2.converged, 'N2 purge converges', passed, failed, total);

    % CH4 injection
    r_ch4 = gasification_equilibrium(1000, 101325, struct('C', 1, 'O', 0.5), ...
        'ch4_flow', 0.5);
    [passed, failed, total] = check(r_ch4.converged, 'CH4 injection converges', passed, failed, total);

    % C3H8 injection
    r_c3h8 = gasification_equilibrium(1000, 101325, struct('C', 1, 'O', 0.5), ...
        'c3h8_flow', 0.5);
    [passed, failed, total] = check(r_c3h8.converged, 'C3H8 injection converges', passed, failed, total);

    % Natural gas injection
    r_ng = gasification_equilibrium(1000, 101325, struct('C', 1, 'O', 0.5), ...
        'ng_flow', 1.0);
    [passed, failed, total] = check(r_ng.converged, 'Natural gas injection converges', passed, failed, total);

    % --- Contract Violations ---
    fprintf('\n--- Design by Contract ---\n');

    try
        gasification_equilibrium(-100, 101325, struct('C', 1));
        [passed, failed, total] = check(false, 'Negative T should raise error', passed, failed, total);
    catch
        [passed, failed, total] = check(true, 'Negative T raises error', passed, failed, total);
    end

    try
        gasification_equilibrium(1000, -1, struct('C', 1));
        [passed, failed, total] = check(false, 'Negative P should raise error', passed, failed, total);
    catch
        [passed, failed, total] = check(true, 'Negative P raises error', passed, failed, total);
    end

    % --- Temperature Sweep ---
    fprintf('\n--- Temperature Sweep ---\n');

    sweep = temperature_sweep(600, 1400, 10, 101325, struct('C', 1, 'H', 1, 'O', 0.5));
    [passed, failed, total] = check(length(sweep.temperatures) == 10, ...
        'Sweep returns 10 points', passed, failed, total);
    [passed, failed, total] = check(sweep.n_converged >= 8, ...
        sprintf('Sweep convergence: %d/10', sweep.n_converged), passed, failed, total);

    % --- Summary ---
    fprintf('\n======================================================\n');
    fprintf('  RESULTS: %d passed, %d failed, %d total\n', passed, failed, total);
    if failed == 0
        fprintf('  ALL TESTS PASSED\n');
    else
        fprintf('  %d TEST(S) FAILED\n', failed);
    end
    fprintf('======================================================\n\n');
end

function [p, f, t] = check(condition, name, p, f, t)
    t = t + 1;
    if condition
        p = p + 1;
        fprintf('  [PASS] %s\n', name);
    else
        f = f + 1;
        fprintf('  [FAIL] %s\n', name);
    end
end
