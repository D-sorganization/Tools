function gasification_app()
%GASIFICATION_APP Interactive gasification equilibrium calculator.
%
%   gasification_app()
%
%   Launches a tabbed interactive GUI with:
%     - Single Point equilibrium calculation
%     - Temperature Sweep plots
%     - 3D Surface plots
%     - Feed composition editor
%
%   Compatible with MATLAB and GNU Octave.
%   Uses only standard plotting functions and uicontrol widgets.

    % ─── State ───
    state.feed = struct('C', 1.0, 'H', 1.0, 'O', 0.5);
    state.T = 1000;
    state.P = 101325;
    state.steam_carbon = 0;
    state.oxygen_carbon = 0;
    state.current_tab = 1;

    % ─── Colors (dark theme) ───
    c.bg      = [0.04 0.05 0.09];
    c.panel   = [0.07 0.10 0.17];
    c.accent  = [0.00 0.83 1.00];
    c.accent2 = [1.00 0.42 0.21];
    c.accent3 = [0.49 0.30 1.00];
    c.text    = [0.88 0.90 0.93];
    c.dim     = [0.42 0.48 0.55];
    c.grid    = [0.12 0.18 0.26];
    c.success = [0.00 0.90 0.46];
    c.warn    = [1.00 0.67 0.00];

    % Species colors
    sp_colors = struct('H2', [0 0.83 1], 'CO', [1 0.42 0.21], ...
        'CO2', [0.49 0.3 1], 'H2O', [0 0.9 0.46], ...
        'CH4', [1 0.67 0], 'N2', [1 0.09 0.27], 'C_solid', [0.47 0.56 0.61]);

    % ─── Main figure ───
    fig = figure('Name', 'Gasification Equilibrium Calculator', ...
        'NumberTitle', 'off', 'Color', c.bg, ...
        'Position', [50 50 1200 750], 'MenuBar', 'none', ...
        'Resize', 'on');

    % ─── Tab buttons ───
    tab_names = {'Single Point', 'Temp Sweep', 'Surface Plots', 'Feed Editor'};
    tab_btns = zeros(1, 4);
    for i = 1:4
        tab_btns(i) = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
            'String', tab_names{i}, ...
            'Units', 'normalized', ...
            'Position', [0.01+(i-1)*0.245, 0.955, 0.235, 0.04], ...
            'BackgroundColor', c.panel, 'ForegroundColor', c.text, ...
            'FontWeight', 'bold', 'FontSize', 10, ...
            'Callback', @(~,~) switch_tab(i));
    end

    % ─── Tab 1: Single Point ───
    tab1_panel = uipanel('Parent', fig, 'Units', 'normalized', ...
        'Position', [0 0 1 0.95], 'BackgroundColor', c.bg, ...
        'BorderType', 'none', 'Visible', 'on');

    % Temperature slider
    uicontrol(tab1_panel, 'Style', 'text', 'String', 'Temperature [K]:', ...
        'Units', 'normalized', 'Position', [0.02 0.22 0.12 0.03], ...
        'BackgroundColor', c.bg, 'ForegroundColor', c.accent, ...
        'HorizontalAlignment', 'left', 'FontSize', 9);
    sl_T = uicontrol(tab1_panel, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.15 0.22 0.30 0.03], ...
        'Min', 300, 'Max', 2000, 'Value', 1000, ...
        'BackgroundColor', c.panel, 'Callback', @cb_calc);
    txt_T = uicontrol(tab1_panel, 'Style', 'text', 'String', '1000 K', ...
        'Units', 'normalized', 'Position', [0.46 0.22 0.06 0.03], ...
        'BackgroundColor', c.bg, 'ForegroundColor', c.accent, 'FontSize', 9);

    % Pressure slider
    uicontrol(tab1_panel, 'Style', 'text', 'String', 'Pressure [atm]:', ...
        'Units', 'normalized', 'Position', [0.02 0.17 0.12 0.03], ...
        'BackgroundColor', c.bg, 'ForegroundColor', c.accent2, ...
        'HorizontalAlignment', 'left', 'FontSize', 9);
    sl_P = uicontrol(tab1_panel, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.15 0.17 0.30 0.03], ...
        'Min', 0.1, 'Max', 50, 'Value', 1, ...
        'BackgroundColor', c.panel, 'Callback', @cb_calc);
    txt_P = uicontrol(tab1_panel, 'Style', 'text', 'String', '1.0 atm', ...
        'Units', 'normalized', 'Position', [0.46 0.17 0.06 0.03], ...
        'BackgroundColor', c.bg, 'ForegroundColor', c.accent2, 'FontSize', 9);

    % Steam/Carbon slider
    uicontrol(tab1_panel, 'Style', 'text', 'String', 'Steam/Carbon:', ...
        'Units', 'normalized', 'Position', [0.02 0.12 0.12 0.03], ...
        'BackgroundColor', c.bg, 'ForegroundColor', c.accent3, ...
        'HorizontalAlignment', 'left', 'FontSize', 9);
    sl_SC = uicontrol(tab1_panel, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.15 0.12 0.30 0.03], ...
        'Min', 0, 'Max', 5, 'Value', 0, ...
        'BackgroundColor', c.panel, 'Callback', @cb_calc);
    txt_SC = uicontrol(tab1_panel, 'Style', 'text', 'String', '0.0', ...
        'Units', 'normalized', 'Position', [0.46 0.12 0.06 0.03], ...
        'BackgroundColor', c.bg, 'ForegroundColor', c.accent3, 'FontSize', 9);

    % O2/Carbon slider
    uicontrol(tab1_panel, 'Style', 'text', 'String', 'O2/Carbon:', ...
        'Units', 'normalized', 'Position', [0.02 0.07 0.12 0.03], ...
        'BackgroundColor', c.bg, 'ForegroundColor', c.success, ...
        'HorizontalAlignment', 'left', 'FontSize', 9);
    sl_OC = uicontrol(tab1_panel, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.15 0.07 0.30 0.03], ...
        'Min', 0, 'Max', 2, 'Value', 0, ...
        'BackgroundColor', c.panel, 'Callback', @cb_calc);
    txt_OC = uicontrol(tab1_panel, 'Style', 'text', 'String', '0.0', ...
        'Units', 'normalized', 'Position', [0.46 0.07 0.06 0.03], ...
        'BackgroundColor', c.bg, 'ForegroundColor', c.success, 'FontSize', 9);

    % Calculate button
    uicontrol(tab1_panel, 'Style', 'pushbutton', 'String', 'CALCULATE', ...
        'Units', 'normalized', 'Position', [0.02 0.01 0.15 0.05], ...
        'BackgroundColor', [0 0.83 1], 'ForegroundColor', c.bg, ...
        'FontWeight', 'bold', 'FontSize', 11, 'Callback', @cb_calc);

    % Axes for bar chart
    ax1_bar = axes('Parent', tab1_panel, 'Units', 'normalized', ...
        'Position', [0.07 0.35 0.42 0.58], 'Color', c.panel, ...
        'XColor', c.dim, 'YColor', c.dim);

    % Axes for info
    ax1_info = axes('Parent', tab1_panel, 'Units', 'normalized', ...
        'Position', [0.55 0.08 0.42 0.85], 'Color', c.panel, ...
        'XColor', c.panel, 'YColor', c.panel);
    set(ax1_info, 'XTick', [], 'YTick', []);

    % ─── Tab 2: Temperature Sweep ───
    tab2_panel = uipanel('Parent', fig, 'Units', 'normalized', ...
        'Position', [0 0 1 0.95], 'BackgroundColor', c.bg, ...
        'BorderType', 'none', 'Visible', 'off');

    ax2_comp = axes('Parent', tab2_panel, 'Units', 'normalized', ...
        'Position', [0.08 0.52 0.55 0.42], 'Color', c.panel, ...
        'XColor', c.dim, 'YColor', c.dim);
    ax2_metrics = axes('Parent', tab2_panel, 'Units', 'normalized', ...
        'Position', [0.08 0.06 0.55 0.38], 'Color', c.panel, ...
        'XColor', c.dim, 'YColor', c.dim);

    uicontrol(tab2_panel, 'Style', 'pushbutton', 'String', 'RUN SWEEP', ...
        'Units', 'normalized', 'Position', [0.72 0.02 0.22 0.06], ...
        'BackgroundColor', [0 0.83 1], 'ForegroundColor', c.bg, ...
        'FontWeight', 'bold', 'FontSize', 11, 'Callback', @cb_sweep);

    % ─── Tab 3: Surface Plots ───
    tab3_panel = uipanel('Parent', fig, 'Units', 'normalized', ...
        'Position', [0 0 1 0.95], 'BackgroundColor', c.bg, ...
        'BorderType', 'none', 'Visible', 'off');

    ax3_surf = axes('Parent', tab3_panel, 'Units', 'normalized', ...
        'Position', [0.05 0.10 0.55 0.82], 'Color', c.panel);
    ax3_contour = axes('Parent', tab3_panel, 'Units', 'normalized', ...
        'Position', [0.62 0.50 0.35 0.42], 'Color', c.panel, ...
        'XColor', c.dim, 'YColor', c.dim);

    % Surface parameter selector
    surf_param_bg = uibuttongroup('Parent', tab3_panel, 'Units', 'normalized', ...
        'Position', [0.62 0.22 0.35 0.25], 'BackgroundColor', c.panel, ...
        'ForegroundColor', c.text, 'Title', 'Sweep Parameter');
    uicontrol(surf_param_bg, 'Style', 'radiobutton', 'String', 'Steam/Carbon', ...
        'Units', 'normalized', 'Position', [0.05 0.7 0.9 0.25], ...
        'BackgroundColor', c.panel, 'ForegroundColor', c.text, 'FontSize', 9, ...
        'Tag', 'steam_carbon');
    uicontrol(surf_param_bg, 'Style', 'radiobutton', 'String', 'O2/Carbon', ...
        'Units', 'normalized', 'Position', [0.05 0.4 0.9 0.25], ...
        'BackgroundColor', c.panel, 'ForegroundColor', c.text, 'FontSize', 9, ...
        'Tag', 'oxygen_carbon');
    uicontrol(surf_param_bg, 'Style', 'radiobutton', 'String', 'Pressure [atm]', ...
        'Units', 'normalized', 'Position', [0.05 0.1 0.9 0.25], ...
        'BackgroundColor', c.panel, 'ForegroundColor', c.text, 'FontSize', 9, ...
        'Tag', 'pressure');

    % Species selector
    sp_selector = uicontrol(tab3_panel, 'Style', 'popupmenu', ...
        'String', {'H2', 'CO', 'CO2', 'CH4', 'H2/CO Ratio'}, ...
        'Units', 'normalized', 'Position', [0.62 0.12 0.15 0.06], ...
        'BackgroundColor', c.panel, 'ForegroundColor', c.text, 'FontSize', 9);

    uicontrol(tab3_panel, 'Style', 'pushbutton', 'String', 'GENERATE SURFACE', ...
        'Units', 'normalized', 'Position', [0.80 0.02 0.17 0.06], ...
        'BackgroundColor', [1 0.42 0.21], 'ForegroundColor', c.bg, ...
        'FontWeight', 'bold', 'FontSize', 10, 'Callback', @cb_surface);

    % ─── Tab 4: Feed Editor ───
    tab4_panel = uipanel('Parent', fig, 'Units', 'normalized', ...
        'Position', [0 0 1 0.95], 'BackgroundColor', c.bg, ...
        'BorderType', 'none', 'Visible', 'off');

    ax4_comp = axes('Parent', tab4_panel, 'Units', 'normalized', ...
        'Position', [0.08 0.40 0.40 0.52], 'Color', c.panel, ...
        'XColor', c.dim, 'YColor', c.dim);
    ax4_eq = axes('Parent', tab4_panel, 'Units', 'normalized', ...
        'Position', [0.55 0.40 0.40 0.52], 'Color', c.panel, ...
        'XColor', c.dim, 'YColor', c.dim);

    % Preset selector
    presets = {'Bituminous Coal', 'Sub-bituminous', 'Lignite', ...
               'Biomass (Wood)', 'Petcoke', 'Natural Gas', 'Custom'};
    uicontrol(tab4_panel, 'Style', 'text', 'String', 'Feed Preset:', ...
        'Units', 'normalized', 'Position', [0.02 0.30 0.10 0.03], ...
        'BackgroundColor', c.bg, 'ForegroundColor', c.text, 'FontSize', 9);
    preset_popup = uicontrol(tab4_panel, 'Style', 'popupmenu', ...
        'String', presets, 'Value', 7, ...
        'Units', 'normalized', 'Position', [0.13 0.30 0.20 0.04], ...
        'BackgroundColor', c.panel, 'ForegroundColor', c.text, ...
        'Callback', @cb_preset);

    % Element editors
    elem_labels = {'C', 'H', 'O', 'N', 'S'};
    elem_edits = zeros(1, 5);
    elem_defaults = [1.0, 1.0, 0.5, 0.0, 0.0];
    for i = 1:5
        uicontrol(tab4_panel, 'Style', 'text', 'String', [elem_labels{i} ':'], ...
            'Units', 'normalized', 'Position', [0.40 0.32-i*0.05 0.05 0.03], ...
            'BackgroundColor', c.bg, 'ForegroundColor', c.text, ...
            'HorizontalAlignment', 'right', 'FontSize', 10, 'FontWeight', 'bold');
        elem_edits(i) = uicontrol(tab4_panel, 'Style', 'edit', ...
            'String', num2str(elem_defaults(i)), ...
            'Units', 'normalized', 'Position', [0.46 0.32-i*0.05 0.10 0.04], ...
            'BackgroundColor', c.panel, 'ForegroundColor', c.accent, 'FontSize', 10);
    end

    uicontrol(tab4_panel, 'Style', 'pushbutton', 'String', 'APPLY FEED', ...
        'Units', 'normalized', 'Position', [0.62 0.02 0.18 0.06], ...
        'BackgroundColor', [0 0.9 0.46], 'ForegroundColor', c.bg, ...
        'FontWeight', 'bold', 'FontSize', 11, 'Callback', @cb_apply_feed);

    panels = {tab1_panel, tab2_panel, tab3_panel, tab4_panel};

    % Initial calculation
    cb_calc();

    % ═════════════════════════════════════════════════════════════════════
    % CALLBACKS
    % ═════════════════════════════════════════════════════════════════════

    function switch_tab(idx)
        state.current_tab = idx;
        for ti = 1:4
            set(panels{ti}, 'Visible', 'off');
            set(tab_btns(ti), 'BackgroundColor', c.panel, 'ForegroundColor', c.text);
        end
        set(panels{idx}, 'Visible', 'on');
        set(tab_btns(idx), 'BackgroundColor', [0 0.83 1], 'ForegroundColor', c.bg);
    end

    function cb_calc(~, ~)
        T = round(get(sl_T, 'Value'));
        P_atm = get(sl_P, 'Value');
        sc = get(sl_SC, 'Value');
        oc = get(sl_OC, 'Value');

        set(txt_T, 'String', sprintf('%d K', T));
        set(txt_P, 'String', sprintf('%.1f atm', P_atm));
        set(txt_SC, 'String', sprintf('%.1f', sc));
        set(txt_OC, 'String', sprintf('%.1f', oc));

        state.T = T;
        state.P = P_atm * 101325;
        state.steam_carbon = sc;
        state.oxygen_carbon = oc;

        r = gasification_equilibrium(state.T, state.P, state.feed, ...
            'steam_carbon', state.steam_carbon, ...
            'oxygen_carbon', state.oxygen_carbon);

        plot_single_point(r);
    end

    function plot_single_point(r)
        % Bar chart
        cla(ax1_bar);
        axes(ax1_bar);

        db = thermo_data();
        gas_idx = [];
        gas_labels = {};
        gas_vals = [];
        gas_colors = [];

        for i = 1:db.n_species
            if strcmp(db.species(i).phase, 'gas')
                gas_idx(end+1) = i;
                gas_labels{end+1} = db.species(i).formula;
                gas_vals(end+1) = r.mole_frac(i) * 100;
                if isfield(sp_colors, db.species(i).key)
                    gas_colors(end+1, :) = sp_colors.(db.species(i).key);
                else
                    gas_colors(end+1, :) = [0.5 0.5 0.5];
                end
            end
        end

        barh(ax1_bar, gas_vals, 'FaceColor', 'flat', 'EdgeColor', 'none');
        b = barh(ax1_bar, gas_vals);
        b.FaceColor = 'flat';
        b.CData = gas_colors;
        b.EdgeColor = 'none';
        set(ax1_bar, 'YTickLabel', gas_labels, 'YDir', 'reverse', ...
            'Color', c.panel, 'XColor', c.dim, 'YColor', c.text, ...
            'FontSize', 9);
        xlabel(ax1_bar, 'Mole Fraction [%]', 'Color', c.text);
        title(ax1_bar, 'Equilibrium Gas Composition', 'Color', c.accent, ...
            'FontWeight', 'bold', 'FontSize', 12);

        % Value labels
        for i = 1:length(gas_vals)
            if gas_vals(i) > 0.5
                text(ax1_bar, gas_vals(i)+0.3, i, sprintf('%.2f%%', gas_vals(i)), ...
                    'Color', c.text, 'FontSize', 8, 'FontWeight', 'bold', ...
                    'VerticalAlignment', 'middle');
            end
        end
        grid(ax1_bar, 'on');
        set(ax1_bar, 'GridColor', c.grid, 'GridAlpha', 0.4);

        % Info panel
        cla(ax1_info);
        axes(ax1_info);
        set(ax1_info, 'XLim', [0 1], 'YLim', [0 1], 'XTick', [], 'YTick', []);

        if r.converged
            status_str = 'CONVERGED';
            status_col = c.success;
        else
            status_str = 'NOT CONVERGED';
            status_col = [1 0.09 0.27];
        end

        lines = {
            {sprintf('Status: %s', status_str), status_col, 14};
            {' ', c.text, 8};
            {sprintf('T = %d K  (%.0f \\circC)', r.temperature, r.temperature-273.15), c.accent, 11};
            {sprintf('P = %.2f atm', r.pressure/101325), c.accent2, 11};
            {' ', c.text, 8};
            {sprintf('H_2/CO = %.3f', r.h2_co_ratio), c.text, 12};
            {sprintf('Carbon Conv. = %.1f%%', r.carbon_conv*100), c.warn, 11};
            {' ', c.text, 8};
            {sprintf('Balance Error = %.2e', r.balance_error), c.dim, 9};
            {sprintf('Iterations = %d', r.iterations), c.dim, 9};
        };

        for i = 1:length(lines)
            text(ax1_info, 0.05, 1.0 - i*0.09, lines{i}{1}, ...
                'Color', lines{i}{2}, 'FontSize', lines{i}{3}, ...
                'FontWeight', 'bold', 'Interpreter', 'tex');
        end
    end

    function cb_sweep(~, ~)
        set(fig, 'Name', 'Computing Temperature Sweep...');
        drawnow;

        results = temperature_sweep(400, 1600, 50, state.P, state.feed, ...
            'steam_carbon', state.steam_carbon, ...
            'oxygen_carbon', state.oxygen_carbon);

        plot_sweep(results);
        set(fig, 'Name', 'Gasification Equilibrium Calculator');
    end

    function plot_sweep(results)
        temps_c = results.temperatures - 273.15;
        db = thermo_data();

        % Composition plot
        cla(ax2_comp);
        axes(ax2_comp);
        hold(ax2_comp, 'on');

        plot_species = {'H2', 'CO', 'CO2', 'H2O', 'CH4'};
        for s = 1:length(plot_species)
            idx = find(strcmp(results.species, plot_species{s}));
            if ~isempty(idx)
                vals = results.mole_fracs(idx, :) * 100;
                if isfield(sp_colors, plot_species{s})
                    col = sp_colors.(plot_species{s});
                else
                    col = [0.5 0.5 0.5];
                end
                plot(ax2_comp, temps_c, vals, 'Color', col, 'LineWidth', 2.5, ...
                    'DisplayName', plot_species{s});
            end
        end

        xlabel(ax2_comp, 'Temperature [\circC]', 'Color', c.text);
        ylabel(ax2_comp, 'Mole Fraction [%]', 'Color', c.text);
        title(ax2_comp, 'Equilibrium Composition vs Temperature', ...
            'Color', c.accent, 'FontWeight', 'bold', 'FontSize', 12);
        legend(ax2_comp, 'show', 'Location', 'best', 'TextColor', c.text, ...
            'Color', c.panel, 'EdgeColor', c.grid);
        grid(ax2_comp, 'on');
        set(ax2_comp, 'Color', c.panel, 'XColor', c.dim, 'YColor', c.dim, ...
            'GridColor', c.grid, 'GridAlpha', 0.4);
        hold(ax2_comp, 'off');

        % Metrics plot
        cla(ax2_metrics);
        axes(ax2_metrics);

        yyaxis(ax2_metrics, 'left');
        plot(ax2_metrics, temps_c, results.h2_co_ratio, 'Color', c.accent, ...
            'LineWidth', 2.5);
        ylabel(ax2_metrics, 'H_2/CO Ratio', 'Color', c.accent);
        set(ax2_metrics, 'YColor', c.accent);

        yyaxis(ax2_metrics, 'right');
        hold(ax2_metrics, 'on');
        plot(ax2_metrics, temps_c, results.carbon_conv * 100, '--', ...
            'Color', c.warn, 'LineWidth', 2, 'DisplayName', 'C Conv [%]');
        hold(ax2_metrics, 'off');
        ylabel(ax2_metrics, 'Carbon Conversion [%]', 'Color', c.warn);
        set(ax2_metrics, 'YColor', c.warn);

        xlabel(ax2_metrics, 'Temperature [\circC]', 'Color', c.text);
        title(ax2_metrics, 'Process Metrics', 'Color', c.accent2, ...
            'FontWeight', 'bold', 'FontSize', 12);
        grid(ax2_metrics, 'on');
        set(ax2_metrics, 'Color', c.panel, 'XColor', c.dim, ...
            'GridColor', c.grid, 'GridAlpha', 0.4);
        legend(ax2_metrics, 'show', 'Location', 'best', 'TextColor', c.text, ...
            'Color', c.panel, 'EdgeColor', c.grid);
    end

    function cb_surface(~, ~)
        set(fig, 'Name', 'Computing Surface Plot...');
        drawnow;

        % Get selected parameter
        sel = get(get(surf_param_bg, 'SelectedObject'), 'Tag');
        switch sel
            case 'steam_carbon'
                pname = 'steam_carbon';
                prange = [0 3];
            case 'oxygen_carbon'
                pname = 'oxygen_carbon';
                prange = [0 1.5];
            case 'pressure'
                pname = 'pressure';
                prange = [101325*0.5, 101325*30];
            otherwise
                pname = 'steam_carbon';
                prange = [0 3];
        end

        data = surface_sweep([400 1600], pname, prange, state.P, state.feed, 20, 15);

        % Get selected species
        sp_idx_sel = get(sp_selector, 'Value');
        sp_names = {'H2', 'CO', 'CO2', 'CH4', 'h2_co'};
        sp_sel = sp_names{sp_idx_sel};

        plot_surface(data, sp_sel, pname);
        set(fig, 'Name', 'Gasification Equilibrium Calculator');
    end

    function plot_surface(data, sp_sel, pname)
        if strcmp(sp_sel, 'h2_co')
            Z = data.h2_co_ratio;
            z_label = 'H_2/CO Ratio';
        else
            idx = find(strcmp(data.species, sp_sel));
            Z = data.compositions(:, :, idx) * 100;
            z_label = [sp_sel ' [mol%]'];
        end

        param_labels = struct('steam_carbon', 'Steam/Carbon', ...
            'oxygen_carbon', 'O_2/Carbon', 'pressure', 'Pressure [atm]');
        if strcmp(pname, 'pressure')
            P_display = data.P_grid / 101325;
        else
            P_display = data.P_grid;
        end

        % 3D Surface
        cla(ax3_surf);
        axes(ax3_surf);
        surf(ax3_surf, data.T_grid, P_display, Z, 'EdgeColor', 'none', ...
            'FaceAlpha', 0.9);
        colormap(ax3_surf, 'parula');
        xlabel(ax3_surf, 'Temperature [\circC]', 'Color', c.accent);
        ylabel(ax3_surf, param_labels.(pname), 'Color', c.accent2);
        zlabel(ax3_surf, z_label, 'Color', c.text);
        title(ax3_surf, [z_label ' Surface'], 'Color', c.accent, ...
            'FontWeight', 'bold', 'FontSize', 12);
        set(ax3_surf, 'Color', c.panel, 'XColor', c.dim, 'YColor', c.dim, ...
            'ZColor', c.dim);
        colorbar(ax3_surf, 'Color', c.dim);
        view(ax3_surf, -37.5, 30);
        grid(ax3_surf, 'on');

        % Contour
        cla(ax3_contour);
        axes(ax3_contour);
        contourf(ax3_contour, data.T_grid, P_display, Z, 20);
        colormap(ax3_contour, 'parula');
        xlabel(ax3_contour, 'Temperature [\circC]', 'Color', c.text);
        ylabel(ax3_contour, param_labels.(pname), 'Color', c.text);
        title(ax3_contour, [z_label ' Contour'], 'Color', c.accent2, ...
            'FontWeight', 'bold', 'FontSize', 11);
        set(ax3_contour, 'Color', c.panel, 'XColor', c.dim, 'YColor', c.dim);
        colorbar(ax3_contour, 'Color', c.dim);
    end

    function cb_preset(~, ~)
        idx = get(preset_popup, 'Value');
        switch idx
            case 1  % Bituminous
                state.feed = struct('C', 0.75, 'H', 0.05, 'O', 0.08, 'N', 0.015, 'S', 0.01);
            case 2  % Sub-bituminous
                state.feed = struct('C', 0.60, 'H', 0.04, 'O', 0.15, 'N', 0.01, 'S', 0.005);
            case 3  % Lignite
                state.feed = struct('C', 0.45, 'H', 0.03, 'O', 0.20, 'N', 0.008, 'S', 0.01);
            case 4  % Biomass
                state.feed = struct('C', 0.50, 'H', 0.06, 'O', 0.42, 'N', 0.002, 'S', 0.001);
            case 5  % Petcoke
                state.feed = struct('C', 0.88, 'H', 0.04, 'O', 0.01, 'N', 0.015, 'S', 0.05);
            case 6  % Natural Gas
                state.feed = struct('C', 1.0, 'H', 4.0);
            case 7  % Custom
                state.feed = struct('C', 1.0, 'H', 1.0, 'O', 0.5);
        end
        update_feed_display();
    end

    function cb_apply_feed(~, ~)
        state.feed = struct();
        for i = 1:5
            val = str2double(get(elem_edits(i), 'String'));
            if ~isnan(val) && val > 0
                state.feed.(elem_labels{i}) = val;
            end
        end
        update_feed_display();
        cb_calc();
    end

    function update_feed_display()
        % Update edit boxes
        for i = 1:5
            if isfield(state.feed, elem_labels{i})
                set(elem_edits(i), 'String', num2str(state.feed.(elem_labels{i})));
            else
                set(elem_edits(i), 'String', '0');
            end
        end

        % Plot feed composition
        cla(ax4_comp);
        axes(ax4_comp);

        fnames = fieldnames(state.feed);
        fvals = zeros(length(fnames), 1);
        fcolors = [1 0.42 0.21; 0 0.83 1; 0 0.9 0.46; 0.49 0.3 1; 1 0.67 0];
        for i = 1:length(fnames)
            fvals(i) = state.feed.(fnames{i});
        end
        b = bar(ax4_comp, fvals, 'FaceColor', 'flat', 'EdgeColor', 'none');
        b.CData = fcolors(1:length(fnames), :);
        set(ax4_comp, 'XTickLabel', fnames, 'Color', c.panel, ...
            'XColor', c.text, 'YColor', c.dim);
        title(ax4_comp, 'Feed Composition', 'Color', c.accent, ...
            'FontWeight', 'bold', 'FontSize', 12);
        ylabel(ax4_comp, 'Molar Amount', 'Color', c.text);
        grid(ax4_comp, 'on');
        set(ax4_comp, 'GridColor', c.grid, 'GridAlpha', 0.4);

        % Quick equilibrium preview
        try
            r = gasification_equilibrium(1000, 101325, state.feed);
            cla(ax4_eq);
            axes(ax4_eq);

            db = thermo_data();
            sig_sp = {};
            sig_vals = [];
            sig_colors = [];
            for i = 1:db.n_species
                if r.mole_frac(i) > 0.005
                    sig_sp{end+1} = db.species(i).formula;
                    sig_vals(end+1) = r.mole_frac(i) * 100;
                    if isfield(sp_colors, db.species(i).key)
                        sig_colors(end+1, :) = sp_colors.(db.species(i).key);
                    else
                        sig_colors(end+1, :) = [0.5 0.5 0.5];
                    end
                end
            end
            if ~isempty(sig_vals)
                bh = barh(ax4_eq, sig_vals, 'FaceColor', 'flat', 'EdgeColor', 'none');
                bh.CData = sig_colors;
                set(ax4_eq, 'YTickLabel', sig_sp, 'Color', c.panel, ...
                    'XColor', c.dim, 'YColor', c.text);
                xlabel(ax4_eq, 'Mole %', 'Color', c.text);
                title(ax4_eq, 'Equilibrium at 1000 K', 'Color', c.accent2, ...
                    'FontWeight', 'bold', 'FontSize', 12);
                grid(ax4_eq, 'on');
            end
        catch
            % Skip preview on error
        end
    end

    % Show initial tab
    switch_tab(1);
    update_feed_display();
end
