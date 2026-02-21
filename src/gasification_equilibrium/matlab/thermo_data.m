function db = thermo_data()
%THERMO_DATA NASA 7-coefficient polynomial thermodynamic database.
%
%   db = thermo_data()
%
%   Returns a struct array of species with NASA polynomial coefficients
%   for computing thermodynamic properties (Cp, H, S, G).
%
%   Compatible with both MATLAB and GNU Octave.
%
%   Design by Contract:
%     Postcondition: All species have valid T ranges and 7-element coefficients
%
%   Data source: NASA Glenn coefficients (Burcat & Ruscic)

    % Universal gas constant [J/(mol*K)]
    db.R = 8.314462618;
    db.P_ref = 101325.0;  % Reference pressure [Pa]
    db.T_ref = 298.15;    % Reference temperature [K]

    % Heating values [kJ/mol] for CGE calculation
    db.HHV = struct('H2', 285.8, 'CO', 283.0, 'CH4', 890.8, 'C2H4', 1411.0, 'C2H6', 1560.7);

    % --- Species definitions ---
    i = 0;

    i = i + 1;
    db.species(i).key = 'H2';
    db.species(i).name = 'Hydrogen';
    db.species(i).formula = 'H_2';
    db.species(i).mw = 2.016;
    db.species(i).elements = struct('H', 2);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [2.34433112, 7.98052075e-03, -1.94781510e-05, 2.01572094e-08, -7.37611761e-12, -917.935173, 0.683010238];
    db.species(i).coeff_high = [3.33727920, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10, 2.00255376e-14, -950.158922, -3.20502331];

    i = i + 1;
    db.species(i).key = 'CO';
    db.species(i).name = 'Carbon Monoxide';
    db.species(i).formula = 'CO';
    db.species(i).mw = 28.010;
    db.species(i).elements = struct('C', 1, 'O', 1);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [3.57953347, -6.10353680e-04, 1.01681433e-06, 9.07005884e-10, -9.04424499e-13, -14344.086, 3.50840928];
    db.species(i).coeff_high = [2.71518561, 2.06252743e-03, -9.98825771e-07, 2.30053008e-10, -2.03647716e-14, -14151.8724, 7.81868772];

    i = i + 1;
    db.species(i).key = 'CO2';
    db.species(i).name = 'Carbon Dioxide';
    db.species(i).formula = 'CO_2';
    db.species(i).mw = 44.009;
    db.species(i).elements = struct('C', 1, 'O', 2);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [2.35677352, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13, -48371.9697, 9.90105222];
    db.species(i).coeff_high = [3.85746029, 4.41437026e-03, -2.21481404e-06, 5.23490188e-10, -4.72084164e-14, -48759.166, 2.27163806];

    i = i + 1;
    db.species(i).key = 'H2O';
    db.species(i).name = 'Water';
    db.species(i).formula = 'H_2O';
    db.species(i).mw = 18.015;
    db.species(i).elements = struct('H', 2, 'O', 1);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [4.19864056, -2.03643410e-03, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12, -30293.7267, -0.849032208];
    db.species(i).coeff_high = [3.03399249, 2.17691804e-03, -1.64072518e-07, -9.70419870e-11, 1.68200992e-14, -30004.2971, 4.96677010];

    i = i + 1;
    db.species(i).key = 'CH4';
    db.species(i).name = 'Methane';
    db.species(i).formula = 'CH_4';
    db.species(i).mw = 16.043;
    db.species(i).elements = struct('C', 1, 'H', 4);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [5.14987613, -1.36709788e-02, 4.91800599e-05, -4.84743026e-08, 1.66693956e-11, -10246.6476, -4.64130376];
    db.species(i).coeff_high = [0.074851495, 1.33909467e-02, -5.73285809e-06, 1.22292535e-09, -1.01815230e-13, -9468.34459, 18.437318];

    i = i + 1;
    db.species(i).key = 'N2';
    db.species(i).name = 'Nitrogen';
    db.species(i).formula = 'N_2';
    db.species(i).mw = 28.014;
    db.species(i).elements = struct('N', 2);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [3.53100528, -1.23660988e-04, -5.02999433e-07, 2.43530612e-09, -1.40881235e-12, -1046.97628, 2.96747038];
    db.species(i).coeff_high = [2.95257637, 1.39690040e-03, -4.92631603e-07, 7.86010195e-11, -4.60755204e-15, -923.948688, 5.87188762];

    i = i + 1;
    db.species(i).key = 'O2';
    db.species(i).name = 'Oxygen';
    db.species(i).formula = 'O_2';
    db.species(i).mw = 31.998;
    db.species(i).elements = struct('O', 2);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12, -1063.94356, 3.65767573];
    db.species(i).coeff_high = [3.28253784, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14, -1088.45772, 5.45323129];

    i = i + 1;
    db.species(i).key = 'C2H4';
    db.species(i).name = 'Ethylene';
    db.species(i).formula = 'C_2H_4';
    db.species(i).mw = 28.054;
    db.species(i).elements = struct('C', 2, 'H', 4);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [3.95920148, -7.57052247e-03, 5.70990292e-05, -6.91588753e-08, 2.69884373e-11, 5089.77593, 4.09733096];
    db.species(i).coeff_high = [2.03611116, 1.46454151e-02, -6.71077915e-06, 1.47222923e-09, -1.25706061e-13, 4939.88614, 10.3053693];

    i = i + 1;
    db.species(i).key = 'C2H6';
    db.species(i).name = 'Ethane';
    db.species(i).formula = 'C_2H_6';
    db.species(i).mw = 30.070;
    db.species(i).elements = struct('C', 2, 'H', 6);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [4.29142492, -5.50154270e-03, 5.99438288e-05, -7.08466285e-08, 2.68685771e-11, -11522.2055, 2.66682316];
    db.species(i).coeff_high = [1.07188150, 2.16852677e-02, -1.00256067e-05, 2.21412001e-09, -1.90002890e-13, -12426.5222, 15.1156107];

    i = i + 1;
    db.species(i).key = 'H2S';
    db.species(i).name = 'Hydrogen Sulfide';
    db.species(i).formula = 'H_2S';
    db.species(i).mw = 34.082;
    db.species(i).elements = struct('H', 2, 'S', 1);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [4.12023590, -3.24803220e-03, 1.67209781e-05, -1.73457074e-08, 6.30820488e-12, -3650.53590, 1.72021024];
    db.species(i).coeff_high = [2.88324232, 3.81130960e-03, -1.47230893e-06, 2.74093019e-10, -1.98241636e-14, -3455.11880, 8.00522400];

    i = i + 1;
    db.species(i).key = 'NH3';
    db.species(i).name = 'Ammonia';
    db.species(i).formula = 'NH_3';
    db.species(i).mw = 17.031;
    db.species(i).elements = struct('N', 1, 'H', 3);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [4.28648920, -4.66055869e-03, 2.17119913e-05, -2.28063689e-08, 8.26395924e-12, -6741.72790, -0.625282180];
    db.species(i).coeff_high = [2.63455580, 5.66694560e-03, -1.72891830e-06, 2.38672510e-10, -1.25756950e-14, -6544.69590, 6.56632780];

    i = i + 1;
    db.species(i).key = 'SO2';
    db.species(i).name = 'Sulfur Dioxide';
    db.species(i).formula = 'SO_2';
    db.species(i).mw = 64.066;
    db.species(i).elements = struct('S', 1, 'O', 2);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [3.26653380, 5.32379020e-03, 6.84375520e-07, -5.28100470e-09, 2.55904540e-12, -36908.14500, 9.66465108];
    db.species(i).coeff_high = [5.24513640, 1.97042040e-03, -8.03757690e-07, 1.51499690e-10, -1.05580040e-14, -37550.73400, -1.07404890];

    i = i + 1;
    db.species(i).key = 'C3H8';
    db.species(i).name = 'Propane';
    db.species(i).formula = 'C_3H_8';
    db.species(i).mw = 44.096;
    db.species(i).elements = struct('C', 3, 'H', 8);
    db.species(i).phase = 'gas';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [4.21093028, 1.73880780e-03, 7.09192623e-05, -9.20376658e-08, 3.64238090e-11, -14381.0876, 5.61282290];
    db.species(i).coeff_high = [0.75341368, 3.18290557e-02, -1.49584302e-05, 3.34975168e-09, -2.90088803e-13, -16467.5165, 17.4587913];

    i = i + 1;
    db.species(i).key = 'C_solid';
    db.species(i).name = 'Graphite';
    db.species(i).formula = 'C(s)';
    db.species(i).mw = 12.011;
    db.species(i).elements = struct('C', 1);
    db.species(i).phase = 'solid';
    db.species(i).T_low = 200; db.species(i).T_mid = 1000; db.species(i).T_high = 3500;
    db.species(i).coeff_low  = [-0.310872072, 4.40353686e-03, 1.90394118e-06, -6.38546966e-09, 2.98964248e-12, -108.650974, 1.11382953];
    db.species(i).coeff_high = [1.45571829, 1.71702216e-03, -6.97562786e-07, 1.35277032e-10, -1.00589440e-14, -695.138840, -8.52583033];

    db.n_species = i;

    % Element list for matrix construction
    db.elements = {'C', 'H', 'O', 'N', 'S'};
    db.n_elements = length(db.elements);

    % Atomic weights [g/mol]
    db.atomic_weights = struct('C', 12.011, 'H', 1.008, 'O', 15.999, ...
                               'N', 14.007, 'S', 32.06);
end
