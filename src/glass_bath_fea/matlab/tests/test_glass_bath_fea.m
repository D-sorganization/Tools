%% Glass Bath FEA - Test Script
% Unit tests for MATLAB FEA solver components
%
% Run with: runtests('test_glass_bath_fea')
%
% Requires: MATLAB Unit Test Framework

classdef test_glass_bath_fea < matlab.unittest.TestCase

    properties
        TestDataPath
        Materials
    end

    methods (TestClassSetup)
        function setupTestData(testCase)
            % Create test data directory
            testCase.TestDataPath = fullfile(tempdir, 'glass_bath_fea_test');
            if ~exist(testCase.TestDataPath, 'dir')
                mkdir(testCase.TestDataPath);
            end

            % Create mock material data
            testCase.Materials = struct();
            testCase.Materials.base_conductivity = 1.0;
            testCase.Materials.activation_energy = 80000;
            testCase.Materials.reference_temp = 1473.15;
            testCase.Materials.composition_factor = 1.0;
            testCase.Materials.gas_constant = 8.314;
            testCase.Materials.metal_conductivity = 10000.0;
            testCase.Materials.fulcher_A = -2.0;
            testCase.Materials.fulcher_B = 4500.0;
            testCase.Materials.fulcher_T0 = 250.0;
            testCase.Materials.operating_temperature = 1350;

            % Save mock material data
            matFile = fullfile(testCase.TestDataPath, 'test_materials.mat');
            save(matFile, '-struct', 'testCase', 'Materials', '-v7.3');
        end
    end

    methods (TestClassTeardown)
        function cleanupTestData(testCase)
            % Remove test directory
            if exist(testCase.TestDataPath, 'dir')
                rmdir(testCase.TestDataPath, 's');
            end
        end
    end

    methods (Test)
        %% Material Property Tests
        function testGlassConductivityPositive(testCase)
            % Test that glass conductivity is always positive
            location = struct('x', [0], 'y', [0], 'z', [0.1]);
            state = struct('u', []);

            sigma = glassConductivity(location, state, testCase.Materials);

            testCase.verifyGreaterThan(sigma, 0, ...
                'Glass conductivity must be positive');
        end

        function testGlassConductivityArrhenius(testCase)
            % Test that conductivity increases with temperature
            location = struct('x', [0], 'y', [0], 'z', [0.1]);

            % Low temperature
            materials_low = testCase.Materials;
            materials_low.operating_temperature = 1200;
            state_low = struct('u', []);
            sigma_low = glassConductivity(location, state_low, materials_low);

            % High temperature
            materials_high = testCase.Materials;
            materials_high.operating_temperature = 1400;
            state_high = struct('u', []);
            sigma_high = glassConductivity(location, state_high, materials_high);

            testCase.verifyGreaterThan(sigma_high, sigma_low, ...
                'Conductivity should increase with temperature');
        end

        function testMetalConductivityHigher(testCase)
            % Test that metal conductivity is much higher than glass
            location = struct('x', [0], 'y', [0], 'z', [0.1]);
            state = struct('u', []);

            sigma_glass = glassConductivity(location, state, testCase.Materials);
            sigma_metal = testCase.Materials.metal_conductivity;

            testCase.verifyGreaterThan(sigma_metal, 100 * sigma_glass, ...
                'Metal conductivity should be >> glass conductivity');
        end

        %% Load Material Data Tests
        function testLoadMaterialData(testCase)
            % Test loading material data from file
            matFile = fullfile(testCase.TestDataPath, 'test_materials.mat');

            materials = loadMaterialData(matFile);

            testCase.verifyClass(materials, 'struct', ...
                'Should return a structure');
            testCase.verifyTrue(isfield(materials, 'base_conductivity'), ...
                'Should have base_conductivity field');
            testCase.verifyTrue(isfield(materials, 'activation_energy'), ...
                'Should have activation_energy field');
        end

        function testLoadMaterialDataValues(testCase)
            % Test that loaded values match expected
            matFile = fullfile(testCase.TestDataPath, 'test_materials.mat');

            materials = loadMaterialData(matFile);

            testCase.verifyEqual(materials.base_conductivity, 1.0, ...
                'Base conductivity should match');
            testCase.verifyEqual(materials.activation_energy, 80000, ...
                'Activation energy should match');
        end

        %% Viscosity Tests
        function testViscosityDecreases(testCase)
            % Test that viscosity decreases with temperature (Fulcher eq)
            A = testCase.Materials.fulcher_A;
            B = testCase.Materials.fulcher_B;
            T0 = testCase.Materials.fulcher_T0;

            T_low = 1100 + 273.15;  % K
            T_high = 1400 + 273.15;  % K

            eta_low = 10^(A + B/(T_low - T0));
            eta_high = 10^(A + B/(T_high - T0));

            testCase.verifyGreaterThan(eta_low, eta_high, ...
                'Viscosity should decrease with temperature');
        end

        function testViscosityPositive(testCase)
            % Test that viscosity is always positive
            A = testCase.Materials.fulcher_A;
            B = testCase.Materials.fulcher_B;
            T0 = testCase.Materials.fulcher_T0;

            T_K = 1350 + 273.15;  % K
            eta = 10^(A + B/(T_K - T0));

            testCase.verifyGreaterThan(eta, 0, ...
                'Viscosity must be positive');
        end

        %% Integration Tests
        function testConductivityAtReferenceTemp(testCase)
            % Test conductivity at reference temperature
            location = struct('x', [0], 'y', [0], 'z', [0.1]);

            % Set temperature to reference (1200°C)
            materials = testCase.Materials;
            materials.operating_temperature = 1200;  % Reference is 1473.15 K
            state = struct('u', []);

            sigma = glassConductivity(location, state, materials);

            % At reference temp, should be close to base * composition
            expected = materials.base_conductivity * materials.composition_factor;
            testCase.verifyEqual(sigma, expected, 'RelTol', 0.01, ...
                'At reference temp, sigma should equal base * comp_factor');
        end
    end
end
