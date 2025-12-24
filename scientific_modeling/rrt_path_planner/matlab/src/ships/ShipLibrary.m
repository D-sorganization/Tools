classdef ShipLibrary < handle
    %SHIPLIBRARY Load and cache STL ship models for the planner GUI.

    properties
        basePath char
        ships containers.Map
    end

    methods
        function obj = ShipLibrary(modelsPath)
            if nargin < 1 || strlength(modelsPath) == 0
                obj.basePath = obj.defaultModelPath();
            else
                obj.basePath = modelsPath;
            end
            obj.ships = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.loadAllShips();
        end

        function names = availableShips(obj)
            names = keys(obj.ships);
        end

        function ship = getShip(obj, shipName)
            if isKey(obj.ships, shipName)
                ship = obj.ships(shipName);
            else
                ship = obj.createFallbackShip(shipName);
                obj.ships(shipName) = ship;
            end
        end
    end

    methods (Access = private)
        function loadAllShips(obj)
            modelFiles = dir(fullfile(obj.basePath, '**', '*.stl'));
            for idx = 1:numel(modelFiles)
                filePath = fullfile(modelFiles(idx).folder, modelFiles(idx).name);
                [~, name, ~] = fileparts(filePath);
                try
                    [vertices, faces] = stlread(filePath);
                    shipStruct = struct('name', name, 'faces', faces, 'vertices', vertices, ...
                        'path', filePath);
                    obj.ships(name) = shipStruct;
                catch
                    shipStruct = obj.createFallbackShip(name);
                    obj.ships(name) = shipStruct;
                end
            end

            if isempty(obj.ships)
                defaultShip = obj.createFallbackShip('falcon_placeholder');
                obj.ships(defaultShip.name) = defaultShip;
            end
        end

        function ship = createFallbackShip(~, shipName)
            % Build a simple tetrahedron as a safe fallback model.
            vertices = [0 0 0; 0.05 -0.02 0; 0.05 0.02 0; 0 0 0.04];
            faces = [1 2 3; 1 2 4; 2 3 4; 3 1 4];
            ship = struct('name', shipName, 'faces', faces, 'vertices', vertices, 'path', 'generated');
        end

        function path = defaultModelPath(~)
            currentDir = fileparts(mfilename('fullpath'));
            % models directory lives alongside src/, so step up two levels
            path = fullfile(currentDir, '..', '..', 'models');
        end
    end
end
