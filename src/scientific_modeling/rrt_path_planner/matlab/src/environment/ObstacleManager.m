classdef ObstacleManager < handle
    %OBSTACLEMANAGER Centralized obstacle handling and preset generation.

    properties
        bounds (1, 6) double
        obstacles (:, 8) double = zeros(0, 8)
    end

    methods
        function obj = ObstacleManager(bounds)
            arguments
                bounds (1, 6) double
            end
            obj.bounds = bounds;
        end

        function obstacles = generatePreset(obj, presetName)
            arguments
                obj
                presetName {mustBeTextScalar}
            end
            obj.obstacles = AsteroidFieldPresets(presetName, obj.bounds);
            obstacles = obj.obstacles;
        end

        function obstacles = generateCustom(obj, obstacleCount, minSize, maxSize, distribution)
            arguments
                obj
                obstacleCount (1, 1) double {mustBePositive} = 30
                minSize (1, 1) double {mustBePositive} = 0.03
                maxSize (1, 1) double {mustBePositive} = 0.08
                distribution {mustBeTextScalar} = "random"
            end

            if minSize > maxSize
                error('ObstacleManager:SizeRange', 'Minimum size must not exceed maximum size.');
            end

            obj.obstacles = obj.createRandomObstacles(obstacleCount, minSize, maxSize, distribution);
            obstacles = obj.obstacles;
        end

        function obstacles = getObstacles(obj)
            obstacles = obj.obstacles;
        end

        function setObstacles(obj, obstacles)
            arguments
                obj
                obstacles (:, 8) double
            end
            obj.obstacles = obstacles;
        end
    end

    methods (Access = private)
        function obstacles = createRandomObstacles(obj, obstacleCount, minSize, maxSize, distribution)
            obstacles = zeros(obstacleCount, 8);
            center = [(obj.bounds(1) + obj.bounds(2)) / 2, (obj.bounds(3) + obj.bounds(4)) / 2, (obj.bounds(5) + obj.bounds(6)) / 2];
            for idx = 1:obstacleCount
                switch lower(string(distribution))
                    case "clustered"
                        x = (obj.bounds(2) - obj.bounds(1)) * rand() + obj.bounds(1);
                        y = (obj.bounds(4) - obj.bounds(3)) * rand() + obj.bounds(3);
                        z = center(3) + 0.1 * (obj.bounds(6) - obj.bounds(5)) * randn();
                    case "layered"
                        x = (obj.bounds(2) - obj.bounds(1)) * rand() + obj.bounds(1);
                        y = center(2) + 0.1 * (obj.bounds(4) - obj.bounds(3)) * randn();
                        z = (obj.bounds(6) - obj.bounds(5)) * rand() + obj.bounds(5);
                    case "ring"
                        angle = rand() * 2 * pi;
                        radius = 0.4 * min(obj.bounds(2) - obj.bounds(1), obj.bounds(4) - obj.bounds(3));
                        x = center(1) + radius * cos(angle);
                        y = center(2) + radius * sin(angle);
                        z = center(3) + 0.05 * (obj.bounds(6) - obj.bounds(5)) * randn();
                    otherwise
                        x = (obj.bounds(2) - obj.bounds(1)) * rand() + obj.bounds(1);
                        y = (obj.bounds(4) - obj.bounds(3)) * rand() + obj.bounds(3);
                        z = (obj.bounds(6) - obj.bounds(5)) * rand() + obj.bounds(5);
                end

                sizeVal = minSize + (maxSize - minSize) * rand();
                color = [0.5 + 0.5 * rand(), 0.3 + 0.7 * rand(), 0.3 + 0.7 * rand()];
                obstacles(idx, :) = [round(rand()), x, y, z, sizeVal, color];
            end
        end
    end
end
