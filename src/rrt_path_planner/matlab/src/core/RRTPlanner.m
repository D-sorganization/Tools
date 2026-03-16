classdef RRTPlanner < handle
    %RRTPLANNER Unified interface for static, dynamic, and optimal RRT variants.
    %   This class wraps existing planners so the GUI can switch algorithms
    %   without duplicating configuration logic.

    properties
        algorithmType char {mustBeMember(algorithmType, {'static', 'dynamic', 'rrt_star'})} = 'static'
        collisionMode char {mustBeMember(collisionMode, {'point', 'edge'})} = 'point'
        parameters struct = struct(...
            'maxNodes', 4000, ...
            'stepSize', 0.06, ...
            'goalRadius', 0.12, ...
            'goalBias', 0.20)
    end

    methods
        function obj = RRTPlanner(parameters)
            if nargin > 0
                obj.parameters = obj.mergeParameters(parameters);
            end
        end

        function setAlgorithm(obj, algorithmType)
            arguments
                obj
                algorithmType char {mustBeMember(algorithmType, {'static', 'dynamic', 'rrt_star'})}
            end
            obj.algorithmType = algorithmType;
        end

        function setCollisionMode(obj, collisionMode)
            arguments
                obj
                collisionMode char {mustBeMember(collisionMode, {'point', 'edge'})}
            end
            obj.collisionMode = collisionMode;
        end

        function setParameters(obj, parameters)
            obj.parameters = obj.mergeParameters(parameters);
        end

        function [nodes, path] = plan(obj, start, goal, obstacles, bounds)
            %PLAN Execute the selected RRT variant.
            arguments
                obj
                start (1, 3) double
                goal (1, 3) double
                obstacles (:, 8) double
                bounds (1, 6) double
            end

            plannerHandle = obj.selectPlanner();
            [nodes, path] = obj.callPlanner(plannerHandle, start, goal, obstacles, bounds, obj.parameters);

            if strcmp(obj.collisionMode, 'edge') && ~isempty(path)
                % Re-check the final path with the edge-based collision checker
                % used by the static implementation to improve robustness.
                if exist('collisionCheckEdge', 'file') == 2
                    isValid = collisionCheckEdge(path, obstacles);
                    if ~isValid
                        nodes = [];
                        path = [];
                    end
                end
            end
        end
    end

    methods (Access = private)
        function parameters = mergeParameters(obj, parameters)
            arguments
                obj
                parameters struct
            end

            parameters = obj.parametersWithDefaults(parameters);
        end

        function parameters = parametersWithDefaults(~, parameters)
            defaults = struct('maxNodes', 4000, 'stepSize', 0.06, 'goalRadius', 0.12, 'goalBias', 0.20);
            defaultsFields = fieldnames(defaults);
            for idx = 1:numel(defaultsFields)
                name = defaultsFields{idx};
                if ~isfield(parameters, name)
                    parameters.(name) = defaults.(name);
                end
            end
        end

        function plannerHandle = selectPlanner(obj)
            switch obj.algorithmType
                case 'static'
                    plannerHandle = obj.resolvePlannerHandle('RRT');
                case 'dynamic'
                    plannerHandle = obj.resolvePlannerHandle('RRTDynamic');
                case 'rrt_star'
                    plannerHandle = obj.resolvePlannerHandle('RRTStar');
                otherwise
                    plannerHandle = obj.resolvePlannerHandle('RRT');
            end
        end

        function plannerHandle = resolvePlannerHandle(~, functionName)
            if exist(functionName, 'file') == 2
                plannerHandle = str2func(functionName);
            else
                plannerHandle = @RRT;
            end
        end

        function [nodes, path] = callPlanner(~, plannerHandle, start, goal, obstacles, bounds, parameters)
            try
                [nodes, path] = plannerHandle(start, goal, obstacles, bounds, parameters);
                return
            catch plannerError
                if strcmp(plannerError.identifier, 'MATLAB:TooManyInputs') || contains(plannerError.message, 'Too many input arguments')
                    [nodes, path] = plannerHandle(start, goal, obstacles, bounds);
                else
                    rethrow(plannerError);
                end
            end
        end
    end
end
