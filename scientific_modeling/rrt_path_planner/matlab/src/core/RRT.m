function [nodes, path] = RRT(start, goal, obstacles, bounds, parameters)
% RRT.m
% Builds a Rapidly-Exploring Random Tree (RRT) with goal biasing
% Outputs:
%   nodes: [x y z parent_index]
%   path: final path from start to goal (Nx3)
if nargin < 5
    parameters = struct();
end

defaults = struct('maxNodes', 4000, 'stepSize', 0.06, 'goalRadius', 0.12, 'goalBias', 0.2);
paramFields = fieldnames(defaults);
for idx = 1:numel(paramFields)
    name = paramFields{idx};
    if ~isfield(parameters, name)
        parameters.(name) = defaults.(name);
    end
end

max_nodes = parameters.maxNodes;
step_size = parameters.stepSize;
goal_radius = parameters.goalRadius;
goal_bias = parameters.goalBias; % 20% chance to aim directly at goal

% Pre-allocate nodes array for performance (avoid O(n²) growth)
nodes = zeros(max_nodes, 4);
nodes(1, :) = [start 0]; % Start node [x y z parent=0]
node_count = 1;

for iter = 1:max_nodes
    % --- Goal-biased random sampling ---
    if rand() < goal_bias
        rand_point = goal;
    else
        rand_point = [
            (bounds(2)-bounds(1))*rand() + bounds(1),
            (bounds(4)-bounds(3))*rand() + bounds(3),
            (bounds(6)-bounds(5))*rand() + bounds(5)
        ];
    end
    rand_point = rand_point(:)'; % <-- Force [1x3] row vector

    % --- Find nearest node ---
    diffs = nodes(1:node_count,1:3) - rand_point;
    dists = vecnorm(diffs,2,2);

    if isempty(dists) || all(isnan(dists))
        continue; % Skip bad sample
    end

    [~, nearest_idx] = min(dists);

    if nearest_idx > node_count || nearest_idx < 1
        continue; % Safety check
    end

    nearest_node = nodes(nearest_idx,1:3);

    % --- Step towards sample safely ---
    direction = rand_point - nearest_node;
    d = norm(direction);
    if d < 1e-6 || isnan(d)
        continue; % Skip zero or NaN directions
    end
    direction = direction / d;
    new_node_pos = nearest_node + step_size * direction;

    % --- Validate new_node_pos ---
    if any(isnan(new_node_pos) | (numel(new_node_pos) ~= 3))
        continue; % Skip invalid nodes
    end

    % --- Check collision ---
    if collisionCheck(new_node_pos, obstacles)
        continue;
    end

    % --- Add new node ---
    node_count = node_count + 1;
    nodes(node_count, :) = [new_node_pos, nearest_idx]; % [x y z parent]

    % --- Check if goal reached ---
    if norm(new_node_pos - goal) < goal_radius
        disp('✅ Goal Reached!');
        break;
    end
end

% Trim unused pre-allocated space
nodes = nodes(1:node_count, :);

% --- Build Final Path (Backtrack from goal) ---
% Pre-allocate path_indices array (worst case: all nodes in path)
path_indices = zeros(1, node_count);
path_length = 1;
path_indices(path_length) = node_count;
current_idx = node_count;

while nodes(current_idx,4) ~= 0
    current_idx = nodes(current_idx,4);
    path_length = path_length + 1;
    path_indices(path_length) = current_idx;
end

% Trim to actual path length and reverse to get start->goal order
path_indices = fliplr(path_indices(1:path_length));
path = nodes(path_indices, 1:3);

if iter == max_nodes
    disp('⚠️ Warning: Max nodes reached, goal NOT found.');
end

end
