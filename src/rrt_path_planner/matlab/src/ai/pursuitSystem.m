function [pursuer_path, target_path] = pursuitSystem(pursuer_start, target_start, obstacles, bounds, max_iterations)
% PURSUITSYSTEM Creates a pursuit scenario between two ships
% Inputs:
%   pursuer_start - starting position of pursuer [x,y,z]
%   target_start - starting position of target [x,y,z]
%   obstacles - obstacle matrix
%   bounds - environment bounds [xmin,xmax,ymin,ymax,zmin,zmax]
%   max_iterations - maximum number of pursuit iterations
% Outputs:
%   pursuer_path - path taken by pursuer
%   target_path - path taken by target

% Initialize paths
pursuer_path = pursuer_start;
target_path = target_start;

% Pursuit parameters
pursuer_speed = 0.02;  % Distance per iteration
target_speed = 0.015;  % Slightly slower than pursuer
evasion_radius = 0.1;  % Distance at which target starts evading
capture_radius = 0.05; % Distance for capture

% Target behavior: move toward a random goal, then change when pursued
target_goal = generateRandomGoal(bounds);
goal_change_counter = 0;

for iter = 1:max_iterations
    current_pursuer = pursuer_path(end, :);
    current_target = target_path(end, :);
    
    % Check for capture
    distance = norm(current_pursuer - current_target);
    if distance < capture_radius
        fprintf('🎯 Target captured at iteration %d!\n', iter);
        break;
    end
    
    % Target behavior: evade if pursued, otherwise move toward goal
    if distance < evasion_radius
        % Evasion mode: move away from pursuer
        evasion_direction = (current_target - current_pursuer) / norm(current_target - current_pursuer);
        target_step = current_target + target_speed * evasion_direction;
        
        % Ensure target stays within bounds
        target_step = constrainToBounds(target_step, bounds);
        
        % Check collision for target
        if ~collisionCheck(target_step, obstacles)
            target_path = [target_path; target_step];
        end
    else
        % Normal mode: move toward goal
        if norm(current_target - target_goal) < 0.1 || goal_change_counter > 50
            % Change goal
            target_goal = generateRandomGoal(bounds);
            goal_change_counter = 0;
        end
        
        goal_direction = (target_goal - current_target) / norm(target_goal - current_target);
        target_step = current_target + target_speed * goal_direction;
        
        % Ensure target stays within bounds
        target_step = constrainToBounds(target_step, bounds);
        
        % Check collision for target
        if ~collisionCheck(target_step, obstacles)
            target_path = [target_path; target_step];
        end
        
        goal_change_counter = goal_change_counter + 1;
    end
    
    % Pursuer behavior: plan path to current target position
    [~, pursuer_to_target] = RRT(current_pursuer, target_path(end, :), obstacles, bounds);
    
    if ~isempty(pursuer_to_target) && size(pursuer_to_target, 1) > 1
        % Move pursuer along planned path
        next_pursuer = pursuer_to_target(2, :);
        pursuer_path = [pursuer_path; next_pursuer];
    else
        % If no path found, move directly toward target
        direction = (target_path(end, :) - current_pursuer) / norm(target_path(end, :) - current_pursuer);
        pursuer_step = current_pursuer + pursuer_speed * direction;
        pursuer_step = constrainToBounds(pursuer_step, bounds);
        
        if ~collisionCheck(pursuer_step, obstacles)
            pursuer_path = [pursuer_path; pursuer_step];
        end
    end
end

end

function goal = generateRandomGoal(bounds)
% Generate a random goal within bounds
goal = [
    (bounds(2)-bounds(1))*rand() + bounds(1),
    (bounds(4)-bounds(3))*rand() + bounds(3),
    (bounds(6)-bounds(5))*rand() + bounds(5)
];
end

function pos = constrainToBounds(pos, bounds)
% Constrain position to bounds
pos(1) = max(bounds(1), min(bounds(2), pos(1)));
pos(2) = max(bounds(3), min(bounds(4), pos(2)));
pos(3) = max(bounds(5), min(bounds(6), pos(3)));
end 