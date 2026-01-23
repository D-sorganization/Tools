% main_improved.m
% Improved Star Wars RRT Path Planner with all features
% Features:
% - Proper STL loading and ship orientation
% - Pursuit mode with intelligent AI
% - Multiple camera views
% - GUI interface
% - Dynamic obstacle avoidance

clc;
clear;
close all;

fprintf('🚀 Star Wars RRT Path Planner - Improved Version\n');
fprintf('================================================\n\n');

% Check if GUI is available
try
    % Try to launch GUI
    starWarsPathPlannerGUI();
    fprintf('✅ GUI launched successfully!\n');
    fprintf('Use the GUI controls to plan paths and create pursuit scenarios.\n\n');
catch ME
    fprintf('⚠️ GUI not available, running command-line version...\n');
    fprintf('Error: %s\n\n', ME.message);
    
    % Fall back to command-line version
    runCommandLineVersion();
end

end

function runCommandLineVersion()
% Command-line version for when GUI is not available

fprintf('Running Command-Line Version\n');
fprintf('============================\n\n');

% Load obstacles
try
    obstacles = readmatrix('obstacles3D.csv');
    fprintf('✅ Loaded obstacles from obstacles3D.csv\n');
catch
    fprintf('⚠️ Could not load obstacles3D.csv, generating random obstacles...\n');
    obstacles = generateRandomObstacles([-1.0, 1.0, -0.6, 0.6, -0.3, 0.3], 30);
end

bounds = [-1.0, 1.0, -0.6, 0.6, -0.3, 0.3];

% Ask user for mode
fprintf('\nSelect mode:\n');
fprintf('1. Single ship navigation\n');
fprintf('2. Pursuit mode\n');
mode = input('Enter choice (1 or 2): ');

if mode == 1
    % Single ship navigation
    fprintf('\n--- Single Ship Navigation ---\n');
    
    start = [-0.8, 0, 0];
    goal = [0.8, 0, 0];
    
    fprintf('Planning path from [%.1f, %.1f, %.1f] to [%.1f, %.1f, %.1f]...\n', ...
            start(1), start(2), start(3), goal(1), goal(2), goal(3));
    
    [nodes, path] = RRT(start, goal, obstacles, bounds);
    
    if isempty(path)
        fprintf('❌ No valid path found.\n');
        return;
    end
    
    fprintf('✅ Path planned successfully! (%d waypoints)\n', size(path, 1));
    
    % Ask for visualization type
    fprintf('\nSelect visualization:\n');
    fprintf('1. Static view\n');
    fprintf('2. Cinematic animation\n');
    viz_type = input('Enter choice (1 or 2): ');
    
    if viz_type == 1
        plotRRT_Static(obstacles, path, nodes, start, goal, bounds);
    else
        plotRRT_Moving_Improved(obstacles, path, nodes, start, goal, bounds);
    end
    
else
    % Pursuit mode
    fprintf('\n--- Pursuit Mode ---\n');
    
    pursuer_start = [-0.8, -0.3, 0];
    target_start = [-0.8, 0.3, 0];
    
    fprintf('Starting pursuit scenario...\n');
    fprintf('Pursuer: [%.1f, %.1f, %.1f]\n', pursuer_start(1), pursuer_start(2), pursuer_start(3));
    fprintf('Target: [%.1f, %.1f, %.1f]\n', target_start(1), target_start(2), target_start(3));
    
    [pursuer_path, target_path] = pursuitSystem(pursuer_start, target_start, obstacles, bounds, 150);
    
    fprintf('✅ Pursuit scenario completed!\n');
    fprintf('Pursuer path: %d waypoints\n', size(pursuer_path, 1));
    fprintf('Target path: %d waypoints\n', size(target_path, 1));
    
    % Visualize pursuit
    plotPursuit_Improved(obstacles, pursuer_path, target_path, bounds);
end

fprintf('\n🎉 Mission accomplished!\n');

end

function obstacles = generateRandomObstacles(bounds, num_obstacles)
% Generate random obstacles for testing
obstacles = zeros(num_obstacles, 8);

for i = 1:num_obstacles
    % Random position
    x = (bounds(2)-bounds(1))*rand() + bounds(1);
    y = (bounds(4)-bounds(3))*rand() + bounds(3);
    z = (bounds(6)-bounds(5))*rand() + bounds(5);
    
    % Random type (0=sphere, 1=cube)
    type = round(rand());
    
    % Random size
    size_val = 0.02 + 0.08*rand();
    
    % Random color
    color = [0.5 + 0.5*rand(), 0.3 + 0.7*rand(), 0.3 + 0.7*rand()];
    
    obstacles(i, :) = [type, x, y, z, size_val, color];
end

end 