% main_static.m
% Static scene: Falcon path planning with RRT

clc;
clear;
close all;

disp('--- Static Falcon RRT Planner ---');

% Load Obstacles (fixed positions)
obstacles = readmatrix('obstacles3D.csv');

% Define World Bounds
bounds = [-1.0, 1.0, -0.6, 0.6, -0.3, 0.3];

% Define Start and Goal
start = [-0.9, 0, 0];
goal = [0.9, 0, 0];

% Run RRT
[nodes, path] = RRT(start, goal, obstacles, bounds);

if isempty(path)
    error('❌ No valid path found.');
end

% Plot Static Scene
plotRRT_Static(obstacles, path, nodes, start, goal, bounds);

disp('✅ Static scene generated.');
