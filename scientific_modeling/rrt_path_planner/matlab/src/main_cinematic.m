% main_cinematic.m

clc;
clear;
close all;

disp('--- Running Cinematic Falcon Planner ---');

obstacles = readmatrix('obstacles3D.csv');
bounds = [-1.0, 1.0, -0.6, 0.6, -0.3, 0.3];
start = [-0.9, 0, 0];
goal = [0.9, 0, 0];

[nodes, path] = RRT(start, goal, obstacles, bounds);

if isempty(path)
    error('❌ No valid path found.');
end

plotRRT_Moving(obstacles, path, nodes, start, goal, bounds);
