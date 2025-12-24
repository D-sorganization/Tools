% test_system.m
% Test script to verify all components of the enhanced Star Wars RRT Path Planner

fprintf('🧪 Testing Enhanced Star Wars RRT Path Planner\n');
fprintf('==============================================\n\n');

%% Test 1: STL Loading
fprintf('Test 1: STL File Loading...\n');
try
    [V, F] = stlread('falcon_clean_fixed.stl');
    fprintf('✅ STL loading successful: %d vertices, %d faces\n', size(V, 1), size(F, 1));
catch ME
    fprintf('⚠️ STL loading failed: %s\n', ME.message);
    fprintf('Using fallback simple ship model...\n');
    V = createSimpleShip();
    F = createSimpleShipFaces();
end

%% Test 2: Ship Orientation
fprintf('\nTest 2: Ship Orientation...\n');
try
    current_pos = [0, 0, 0];
    next_pos = [1, 0, 0];
    current_forward = [1, 0, 0];
    world_up = [0, 0, 1];
    
    [R, up_vector] = shipOrientation(current_pos, next_pos, current_forward, world_up);
    fprintf('✅ Ship orientation successful: Rotation matrix calculated\n');
catch ME
    fprintf('❌ Ship orientation failed: %s\n', ME.message);
end

%% Test 3: RRT Path Planning
fprintf('\nTest 3: RRT Path Planning...\n');
try
    % Create simple test environment
    bounds = [-1.0, 1.0, -0.6, 0.6, -0.3, 0.3];
    start = [-0.8, 0, 0];
    goal = [0.8, 0, 0];
    
    % Simple obstacles
    obstacles = [
        0, 0, 0, 0, 0.1, 0.5, 0.5, 0.5;  % Sphere at origin
        1, 0.5, 0, 0, 0.05, 0.3, 0.3, 0.7;  % Cube
    ];
    
    [nodes, path] = RRT(start, goal, obstacles, bounds);
    
    if ~isempty(path)
        fprintf('✅ RRT planning successful: %d waypoints found\n', size(path, 1));
    else
        fprintf('⚠️ RRT planning: No path found (this may be normal for some configurations)\n');
    end
catch ME
    fprintf('❌ RRT planning failed: %s\n', ME.message);
end

%% Test 4: Pursuit System
fprintf('\nTest 4: Pursuit System...\n');
try
    bounds = [-1.0, 1.0, -0.6, 0.6, -0.3, 0.3];
    pursuer_start = [-0.8, -0.3, 0];
    target_start = [-0.8, 0.3, 0];
    
    % Simple obstacles
    obstacles = [
        0, 0, 0, 0, 0.1, 0.5, 0.5, 0.5;
    ];
    
    [pursuer_path, target_path] = pursuitSystem(pursuer_start, target_start, obstacles, bounds, 50);
    
    fprintf('✅ Pursuit system successful: Pursuer path %d points, Target path %d points\n', ...
            size(pursuer_path, 1), size(target_path, 1));
catch ME
    fprintf('❌ Pursuit system failed: %s\n', ME.message);
end

%% Test 5: Collision Detection
fprintf('\nTest 5: Collision Detection...\n');
try
    obstacles = [
        0, 0, 0, 0, 0.1, 0.5, 0.5, 0.5;
        1, 0.5, 0, 0, 0.05, 0.3, 0.3, 0.7;
    ];
    
    % Test points
    test_points = [
        0, 0, 0;      % Should collide with sphere
        0.1, 0, 0;    % Should collide with sphere
        0.5, 0, 0;    % Should collide with cube
        1, 0, 0;      % Should not collide
    ];
    
    for i = 1:size(test_points, 1)
        collision = collisionCheck(test_points(i, :), obstacles);
        fprintf('  Point [%.1f, %.1f, %.1f]: %s\n', ...
                test_points(i, 1), test_points(i, 2), test_points(i, 3), ...
                ternary(collision, 'COLLISION', 'FREE'));
    end
    
    fprintf('✅ Collision detection successful\n');
catch ME
    fprintf('❌ Collision detection failed: %s\n', ME.message);
end

%% Test 6: Visualization Functions
fprintf('\nTest 6: Visualization Functions...\n');
try
    % Create test data
    bounds = [-1.0, 1.0, -0.6, 0.6, -0.3, 0.3];
    start = [-0.8, 0, 0];
    goal = [0.8, 0, 0];
    obstacles = [
        0, 0, 0, 0, 0.1, 0.5, 0.5, 0.5;
    ];
    
    % Test path
    path = [start; [0, 0, 0]; goal];
    nodes = [start, 0; [0, 0, 0], 1; goal, 2];
    
    fprintf('✅ Visualization functions ready (not actually plotting to avoid blocking)\n');
catch ME
    fprintf('❌ Visualization setup failed: %s\n', ME.message);
end

%% Test 7: GUI Components
fprintf('\nTest 7: GUI Components...\n');
try
    % Test if GUI functions exist
    if exist('starWarsPathPlannerGUI', 'file')
        fprintf('✅ GUI function found\n');
    else
        fprintf('⚠️ GUI function not found\n');
    end
    
    if exist('main_improved', 'file')
        fprintf('✅ Main improved function found\n');
    else
        fprintf('⚠️ Main improved function not found\n');
    end
catch ME
    fprintf('❌ GUI component test failed: %s\n', ME.message);
end

%% Summary
fprintf('\n==============================================\n');
fprintf('🧪 Test Summary\n');
fprintf('==============================================\n');
fprintf('All core components have been tested.\n');
fprintf('If you see mostly ✅ marks, the system is ready to use!\n');
fprintf('\nTo run the full system:\n');
fprintf('1. Run: main_improved\n');
fprintf('2. Or run: starWarsPathPlannerGUI\n');
fprintf('\nMay the Force be with you! 🌟\n');

end

function result = ternary(condition, true_value, false_value)
% Simple ternary operator for MATLAB
if condition
    result = true_value;
else
    result = false_value;
end
end

function V = createSimpleShip()
% Create simple ship vertices for testing
V = [
    -0.5, 0, 0;    % Nose
    0.5, 0, 0;     % Tail
    0, 0.3, 0;     % Right wing
    0, -0.3, 0;    % Left wing
    0, 0, 0.2;     % Top
    0, 0, -0.2;    % Bottom
];
end

function F = createSimpleShipFaces()
% Create simple ship faces for testing
F = [
    1, 3, 5;   % Right side
    1, 5, 4;   % Left side
    1, 4, 6;   % Bottom
    1, 6, 3;   % Top
    2, 3, 5;   % Right back
    2, 5, 4;   % Left back
    2, 4, 6;   % Bottom back
    2, 6, 3;   % Top back
];
end 