function plotRRT_Moving_Improved(obstacles, path, nodes, start, goal, bounds)
% PLOTRRT_MOVING_IMPROVED Enhanced cinematic visualization with proper ship orientation
% Features:
% - Proper STL loading and orientation
% - Nose-forward ship movement
% - Multiple camera views
% - Enhanced visual effects
% - Smooth path interpolation

%% Setup Figure and Environment
fig = figure('Color', 'k', 'Units', 'pixels', 'Position', [100, 100, 1600, 900]);
ax = axes('Parent', fig, 'Color', 'k', 'Position', [0, 0, 1, 1]);
hold on;
axis off;
set(gca, 'DataAspectRatio', [1, 1, 1]);
xlim(bounds(1:2));
ylim(bounds(3:4));
zlim(bounds(5:6));

% Set initial camera view
view(45, 30);
campos([2, -2, 1]);
camtarget([0, 0, 0]);
camup([0, 0, 1]);
camva(30);

%% Load and Prepare Ship Model
try
    [V, F] = stlread('falcon_clean_fixed.stl');
    fprintf('✅ STL model loaded successfully\n');
catch ME
    fprintf('⚠️ Could not load STL file: %s\n', ME.message);
    fprintf('Using simple ship model instead...\n');
    V = createSimpleShip();
    F = createSimpleShipFaces();
end

% Apply pre-rotation to align ship properly
% Rotate to make ship face forward (nose in +X direction)
Rz = [cosd(-90), -sind(-90), 0; sind(-90), cosd(-90), 0; 0, 0, 1];
Rx = [1, 0, 0; 0, cosd(90), -sind(90); 0, sind(90), cosd(90)];
Ry = [cosd(-90), 0, sind(-90); 0, 1, 0; -sind(-90), 0, cosd(-90)];

pre_rotation = Ry * Rx * Rz;
V = (pre_rotation * V')';

% Scale ship
scale_factor = 0.05;
V = V * scale_factor;

% Create ship patch
ship_patch = patch('Faces', F, 'Vertices', V, ...
                   'FaceColor', [0.8, 0.8, 0.8], 'EdgeColor', 'none');

%% Create Starfield
num_stars = 800;
star_positions = randn(num_stars, 3) * 3;
star_sizes = 2 + 8 * rand(num_stars, 1);
star_colors = [0.8 + 0.2*rand(num_stars, 1), ...
               0.8 + 0.2*rand(num_stars, 1), ...
               0.8 + 0.2*rand(num_stars, 1)];

scatter3(star_positions(:, 1), star_positions(:, 2), star_positions(:, 3), ...
         star_sizes, star_colors, 'filled');

%% Draw Obstacles
drawObstacles(obstacles);

%% Draw Start and Goal
plot3(start(1), start(2), start(3), 'go', 'MarkerSize', 15, 'MarkerFaceColor', 'g', 'LineWidth', 2);
plot3(goal(1), goal(2), goal(3), 'ro', 'MarkerSize', 15, 'MarkerFaceColor', 'r', 'LineWidth', 2);

%% Create Path Trace
trace_line = plot3(NaN, NaN, NaN, 'LineWidth', 3, 'Color', [1, 1, 0]);

%% Smooth Path Interpolation
if size(path, 1) > 2
    t_original = linspace(0, 1, size(path, 1));
    t_smooth = linspace(0, 1, 1000);
    path_smooth = interp1(t_original, path, t_smooth, 'spline');
else
    path_smooth = path;
end

%% Animation Loop
fprintf('🎬 Starting cinematic animation...\n');

% Initialize ship orientation
current_forward = [1, 0, 0];
world_up = [0, 0, 1];

for frame = 1:size(path_smooth, 1)-1
    current_pos = path_smooth(frame, :);
    next_pos = path_smooth(frame+1, :);
    
    % Calculate proper ship orientation
    [R, ~] = shipOrientation(current_pos, next_pos, current_forward, world_up);
    
    % Update current forward direction
    current_forward = (next_pos - current_pos) / norm(next_pos - current_pos);
    
    % Transform ship
    rotated_vertices = (V * R') + current_pos;
    ship_patch.Vertices = rotated_vertices;
    
    % Update trace
    trace_line.XData = [trace_line.XData, current_pos(1)];
    trace_line.YData = [trace_line.YData, current_pos(2)];
    trace_line.ZData = [trace_line.ZData, current_pos(3)];
    
    % Dynamic camera movement (optional)
    if mod(frame, 100) == 0
        % Change camera angle periodically
        angle = frame * 0.1;
        campos([2*cos(angle), 2*sin(angle), 1]);
        camtarget(current_pos);
    end
    
    drawnow;
    pause(0.01);
end

fprintf('✅ Animation complete!\n');

end

function drawObstacles(obstacles)
% Draw obstacles with enhanced visual effects
[theta, phi] = meshgrid(linspace(0, 2*pi, 20), linspace(0, pi, 20));

for i = 1:size(obstacles, 1)
    type = obstacles(i, 1);
    x = obstacles(i, 2);
    y = obstacles(i, 3);
    z = obstacles(i, 4);
    size_val = obstacles(i, 5);
    color = obstacles(i, 6:8);
    
    if type == 0  % Sphere
        r = size_val;
        xc = x + r*sin(phi).*cos(theta);
        yc = y + r*sin(phi).*sin(theta);
        zc = z + r*cos(phi);
        surf(xc, yc, zc, 'FaceColor', color, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
    else  % Cube
        [X, Y, Z] = ndgrid([-1 1]*size_val/2, [-1 1]*size_val/2, [-1 1]*size_val/2);
        vertices = [X(:), Y(:), Z(:)];
        
        % Add random rotation
        theta_rand = 2*pi*rand();
        phi_rand = 2*pi*rand();
        psi_rand = 2*pi*rand();
        R = eul2rotm([psi_rand, phi_rand, theta_rand]);
        vertices = (R * vertices')';
        
        faces = [1 2 4 3; 5 6 8 7; 1 2 6 5; 3 4 8 7; 1 3 7 5; 2 4 8 6];
        patch('Vertices', vertices + [x y z], 'Faces', faces, ...
              'FaceColor', color, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
    end
end

end

function V = createSimpleShip()
% Create simple ship vertices
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
% Create simple ship faces
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