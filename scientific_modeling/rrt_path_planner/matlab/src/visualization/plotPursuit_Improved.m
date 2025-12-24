function plotPursuit_Improved(obstacles, pursuer_path, target_path, bounds)
% PLOTPURSUIT_IMPROVED Enhanced pursuit visualization with two ships
% Features:
% - Two ships with different models/colors
% - Dynamic camera following
% - Pursuit-specific visual effects
% - Capture detection and celebration

%% Setup Figure and Environment
fig = figure('Color', 'k', 'Units', 'pixels', 'Position', [100, 100, 1600, 900]);
ax = axes('Parent', fig, 'Color', 'k', 'Position', [0, 0, 1, 1]);
hold on;
axis off;
set(gca, 'DataAspectRatio', [1, 1, 1]);
xlim(bounds(1:2));
ylim(bounds(3:4));
zlim(bounds(5:6));

% Set initial camera view for pursuit
view(45, 30);
campos([2, -2, 1]);
camtarget([0, 0, 0]);
camup([0, 0, 1]);
camva(30);

%% Load Ship Models
try
    [V1, F1] = stlread('falcon_clean_fixed.stl');
    [V2, F2] = stlread('falcon_clean_fixed.stl');  % Use same model for now
    fprintf('✅ STL models loaded successfully\n');
catch ME
    fprintf('⚠️ Could not load STL files: %s\n', ME.message);
    fprintf('Using simple ship models instead...\n');
    V1 = createSimpleShip();
    F1 = createSimpleShipFaces();
    V2 = createSimpleShip();
    F2 = createSimpleShipFaces();
end

% Apply pre-rotation to align ships properly
Rz = [cosd(-90), -sind(-90), 0; sind(-90), cosd(-90), 0; 0, 0, 1];
Rx = [1, 0, 0; 0, cosd(90), -sind(90); 0, sind(90), cosd(90)];
Ry = [cosd(-90), 0, sind(-90); 0, 1, 0; -sind(-90), 0, cosd(-90)];

pre_rotation = Ry * Rx * Rz;
V1 = (pre_rotation * V1')';
V2 = (pre_rotation * V2')';

% Scale ships
scale_factor = 0.05;
V1 = V1 * scale_factor;
V2 = V2 * scale_factor;

% Create ship patches with different colors
pursuer_patch = patch('Faces', F1, 'Vertices', V1, ...
                      'FaceColor', [0.8, 0.8, 0.8], 'EdgeColor', 'none');  % Light gray
target_patch = patch('Faces', F2, 'Vertices', V2, ...
                     'FaceColor', [0.6, 0.6, 0.6], 'EdgeColor', 'none');   % Dark gray

%% Create Enhanced Starfield
num_stars = 1000;
star_positions = randn(num_stars, 3) * 3;
star_sizes = 1 + 10 * rand(num_stars, 1);
star_colors = [0.7 + 0.3*rand(num_stars, 1), ...
               0.7 + 0.3*rand(num_stars, 1), ...
               0.7 + 0.3*rand(num_stars, 1)];

scatter3(star_positions(:, 1), star_positions(:, 2), star_positions(:, 3), ...
         star_sizes, star_colors, 'filled');

%% Draw Obstacles
drawObstacles(obstacles);

%% Create Path Traces
pursuer_trace = plot3(NaN, NaN, NaN, 'LineWidth', 3, 'Color', [0, 1, 1]);  % Cyan
target_trace = plot3(NaN, NaN, NaN, 'LineWidth', 3, 'Color', [1, 0.5, 0]); % Orange

%% Smooth Path Interpolation
if size(pursuer_path, 1) > 2
    t_original = linspace(0, 1, size(pursuer_path, 1));
    t_smooth = linspace(0, 1, 1000);
    pursuer_smooth = interp1(t_original, pursuer_path, t_smooth, 'spline');
else
    pursuer_smooth = pursuer_path;
end

if size(target_path, 1) > 2
    t_original = linspace(0, 1, size(target_path, 1));
    t_smooth = linspace(0, 1, 1000);
    target_smooth = interp1(t_original, target_path, t_smooth, 'spline');
else
    target_smooth = target_path;
end

%% Animation Loop
fprintf('🎬 Starting pursuit animation...\n');

% Initialize ship orientations
pursuer_forward = [1, 0, 0];
target_forward = [1, 0, 0];
world_up = [0, 0, 1];

% Determine animation length
max_frames = max(size(pursuer_smooth, 1), size(target_smooth, 1));

for frame = 1:max_frames-1
    % Update pursuer
    if frame <= size(pursuer_smooth, 1)
        pursuer_pos = pursuer_smooth(frame, :);
        if frame < size(pursuer_smooth, 1)
            pursuer_next = pursuer_smooth(frame+1, :);
            [R1, ~] = shipOrientation(pursuer_pos, pursuer_next, pursuer_forward, world_up);
            pursuer_forward = (pursuer_next - pursuer_pos) / norm(pursuer_next - pursuer_pos);
        else
            [R1, ~] = shipOrientation(pursuer_pos, pursuer_pos, pursuer_forward, world_up);
        end
        
        rotated_vertices1 = (V1 * R1') + pursuer_pos;
        pursuer_patch.Vertices = rotated_vertices1;
        
        % Update pursuer trace
        pursuer_trace.XData = [pursuer_trace.XData, pursuer_pos(1)];
        pursuer_trace.YData = [pursuer_trace.YData, pursuer_pos(2)];
        pursuer_trace.ZData = [pursuer_trace.ZData, pursuer_pos(3)];
    end
    
    % Update target
    if frame <= size(target_smooth, 1)
        target_pos = target_smooth(frame, :);
        if frame < size(target_smooth, 1)
            target_next = target_smooth(frame+1, :);
            [R2, ~] = shipOrientation(target_pos, target_next, target_forward, world_up);
            target_forward = (target_next - target_pos) / norm(target_next - target_pos);
        else
            [R2, ~] = shipOrientation(target_pos, target_pos, target_forward, world_up);
        end
        
        rotated_vertices2 = (V2 * R2') + target_pos;
        target_patch.Vertices = rotated_vertices2;
        
        % Update target trace
        target_trace.XData = [target_trace.XData, target_pos(1)];
        target_trace.YData = [target_trace.YData, target_pos(2)];
        target_trace.ZData = [target_trace.ZData, target_pos(3)];
    end
    
    % Check for capture
    if frame <= size(pursuer_smooth, 1) && frame <= size(target_smooth, 1)
        distance = norm(pursuer_smooth(frame, :) - target_smooth(frame, :));
        if distance < 0.05  % Capture radius
            % Capture effect
            target_patch.FaceColor = [1, 0, 0];  % Turn target red
            fprintf('🎯 Target captured at frame %d!\n', frame);
            
            % Add explosion effect
            explosion_pos = target_smooth(frame, :);
            scatter3(explosion_pos(1), explosion_pos(2), explosion_pos(3), ...
                    500, [1, 0.5, 0], 'filled', 'MarkerEdgeColor', [1, 1, 0], 'LineWidth', 2);
        end
    end
    
    % Dynamic camera following
    if frame <= size(pursuer_smooth, 1) && frame <= size(target_smooth, 1)
        % Camera follows midpoint between ships
        mid_point = (pursuer_smooth(frame, :) + target_smooth(frame, :)) / 2;
        camtarget(mid_point);
        
        % Camera position based on pursuit dynamics
        pursuit_direction = pursuer_smooth(frame, :) - target_smooth(frame, :);
        pursuit_direction = pursuit_direction / norm(pursuit_direction);
        
        camera_distance = 2;
        camera_height = 1;
        camera_pos = mid_point + camera_distance * pursuit_direction + [0, 0, camera_height];
        campos(camera_pos);
    end
    
    % Add laser effects (optional)
    if mod(frame, 50) == 0 && frame <= size(pursuer_smooth, 1) && frame <= size(target_smooth, 1)
        % Simulate laser shots
        laser_start = pursuer_smooth(frame, :);
        laser_end = target_smooth(frame, :) + randn(1, 3) * 0.1;  % Add some miss factor
        
        plot3([laser_start(1), laser_end(1)], ...
              [laser_start(2), laser_end(2)], ...
              [laser_start(3), laser_end(3)], ...
              'r-', 'LineWidth', 2, 'Color', [1, 0, 0]);
    end
    
    drawnow;
    pause(0.01);
end

fprintf('✅ Pursuit animation complete!\n');

% Final celebration if capture occurred
if exist('explosion_pos', 'var')
    fprintf('🎉 Mission accomplished! Target eliminated!\n');
    
    % Add victory text
    text(0, 0, 0.5, 'MISSION ACCOMPLISHED!', ...
         'Color', [1, 1, 0], 'FontSize', 24, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center');
end

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
        surf(xc, yc, zc, 'FaceColor', color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
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
              'FaceColor', color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
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