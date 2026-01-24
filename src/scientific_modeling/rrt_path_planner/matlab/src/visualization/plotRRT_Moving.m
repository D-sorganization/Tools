function plotRRT_Moving(obstacles, path, nodes, start, goal, bounds)
% plotRRT_Moving.m
% Falcon cinematic: disc-flat, true nose-forward

%% Setup Visible and Outer Bounds
visible_bounds = bounds;
expand = 0.2;
outer_bounds = [
    bounds(1)*(1+expand), bounds(2)*(1+expand);
    bounds(3)*(1+expand), bounds(4)*(1+expand);
    bounds(5)*(1+expand), bounds(6)*(1+expand)
];
outer_bounds = reshape(outer_bounds',1,6);

fig = figure('Color','k', 'Units','pixels', 'Position', [100 100 1600 900]);
hold on;
axis off;
set(gca,'DataAspectRatio',[1 1 1]);
set(gca, 'Units', 'normalized', 'Position', [0 0 1 1]);
set(gca, 'Color', 'k');
xlim(visible_bounds(1:2));
ylim(visible_bounds(3:4));
zlim(visible_bounds(5:6));
view(3);
campos([0, -5, 2]);
camtarget([0, 0, 0]);
camup([0, 0, 1]);
camva(30);
camzoom(1.8);

%% --- C-3PO Quote (Gold Crawl) ---
quote_str = 'Sir, the possibility of successfully navigating an asteroid field is approximately 3,720 to 1!';
quote_text = text(0.5, 0.3, quote_str, ...
    'Color', [1.0, 0.84, 0.0], 'FontSize', 30, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Units', 'normalized');

for k = 1:300
    quote_text.Position(2) = quote_text.Position(2) + 0.0015;
    quote_text.FontSize = quote_text.FontSize * 0.985;
    pause(0.01);
end

delete(quote_text);

disp('✅ Quote finished. Starting Falcon animation.');

%% --- Load Falcon Model and Apply TRIPLE Pre-Rotation ---
fv = stlread('falcon_clean_fixed.stl');

% Triple pre-rotation: -90 Z, +90 X, -90 Y
Rz = [cosd(-90) -sind(-90) 0; sind(-90) cosd(-90) 0; 0 0 1];
Rx = [1 0 0; 0 cosd(90) -sind(90); 0 sind(90) cosd(90)];
Ry = [cosd(-90) 0 sind(-90); 0 1 0; -sind(-90) 0 cosd(-90)];

pre_rotation = Ry * Rx * Rz;
original_vertices = (pre_rotation * fv.Points')';

scale_factor = 0.1;

falcon_plot = patch('Faces',fv.ConnectivityList,'Vertices',original_vertices*scale_factor, ...
    'FaceColor',[0.8 0.8 0.8],'EdgeColor','none');

live_trace = plot3(NaN, NaN, NaN, 'LineWidth', 2, 'Color', [1 1 0]);

% Starfield
num_stars = 500;
scatter3(randn(num_stars,1)*1.5, randn(num_stars,1)*1.5, randn(num_stars,1)*1.5, 5, [0.8 0.8 0.8], 'filled');

% Load Velocities
load velocities.mat velocities;
load angular_velocities.mat angular_velocities;

obstacle_patches = gobjects(size(obstacles,1),1);
obstacle_types = strings(size(obstacles,1),1);

[theta,phi] = meshgrid(linspace(0,2*pi,20),linspace(0,pi,20));

for i = 1:size(obstacles,1)
    type = obstacles(i,1);
    x = obstacles(i,2);
    y = obstacles(i,3);
    z = obstacles(i,4);
    size_val = obstacles(i,5);
    color = obstacles(i,6:8);

    if type == 0
        r = size_val;
        xc = x + r*sin(phi).*cos(theta);
        yc = y + r*sin(phi).*sin(theta);
        zc = z + r*cos(phi);
        obstacle_patches(i) = surf(xc, yc, zc, 'FaceColor', color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        obstacle_types(i) = "sphere";
    else
        [X,Y,Z] = ndgrid([-1 1]*size_val/2, [-1 1]*size_val/2, [-1 1]*size_val/2);
        vertices = [X(:), Y(:), Z(:)];
        theta_rand = 2*pi*rand(); phi_rand = 2*pi*rand(); psi_rand = 2*pi*rand();
        R = eul2rotm([psi_rand, phi_rand, theta_rand]);
        vertices = (R * vertices')';
        faces = [1 2 4 3; 5 6 8 7; 1 2 6 5; 3 4 8 7; 1 3 7 5; 2 4 8 6];
        patch_obj = patch('Vertices',vertices + [x y z], 'Faces',faces, 'FaceColor',color, 'FaceAlpha',0.2, 'EdgeColor','none');
        obstacle_patches(i) = patch_obj;
        obstacle_types(i) = "cube";
    end
end

%% --- Falcon Smooth Path ---
dt = 0.01;
t_original = linspace(0, 1, size(path,1));
t_smooth = linspace(0, 1, 1000);
path_smooth = interp1(t_original, path, t_smooth, 'spline');

v = VideoWriter('falcon_flight.mp4','MPEG-4');
open(v);

for frame = 1:size(path_smooth,1)-1
    p1 = path_smooth(frame,:);
    p2 = path_smooth(frame+1,:);

    % --- Falcon Proper Orientation (True Nose-Forward) ---
    forward = (p2 - p1);
    forward = forward / norm(forward);

    world_up = [0 0 1];

    right = cross(world_up, forward);
    if norm(right) < 1e-6
        right = [1 0 0];
    else
        right = right / norm(right);
    end

    up = cross(forward, right);

    % Build full rotation matrix (NO artificial roll yet)
    R_falcon = [forward' right' up'];

    rotated_vertices = (original_vertices * scale_factor) * R_falcon;
    translated_vertices = rotated_vertices + p1;
    falcon_plot.Vertices = translated_vertices;

    % Update Trace
    live_trace.XData(end+1) = p1(1);
    live_trace.YData(end+1) = p1(2);
    live_trace.ZData(end+1) = p1(3);

    % Obstacles Movement [unchanged wrapping logic]

    drawnow;
    pause(0.001);
    frame_captured = getframe(gcf);
    writeVideo(v, frame_captured);
end

close(v);
hold off;
end
