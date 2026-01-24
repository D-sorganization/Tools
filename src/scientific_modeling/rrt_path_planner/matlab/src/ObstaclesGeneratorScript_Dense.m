% ObstaclesGeneratorScript_Dense.m
% Creates a dense field with moving big central sphere and rotating cubes

disp('--- Generating Obstacles ---');

rng('shuffle'); % Random seed

% Settings
num_obstacles = 74; % 74 random obstacles
bounds = [-1.0, 1.0, -0.6, 0.6, -0.3, 0.3];

obstacles = zeros(num_obstacles+1, 8); % +1 for large sphere
velocities = zeros(num_obstacles+1, 3);
angular_velocities = zeros(num_obstacles+1, 3); % Only used for cubes

for i = 1:num_obstacles
    type = randi([0 1]); % 0 = sphere, 1 = cube

    if type == 0
        size_val = 0.03 + 0.02*rand(); % Small spheres
    else
        size_val = 0.04 + 0.04*rand(); % Small cubes
    end

    margin = size_val/2 + 0.01;

    x = (bounds(1) + margin) + (bounds(2) - bounds(1) - 2*margin)*rand();
    y = (bounds(3) + margin) + (bounds(4) - bounds(3) - 2*margin)*rand();
    z = (bounds(5) + margin) + (bounds(6) - bounds(5) - 2*margin)*rand();

    if type == 0
        base_color = [0.6 0.6 0.6] + 0.2*(rand(1,3)-0.5);
    else
        base_color = [0.5 0.25 0] + 0.2*(rand(1,3)-0.5);
    end
    base_color = min(max(base_color,0),1);

    velocities(i,:) = 0.05*(rand(1,3)-0.5);

    if type == 1 % Only cubes rotate
        angular_velocities(i,:) = deg2rad(10*(rand(1,3)-0.5)); % ~10 deg/sec random
    end

    obstacles(i,:) = [type, x, y, z, size_val, base_color];
end

% Add large central sphere
central_sphere_type = 0;
central_sphere_size = 0.15;
central_sphere_color = [0.7 0.7 0.7];
central_sphere_position = [0, 0, 0];

obstacles(end,:) = [central_sphere_type, ...
                    central_sphere_position, ...
                    central_sphere_size, ...
                    central_sphere_color];

velocities(end,:) = [0.005, 0, 0]; % Very slow drift
angular_velocities(end,:) = [0, 0, 0]; % No spin for central sphere

% Save files
writematrix(obstacles, 'obstacles3D.csv');
save('velocities.mat', 'velocities');
save('angular_velocities.mat', 'angular_velocities');

disp('✅ Obstacles, velocities, and angular velocities generated.');
