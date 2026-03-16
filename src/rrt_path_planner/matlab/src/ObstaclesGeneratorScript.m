% Define number of obstacles
numObstacles = 75;

% Initialize arrays
obstacles = zeros(numObstacles, 8); % [type, x, y, z, size, r, g, b]
velocities = zeros(numObstacles, 3); % [vx, vy, vz]

% Bounds
xrange = [-1.0, 1.0];
yrange = [-0.6, 0.6];
zrange = [-0.3, 0.3];

% Big grey sphere
obstacles(1,:) = [0, 0, 0, 0, 0.2, 0.68, 0.71, 0.73]; % type 0, center, large, grey
velocities(1,:) = [0.01, 0, 0]; % small velocity only along x

% Other obstacles
for i = 2:numObstacles
    type = randi([0 1]); % random 0 (sphere) or 1 (cube)
    x = rand*(xrange(2)-xrange(1)) + xrange(1);
    y = rand*(yrange(2)-yrange(1)) + yrange(1);
    z = rand*(zrange(2)-zrange(1)) + zrange(1);
    size_val = 0.03 + 0.03*rand; % random size 0.03 - 0.06

    if type == 0 % Sphere
        baseColor = [0.6 0.6 0.6] + 0.1*(rand(1,3)-0.5);
    else % Cube
        baseColor = [0.5 0.3 0.1] + 0.1*(rand(1,3)-0.5);
    end

    obstacles(i,:) = [type, x, y, z, size_val, baseColor];

    % Random velocity
    velocities(i,:) = 0.1*(rand(1,3)-0.5)*2; % random between [-0.1, 0.1]
end

% Save files
writematrix(obstacles, 'obstacles3D.csv');
save('velocities.mat', 'velocities');
