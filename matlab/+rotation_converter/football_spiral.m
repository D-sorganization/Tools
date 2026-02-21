function trajectory = football_spiral(n_frames, speed, spin_rate, launch_angle_deg)
%FOOTBALL_SPIRAL Generate SE(3) trajectory of a spiraling football throw.
%   trajectory = rotation_converter.football_spiral()
%   trajectory = rotation_converter.football_spiral(N_FRAMES, SPEED, SPIN_RATE, LAUNCH_ANGLE_DEG)
%
%   N_FRAMES: number of trajectory frames (default 60).
%   SPEED: initial speed in m/s (default 20).
%   SPIN_RATE: spin rate in rad/s (default 10).
%   LAUNCH_ANGLE_DEG: launch angle in degrees (default 35).
%
%   Returns: cell array of N 4x4 SE(3) matrices.

    if nargin < 1; n_frames = 60; end
    if nargin < 2; speed = 20.0; end
    if nargin < 3; spin_rate = 10.0; end
    if nargin < 4; launch_angle_deg = 35.0; end

    g = 9.81;
    launch_angle = deg2rad(launch_angle_deg);
    vx = speed * cos(launch_angle);
    vz = speed * sin(launch_angle);

    % Total flight time (up and back down)
    t_flight = 2 * vz / g;
    dt = t_flight / (n_frames - 1);

    trajectory = cell(1, n_frames);
    for i = 1:n_frames
        t = (i - 1) * dt;

        % Ballistic trajectory
        px = vx * t;
        py = 0;
        pz = vz * t - 0.5 * g * t^2;

        % Velocity direction for pitch angle
        cur_vz = vz - g * t;
        pitch = atan2(cur_vz, vx);

        % Football spin about its long axis
        spin_angle = spin_rate * t;

        % Rotation: first pitch about Y, then spin about X (body axis)
        R_pitch = rotation_converter.axis_angle_to_rotation_matrix([0 1 0], pitch);
        R_spin = rotation_converter.axis_angle_to_rotation_matrix([1 0 0], spin_angle);
        R = R_pitch * R_spin;

        T = eye(4);
        T(1:3, 1:3) = R;
        T(1:3, 4) = [px; py; pz];
        trajectory{i} = T;
    end
end
