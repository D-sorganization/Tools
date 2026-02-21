function trajectory = frisbee_flight(n_frames, speed, spin_rate, launch_angle_deg, tilt_deg)
%FRISBEE_FLIGHT Generate SE(3) trajectory of a spinning frisbee.
%   trajectory = rotation_converter.frisbee_flight()
%   trajectory = rotation_converter.frisbee_flight(N_FRAMES, SPEED, SPIN_RATE, LAUNCH_DEG, TILT_DEG)
%
%   N_FRAMES: number of trajectory frames (default 60).
%   SPEED: initial speed in m/s (default 14).
%   SPIN_RATE: spin rate in rad/s (default 7).
%   LAUNCH_ANGLE_DEG: launch angle in degrees (default 8).
%   TILT_DEG: initial tilt in degrees (default 15).
%
%   Returns: cell array of N 4x4 SE(3) matrices.

    if nargin < 1; n_frames = 60; end
    if nargin < 2; speed = 14.0; end
    if nargin < 3; spin_rate = 7.0; end
    if nargin < 4; launch_angle_deg = 8.0; end
    if nargin < 5; tilt_deg = 15.0; end

    g = 9.81;
    launch_angle = deg2rad(launch_angle_deg);
    tilt = deg2rad(tilt_deg);
    vx = speed * cos(launch_angle);
    vz = speed * sin(launch_angle);

    t_flight = 2 * vz / g + 0.5;
    dt = t_flight / (n_frames - 1);

    trajectory = cell(1, n_frames);
    for i = 1:n_frames
        t = (i - 1) * dt;

        px = vx * t;
        py = 0;
        pz = vz * t - 0.5 * g * t^2;

        % Frisbee spin about Z, tilt about X
        spin_angle = spin_rate * t;
        R_tilt = rotation_converter.axis_angle_to_rotation_matrix([1 0 0], tilt);
        R_spin = rotation_converter.axis_angle_to_rotation_matrix([0 0 1], spin_angle);
        R = R_tilt * R_spin;

        T = eye(4);
        T(1:3, 1:3) = R;
        T(1:3, 4) = [px; py; pz];
        trajectory{i} = T;
    end
end
