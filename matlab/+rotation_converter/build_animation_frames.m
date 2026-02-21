function frames = build_animation_frames(trajectory, body_axis_length)
%BUILD_ANIMATION_FRAMES Build frame data for visualization from SE(3) trajectory.
%   frames = rotation_converter.build_animation_frames(TRAJECTORY)
%   frames = rotation_converter.build_animation_frames(TRAJECTORY, BODY_AXIS_LENGTH)
%
%   TRAJECTORY: cell array of 4x4 SE(3) matrices.
%   BODY_AXIS_LENGTH: length of body frame arrows (default 0.3).
%
%   Returns: struct array with fields for each frame:
%     .origin - 3-element origin position
%     .x_axis, .y_axis, .z_axis - 3-element axis direction vectors
%     .T - original 4x4 SE(3) matrix

    if nargin < 2; body_axis_length = 0.3; end

    n = numel(trajectory);
    frames = struct('origin', {}, 'x_axis', {}, 'y_axis', {}, 'z_axis', {}, 'T', {});

    for i = 1:n
        T = trajectory{i};
        R = T(1:3, 1:3);
        p = T(1:3, 4)';

        frames(i).origin = p;
        frames(i).x_axis = (R * [body_axis_length; 0; 0])';
        frames(i).y_axis = (R * [0; body_axis_length; 0])';
        frames(i).z_axis = (R * [0; 0; body_axis_length])';
        frames(i).T = T;
    end
end
