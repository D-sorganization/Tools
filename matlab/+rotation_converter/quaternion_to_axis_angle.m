function [axis, angle] = quaternion_to_axis_angle(q)
%QUATERNION_TO_AXIS_ANGLE Convert unit quaternion to axis-angle representation.
%   [AXIS, ANGLE] = rotation_converter.quaternion_to_axis_angle(Q)
%   Q: [w, x, y, z] unit quaternion.
%   Returns: AXIS (3-element unit vector), ANGLE (non-negative radians).
%
%   For zero rotation, returns axis=[0,0,1], angle=0.

    q = q(:)';
    rotation_converter.internal.require(numel(q) == 4, ...
        'quaternion must have 4 elements');
    rotation_converter.internal.require_finite(q, 'quaternion');
    rotation_converter.internal.require(abs(norm(q) - 1.0) < 1e-6, ...
        'quaternion must be unit', norm(q));

    % Canonical form
    if q(1) < 0
        q = -q;
    end

    w = q(1);
    v = q(2:4);
    sin_half = norm(v);

    if sin_half < 1e-12
        % Near-zero rotation
        axis = [0, 0, 1];
        angle = 0.0;
    else
        axis = v / sin_half;
        angle = 2.0 * atan2(sin_half, w);
    end

    % Ensure non-negative angle
    if angle < 0
        angle = -angle;
        axis = -axis;
    end
end
