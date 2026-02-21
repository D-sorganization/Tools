function q = axis_angle_to_quaternion(axis, angle)
%AXIS_ANGLE_TO_QUATERNION Convert axis-angle to unit quaternion.
%   q = rotation_converter.axis_angle_to_quaternion(AXIS, ANGLE)
%   AXIS: 3-element unit vector.
%   ANGLE: rotation angle in radians.
%   Returns: [w, x, y, z] unit quaternion.

    axis = axis(:)';
    rotation_converter.internal.require(numel(axis) == 3, ...
        'axis must have 3 elements');
    rotation_converter.internal.require_unit_vector(axis, 'axis');
    rotation_converter.internal.require(isfinite(angle), ...
        'angle must be finite');

    w = cos(angle / 2);
    s = sin(angle / 2);
    q = [w, axis(1)*s, axis(2)*s, axis(3)*s];

    % Canonical form
    if q(1) < 0
        q = -q;
    end

    rotation_converter.internal.ensure(abs(norm(q) - 1.0) < 1e-9, ...
        'result must be unit quaternion');
end
