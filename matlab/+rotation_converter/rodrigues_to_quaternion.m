function q = rodrigues_to_quaternion(r)
%RODRIGUES_TO_QUATERNION Convert Rodrigues vector to unit quaternion.
%   q = rotation_converter.rodrigues_to_quaternion(R_VEC)
%   R_VEC: 3-element Rodrigues vector (axis * angle).
%   Returns: [w, x, y, z] unit quaternion.

    r = r(:)';
    rotation_converter.internal.require(numel(r) == 3, ...
        'rodrigues vector must have 3 elements');
    rotation_converter.internal.require_finite(r, 'rodrigues vector');

    angle = norm(r);
    if angle < 1e-12
        q = [1, 0, 0, 0];
    else
        axis = r / angle;
        q = rotation_converter.axis_angle_to_quaternion(axis, angle);
    end
end
