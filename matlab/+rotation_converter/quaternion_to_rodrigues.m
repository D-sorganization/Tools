function r = quaternion_to_rodrigues(q)
%QUATERNION_TO_RODRIGUES Convert unit quaternion to Rodrigues vector.
%   r = rotation_converter.quaternion_to_rodrigues(Q)
%   Q: [w, x, y, z] unit quaternion.
%   Returns: 3-element Rodrigues vector (axis * angle).

    q = q(:)';
    rotation_converter.internal.require(numel(q) == 4, ...
        'quaternion must have 4 elements');
    rotation_converter.internal.require_finite(q, 'quaternion');

    [axis, angle] = rotation_converter.quaternion_to_axis_angle(q);
    r = axis * angle;
end
