function R = axis_angle_to_rotation_matrix(axis, angle)
%AXIS_ANGLE_TO_ROTATION_MATRIX Convert axis-angle to 3x3 rotation matrix.
%   R = rotation_converter.axis_angle_to_rotation_matrix(AXIS, ANGLE)
%   AXIS: 3-element unit vector.
%   ANGLE: rotation angle in radians.
%   Returns: 3x3 SO(3) rotation matrix via Rodrigues' formula.

    axis = axis(:)';
    rotation_converter.internal.require(numel(axis) == 3, ...
        'axis must have 3 elements');
    rotation_converter.internal.require_unit_vector(axis, 'axis');

    K = rotation_converter.internal.skew_symmetric(axis);
    R = eye(3) + sin(angle) * K + (1 - cos(angle)) * (K * K);

    rotation_converter.internal.ensure(abs(det(R) - 1.0) < 1e-9, ...
        'result must be SO(3)');
end
