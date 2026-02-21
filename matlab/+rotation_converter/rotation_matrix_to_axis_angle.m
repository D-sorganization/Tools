function [axis, angle] = rotation_matrix_to_axis_angle(R)
%ROTATION_MATRIX_TO_AXIS_ANGLE Convert rotation matrix to axis-angle.
%   [AXIS, ANGLE] = rotation_converter.rotation_matrix_to_axis_angle(R)
%   R: 3x3 SO(3) rotation matrix.
%   Returns: AXIS (unit vector), ANGLE (non-negative radians).
%
%   Routes through quaternion hub for numerical stability.

    rotation_converter.internal.validate_rotation_matrix(R);
    q = rotation_converter.rotation_matrix_to_quaternion(R);
    [axis, angle] = rotation_converter.quaternion_to_axis_angle(q);
end
