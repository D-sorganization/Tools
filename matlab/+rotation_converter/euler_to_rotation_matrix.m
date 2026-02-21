function R = euler_to_rotation_matrix(a, b, c, convention)
%EULER_TO_ROTATION_MATRIX Convert Euler angles to rotation matrix.
%   R = rotation_converter.euler_to_rotation_matrix(A, B, C, CONVENTION)
%   Routes through quaternion hub.

    q = rotation_converter.euler_to_quaternion(a, b, c, convention);
    R = rotation_converter.quaternion_to_rotation_matrix(q);
end
