function [a, b, c] = quaternion_to_euler(q, convention)
%QUATERNION_TO_EULER Convert unit quaternion to Euler angles.
%   [A, B, C] = rotation_converter.quaternion_to_euler(Q, CONVENTION)
%   Q: [w, x, y, z] unit quaternion.
%   CONVENTION: string like 'xyz', 'zyx', 'zyz', etc.
%   Returns: A, B, C Euler angles in radians.
%
%   Routes through rotation matrix for robust angle extraction.
%   Handles gimbal lock for all 12 conventions.

    q = q(:)';
    rotation_converter.internal.require(numel(q) == 4, ...
        'quaternion must have 4 elements');
    rotation_converter.internal.require_finite(q, 'quaternion');

    convention = lower(convention);
    valid = {'xyz','xzy','yxz','yzx','zxy','zyx', ...
             'xyx','xzx','yxy','yzy','zxz','zyz'};
    rotation_converter.internal.require(any(strcmp(convention, valid)), ...
        'unknown Euler convention', convention);

    R = rotation_converter.quaternion_to_rotation_matrix( ...
            rotation_converter.normalize_quaternion(q));
    [a, b, c] = rotation_converter.rotation_matrix_to_euler(R, convention);
end
