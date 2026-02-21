function q = euler_to_quaternion(a, b, c, convention)
%EULER_TO_QUATERNION Convert Euler angles to unit quaternion.
%   q = rotation_converter.euler_to_quaternion(A, B, C, CONVENTION)
%   A, B, C: Euler angles in radians.
%   CONVENTION: string like 'xyz', 'zyx', 'zyz', etc. (12 conventions).
%   Returns: [w, x, y, z] unit quaternion.
%
%   Supports all 12 Euler conventions:
%     Tait-Bryan: xyz, xzy, yxz, yzx, zxy, zyx
%     Proper Euler: xyx, xzx, yxy, yzy, zxz, zyz

    convention = lower(convention);
    rotation_converter.internal.require(length(convention) == 3, ...
        'convention must be 3 characters', convention);

    valid = {'xyz','xzy','yxz','yzx','zxy','zyx', ...
             'xyx','xzx','yxy','yzy','zxz','zyz'};
    rotation_converter.internal.require(any(strcmp(convention, valid)), ...
        'unknown Euler convention', convention);

    ax1 = rotation_converter.internal.axis_index(convention(1));
    ax2 = rotation_converter.internal.axis_index(convention(2));
    ax3 = rotation_converter.internal.axis_index(convention(3));

    q1 = rotation_converter.internal.elementary_quaternion(ax1, a);
    q2 = rotation_converter.internal.elementary_quaternion(ax2, b);
    q3 = rotation_converter.internal.elementary_quaternion(ax3, c);

    q = rotation_converter.quaternion_multiply( ...
            rotation_converter.quaternion_multiply(q1, q2), q3);

    q = rotation_converter.normalize_quaternion(q);
end
