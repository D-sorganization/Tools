function q = elementary_quaternion(axis_idx, angle)
%ELEMENTARY_QUATERNION Quaternion for rotation about a single coordinate axis.
%   q = elementary_quaternion(AXIS_IDX, ANGLE)
%   AXIS_IDX: 1=x, 2=y, 3=z
%   Returns: [w, x, y, z] quaternion.

    w = cos(angle / 2);
    s = sin(angle / 2);
    q = [w, 0, 0, 0];
    q(axis_idx + 1) = s;  % +1 because q = [w, x, y, z]
end
