function [R, p] = TransToRp(T)
%TransToRp Extract rotation matrix and position from SE(3) matrix.
%   [R, p] = rotation_converter.TransToRp(T)
%   T: 4x4 SE(3) homogeneous transformation matrix.
%   Returns: R (3x3 rotation), p (3x1 position column vector).

    rotation_converter.internal.require(all(size(T) == [4, 4]), ...
        'T must be 4x4');
    R = T(1:3, 1:3);
    p = T(1:3, 4);
end
