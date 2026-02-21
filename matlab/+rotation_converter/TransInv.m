function T_inv = TransInv(T)
%TransInv Inverse of an SE(3) homogeneous transformation matrix.
%   T_inv = rotation_converter.TransInv(T)
%   T: 4x4 SE(3) matrix.
%   Returns: 4x4 SE(3) inverse [R' -R'*p; 0 1].
%
%   Uses the structure of SE(3) for efficient inversion (avoids inv(T)).

    rotation_converter.internal.require(all(size(T) == [4, 4]), ...
        'T must be 4x4');

    R = T(1:3, 1:3);
    p = T(1:3, 4);

    T_inv = [R', -R' * p; 0 0 0 1];
end
