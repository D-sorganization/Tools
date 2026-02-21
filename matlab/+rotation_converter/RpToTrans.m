function T = RpToTrans(R, p)
%RpToTrans Build SE(3) matrix from rotation and position.
%   T = rotation_converter.RpToTrans(R, P)
%   R: 3x3 rotation matrix.
%   P: 3-element position vector.
%   Returns: 4x4 SE(3) homogeneous transformation matrix.

    rotation_converter.internal.require(all(size(R) == [3, 3]), ...
        'R must be 3x3');
    p = p(:);
    rotation_converter.internal.require(numel(p) == 3, ...
        'p must have 3 elements');

    T = [R, p; 0 0 0 1];
end
