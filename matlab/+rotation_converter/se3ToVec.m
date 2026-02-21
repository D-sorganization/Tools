function V = se3ToVec(se3mat)
%se3ToVec Convert 4x4 se(3) matrix to 6-vector twist.
%   V = rotation_converter.se3ToVec(SE3MAT)
%   SE3MAT: 4x4 se(3) matrix.
%   Returns: [omega1 omega2 omega3 v1 v2 v3] spatial velocity.

    rotation_converter.internal.require(all(size(se3mat) == [4, 4]), ...
        'se(3) matrix must be 4x4');

    V = [se3mat(3,2), se3mat(1,3), se3mat(2,1), ...
         se3mat(1,4), se3mat(2,4), se3mat(3,4)];
end
