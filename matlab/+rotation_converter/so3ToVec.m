function omega = so3ToVec(so3mat)
%so3ToVec Convert 3x3 skew-symmetric so(3) matrix to 3-vector.
%   omega = rotation_converter.so3ToVec(SO3MAT)
%   SO3MAT: 3x3 skew-symmetric matrix.
%   Returns: 3-element vector [omega1 omega2 omega3].

    rotation_converter.internal.require(all(size(so3mat) == [3, 3]), ...
        'so(3) matrix must be 3x3');
    omega = [so3mat(3,2), so3mat(1,3), so3mat(2,1)];
end
