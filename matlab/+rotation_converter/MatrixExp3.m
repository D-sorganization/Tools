function R = MatrixExp3(so3mat)
%MatrixExp3 Matrix exponential of so(3) -> SO(3) via Rodrigues' formula.
%   R = rotation_converter.MatrixExp3(SO3MAT)
%   SO3MAT: 3x3 so(3) matrix (skew-symmetric * angle).
%   Returns: 3x3 rotation matrix in SO(3).
%
%   Reference: Lynch & Park, Modern Robotics, Eq. 3.51.

    rotation_converter.internal.require(all(size(so3mat) == [3, 3]), ...
        'so(3) matrix must be 3x3');
    rotation_converter.internal.require_finite(so3mat, 'so(3) matrix');

    omega_vec = rotation_converter.so3ToVec(so3mat);
    theta = norm(omega_vec);

    if rotation_converter.internal.near_zero(theta)
        R = eye(3);
        return;
    end

    omega_hat = so3mat / theta;
    R = eye(3) + sin(theta) * omega_hat + (1 - cos(theta)) * (omega_hat * omega_hat);

    rotation_converter.internal.ensure(abs(det(R) - 1.0) < 1e-9, ...
        'result must be SO(3)');
end
