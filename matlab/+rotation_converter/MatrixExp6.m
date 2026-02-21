function T = MatrixExp6(se3mat)
%MatrixExp6 Matrix exponential of se(3) -> SE(3).
%   T = rotation_converter.MatrixExp6(SE3MAT)
%   SE3MAT: 4x4 se(3) matrix.
%   Returns: 4x4 SE(3) homogeneous transformation matrix.
%
%   Reference: Lynch & Park, Modern Robotics, Eq. 3.88.

    rotation_converter.internal.require(all(size(se3mat) == [4, 4]), ...
        'se(3) matrix must be 4x4');
    rotation_converter.internal.require_finite(se3mat, 'se(3) matrix');

    omega_mat = se3mat(1:3, 1:3);
    omega_vec = rotation_converter.so3ToVec(omega_mat);
    v = se3mat(1:3, 4);
    theta = norm(omega_vec);

    T = eye(4);

    if rotation_converter.internal.near_zero(theta)
        % Pure translation
        T(1:3, 4) = v;
        return;
    end

    omega_hat = omega_mat / theta;
    v_unit = v / theta;

    R = rotation_converter.MatrixExp3(omega_mat);
    % G(theta) from Lynch & Park Eq. 3.84
    G = eye(3) * theta ...
        + (1.0 - cos(theta)) * omega_hat ...
        + (theta - sin(theta)) * (omega_hat * omega_hat);

    T(1:3, 1:3) = R;
    T(1:3, 4) = G * v_unit;

    rotation_converter.internal.ensure(abs(det(T(1:3, 1:3)) - 1.0) < 1e-9, ...
        'result must be SE(3)');
end
