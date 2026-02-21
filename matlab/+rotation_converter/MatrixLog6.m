function se3mat = MatrixLog6(T)
%MatrixLog6 Matrix logarithm of SE(3) -> se(3).
%   se3mat = rotation_converter.MatrixLog6(T)
%   T: 4x4 SE(3) matrix.
%   Returns: 4x4 se(3) matrix such that expm(result) = T.

    rotation_converter.internal.require(all(size(T) == [4, 4]), ...
        'SE(3) matrix must be 4x4');
    rotation_converter.internal.require_finite(T, 'SE(3) matrix');

    [R, p] = rotation_converter.TransToRp(T);
    omega_mat = rotation_converter.MatrixLog3(R);
    omega_vec = rotation_converter.so3ToVec(omega_mat);
    theta = norm(omega_vec);

    se3mat = zeros(4, 4);

    if rotation_converter.internal.near_zero(theta)
        % Pure translation
        se3mat(1:3, 4) = p;
        return;
    end

    omega_hat = omega_mat / theta;
    G_inv = eye(3) / theta ...
            - omega_hat / 2.0 ...
            + (1.0 / theta - 1.0 / (2.0 * tan(theta / 2.0))) ...
              * (omega_hat * omega_hat);

    se3mat(1:3, 1:3) = omega_mat;
    se3mat(1:3, 4) = (G_inv * p) * theta;
end
