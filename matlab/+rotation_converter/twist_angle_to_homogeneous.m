function T = twist_angle_to_homogeneous(xi, theta)
%TWIST_ANGLE_TO_HOMOGENEOUS Compute matrix exponential of twist * angle -> SE(3).
%   T = rotation_converter.twist_angle_to_homogeneous(XI, THETA)
%   XI: 6-element unit twist [omega; v] with ||omega||=1 or omega=0.
%   THETA: scalar angle/displacement.
%   Returns: 4x4 SE(3) homogeneous transformation matrix.

    xi = xi(:)';
    rotation_converter.internal.require(numel(xi) == 6, ...
        'twist must have 6 elements');
    rotation_converter.internal.require_finite(xi, 'twist');
    rotation_converter.internal.require(isfinite(theta), 'theta must be finite');

    omega = xi(1:3);
    v = xi(4:6);
    omega_norm = norm(omega);

    T = eye(4);

    if omega_norm < 1e-12
        % Pure translation
        T(1:3, 4) = v(:) * theta;
    else
        rotation_converter.internal.require(abs(omega_norm - 1.0) < 1e-6, ...
            'omega must be unit vector when non-zero', omega_norm);

        K = rotation_converter.internal.skew_symmetric(omega);
        R = eye(3) + sin(theta) * K + (1 - cos(theta)) * (K * K);
        % G(theta) = I*theta + (1-cos(theta))*K + (theta-sin(theta))*K^2
        G = eye(3) * theta + (1 - cos(theta)) * K + (theta - sin(theta)) * (K * K);

        T(1:3, 1:3) = R;
        T(1:3, 4) = G * v(:);
    end

    rotation_converter.internal.ensure(norm(T(4, :) - [0 0 0 1]) < 1e-12, ...
        'bottom row must be [0,0,0,1]');
end
