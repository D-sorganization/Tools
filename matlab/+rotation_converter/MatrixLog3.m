function so3mat = MatrixLog3(R)
%MatrixLog3 Matrix logarithm of SO(3) -> so(3).
%   so3mat = rotation_converter.MatrixLog3(R)
%   R: 3x3 rotation matrix in SO(3).
%   Returns: 3x3 skew-symmetric matrix in so(3).
%
%   Reference: Lynch & Park, Modern Robotics, Algorithm.

    rotation_converter.internal.require(all(size(R) == [3, 3]), ...
        'rotation matrix must be 3x3');
    rotation_converter.internal.require_finite(R, 'rotation matrix');

    cos_theta = max(-1.0, min(1.0, (trace(R) - 1.0) / 2.0));

    if rotation_converter.internal.near_zero(cos_theta - 1.0)
        % theta ~ 0: near identity
        so3mat = zeros(3, 3);
        return;
    end

    if rotation_converter.internal.near_zero(cos_theta + 1.0)
        % theta ~ pi
        theta = pi;
        % Find column of R + I with largest norm
        RpI = R + eye(3);
        col_norms = [norm(RpI(:, 1)), norm(RpI(:, 2)), norm(RpI(:, 3))];
        [~, best_col] = max(col_norms);
        omega = RpI(:, best_col) / norm(RpI(:, best_col));
        so3mat = rotation_converter.VecToso3(omega(:)') * theta;
        return;
    end

    theta = acos(cos_theta);
    omega_hat = (R - R') / (2.0 * sin(theta));
    so3mat = omega_hat * theta;
end
