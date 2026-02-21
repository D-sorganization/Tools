function [xi, theta] = homogeneous_to_twist_angle(T)
%HOMOGENEOUS_TO_TWIST_ANGLE Matrix logarithm of SE(3) -> (twist, angle).
%   [XI, THETA] = rotation_converter.homogeneous_to_twist_angle(T)
%   T: 4x4 SE(3) homogeneous transformation matrix.
%   Returns: XI (6-element twist), THETA (non-negative scalar).
%
%   Decomposes T such that exp([xi]*theta) = T.

    rotation_converter.internal.require(all(size(T) == [4, 4]), ...
        'homogeneous matrix must be 4x4');
    rotation_converter.internal.require( ...
        norm(T(4, :) - [0 0 0 1]) < 1e-9, 'bottom row must be [0,0,0,1]');

    R = T(1:3, 1:3);
    p = T(1:3, 4);

    % Check if R is identity (pure translation)
    if norm(R - eye(3), 'fro') < 1e-9
        p_norm = norm(p);
        if p_norm < 1e-12
            % Identity transform
            xi = zeros(1, 6);
            theta = 0.0;
        else
            % Pure translation
            v_hat = p(:)' / p_norm;
            xi = [0 0 0, v_hat];
            theta = p_norm;
        end
    else
        % General case: extract axis-angle from R
        rotation_converter.internal.validate_rotation_matrix(R);
        [axis, theta] = rotation_converter.rotation_matrix_to_axis_angle(R);

        if theta < 1e-12
            % Near-identity rotation, treat as pure translation
            p_norm = norm(p);
            if p_norm < 1e-12
                xi = zeros(1, 6);
                theta = 0.0;
            else
                v_hat = p(:)' / p_norm;
                xi = [0 0 0, v_hat];
                theta = p_norm;
            end
        else
            omega = axis(:)';
            K = rotation_converter.internal.skew_symmetric(omega);
            % G_inv = (1/theta)*I - 0.5*K + (1/theta - 0.5*cot(theta/2))*K^2
            cot_half = cos(theta / 2) / sin(theta / 2);
            G_inv = (1.0 / theta) * eye(3) - 0.5 * K ...
                    + (1.0 / theta - 0.5 * cot_half) * (K * K);
            v = (G_inv * p(:))';
            xi = [omega, v];
        end
    end

    % Postconditions (single exit point)
    rotation_converter.internal.ensure(numel(xi) == 6, ...
        'result twist must have 6 elements');
    rotation_converter.internal.ensure(theta >= 0, ...
        'angle must be non-negative');
end
