function [a, b, c] = rotation_matrix_to_euler(R, convention)
%ROTATION_MATRIX_TO_EULER Extract Euler angles from a rotation matrix.
%   [A, B, C] = rotation_converter.rotation_matrix_to_euler(R, CONVENTION)
%   R: 3x3 SO(3) rotation matrix.
%   CONVENTION: string like 'xyz', 'zyx', 'zyz', etc. (12 supported).
%   Returns: A, B, C Euler angles in radians.
%
%   Handles gimbal lock for all 12 conventions (Tait-Bryan and proper Euler).

    rotation_converter.internal.validate_rotation_matrix(R);
    convention = lower(convention);

    % 1-based axis indices (MATLAB convention)
    i = rotation_converter.internal.axis_index(convention(1));
    j = rotation_converter.internal.axis_index(convention(2));
    k = rotation_converter.internal.axis_index(convention(3));

    is_proper = (convention(1) == convention(3));

    if is_proper
        % Proper Euler angles: R = Ri(a) * Rj(b) * Ri(c)
        other = 6 - i - j;  % 1+2+3=6, so other = 6 - i - j

        % Sign factor: +1 if (i,j) is cyclic (1->2, 2->3, 3->1), else -1
        if mod(j - i + 3, 3) == 1
            sign_factor = 1.0;
        else
            sign_factor = -1.0;
        end

        cb = max(-1.0, min(1.0, R(i, i)));
        b = acos(cb);

        if abs(sin(b)) > 1e-8
            % General case
            a = atan2(R(j, i), -sign_factor * R(other, i));
            c = atan2(R(i, j),  sign_factor * R(i, other));
        else
            % Gimbal lock: b ≈ 0 or b ≈ pi
            a = atan2(sign_factor * R(other, j), R(j, j));
            c = 0.0;
        end
    else
        % Tait-Bryan angles: R = Ri(a) * Rj(b) * Rk(c), all axes distinct
        if mod(j - i + 3, 3) == 1
            sign_factor = 1.0;
        else
            sign_factor = -1.0;
        end

        sb = max(-1.0, min(1.0, sign_factor * R(i, k)));
        b = asin(sb);

        if abs(cos(b)) > 1e-6
            a = atan2(-sign_factor * R(j, k), R(k, k));
            c = atan2(-sign_factor * R(i, j), R(i, i));
        else
            % Gimbal lock
            a = atan2(sign_factor * R(k, j), R(j, j));
            c = 0.0;
        end
    end
end
