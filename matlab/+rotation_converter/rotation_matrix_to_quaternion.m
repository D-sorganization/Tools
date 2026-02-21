function q = rotation_matrix_to_quaternion(R)
%ROTATION_MATRIX_TO_QUATERNION Convert 3x3 rotation matrix to unit quaternion.
%   q = rotation_converter.rotation_matrix_to_quaternion(R)
%   R: 3x3 SO(3) rotation matrix.
%   Returns: [w, x, y, z] unit quaternion (canonical form, w >= 0).
%
%   Uses Shepperd's method for numerical stability at all rotation angles.

    rotation_converter.internal.validate_rotation_matrix(R);

    % Shepperd's method: pick the largest diagonal element to avoid
    % division by a near-zero number.
    tr = trace(R);
    candidates = [tr, R(1,1), R(2,2), R(3,3)];
    [~, best] = max(candidates);

    switch best
        case 1
            % tr is largest: w is largest component
            s = sqrt(tr + 1.0) * 2;  % s = 4*w
            q = [ 0.25 * s, ...
                  (R(3,2) - R(2,3)) / s, ...
                  (R(1,3) - R(3,1)) / s, ...
                  (R(2,1) - R(1,2)) / s ];
        case 2
            % R(1,1) largest: x is largest component
            s = sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2;  % s = 4*x
            q = [ (R(3,2) - R(2,3)) / s, ...
                  0.25 * s, ...
                  (R(1,2) + R(2,1)) / s, ...
                  (R(1,3) + R(3,1)) / s ];
        case 3
            % R(2,2) largest: y is largest component
            s = sqrt(1.0 + R(2,2) - R(1,1) - R(3,3)) * 2;  % s = 4*y
            q = [ (R(1,3) - R(3,1)) / s, ...
                  (R(1,2) + R(2,1)) / s, ...
                  0.25 * s, ...
                  (R(2,3) + R(3,2)) / s ];
        case 4
            % R(3,3) largest: z is largest component
            s = sqrt(1.0 + R(3,3) - R(1,1) - R(2,2)) * 2;  % s = 4*z
            q = [ (R(2,1) - R(1,2)) / s, ...
                  (R(1,3) + R(3,1)) / s, ...
                  (R(2,3) + R(3,2)) / s, ...
                  0.25 * s ];
    end

    % Canonical form: ensure w >= 0
    if q(1) < 0
        q = -q;
    end

    rotation_converter.internal.ensure(abs(norm(q) - 1.0) < 1e-9, ...
        'result must be unit quaternion');
end
