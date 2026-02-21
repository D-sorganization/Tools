function q_out = normalize_quaternion(q)
%NORMALIZE_QUATERNION Normalize a quaternion to unit length.
%   q_out = rotation_converter.normalize_quaternion(Q)
%   Q: [w, x, y, z] quaternion (4-element vector).
%   Returns: unit quaternion with positive w (canonical form).
%
%   Precondition:  Q must be finite and non-zero.
%   Postcondition: norm(q_out) ≈ 1.

    q = q(:)';  % ensure row vector
    rotation_converter.internal.require(numel(q) == 4, ...
        'quaternion must have 4 elements', numel(q));
    rotation_converter.internal.require_finite(q, 'quaternion');

    n = norm(q);
    rotation_converter.internal.require(n > 1e-12, ...
        'quaternion must be non-zero');

    q_out = q / n;

    % Canonical form: ensure w >= 0
    if q_out(1) < 0
        q_out = -q_out;
    end

    rotation_converter.internal.ensure(abs(norm(q_out) - 1.0) < 1e-12, ...
        'result must be unit quaternion');
end
