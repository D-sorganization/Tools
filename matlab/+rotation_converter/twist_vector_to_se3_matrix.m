function M = twist_vector_to_se3_matrix(xi)
%TWIST_VECTOR_TO_SE3_MATRIX Convert 6-vector twist to 4x4 se(3) matrix.
%   M = rotation_converter.twist_vector_to_se3_matrix(XI)
%   XI: [omega1 omega2 omega3 v1 v2 v3] 6-element twist vector.
%   Returns: 4x4 se(3) matrix [[omega] v; 0 0].

    xi = xi(:)';
    rotation_converter.internal.require(numel(xi) == 6, ...
        'twist must have 6 elements', numel(xi));

    omega = xi(1:3);
    v = xi(4:6);

    M = zeros(4, 4);
    M(1:3, 1:3) = rotation_converter.internal.skew_symmetric(omega);
    M(1:3, 4) = v(:);

    rotation_converter.internal.ensure(all(M(4, :) == 0), ...
        'bottom row must be zero');
end
