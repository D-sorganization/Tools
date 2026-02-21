function xi = se3_matrix_to_twist_vector(M)
%SE3_MATRIX_TO_TWIST_VECTOR Convert 4x4 se(3) matrix to 6-vector twist.
%   xi = rotation_converter.se3_matrix_to_twist_vector(M)
%   M: 4x4 se(3) matrix.
%   Returns: [omega1 omega2 omega3 v1 v2 v3] twist vector.

    rotation_converter.internal.require(all(size(M) == [4, 4]), ...
        'se(3) matrix must be 4x4');
    rotation_converter.internal.require(all(M(4, :) == 0), ...
        'bottom row must be zero');

    omega = [M(3,2), M(1,3), M(2,1)];
    v = M(1:3, 4)';

    xi = [omega, v];
    rotation_converter.internal.ensure(numel(xi) == 6, ...
        'result must have 6 elements');
end
