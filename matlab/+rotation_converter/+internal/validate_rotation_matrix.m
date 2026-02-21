function validate_rotation_matrix(R)
%VALIDATE_ROTATION_MATRIX Check that R is a valid 3x3 SO(3) rotation matrix.
%   Verifies orthogonality (R'*R ≈ I) and determinant ≈ +1.

    rotation_converter.internal.require_finite(R, 'rotation matrix');
    rotation_converter.internal.require(all(size(R) == [3, 3]), ...
        'rotation matrix must be 3x3', size(R));
    rotation_converter.internal.require(norm(R' * R - eye(3), 'fro') < 1e-6, ...
        'rotation matrix must be orthogonal');
    rotation_converter.internal.require(abs(det(R) - 1.0) < 1e-6, ...
        'rotation matrix determinant must be +1');
end
