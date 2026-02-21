function R = quaternion_to_rotation_matrix(q)
%QUATERNION_TO_ROTATION_MATRIX Convert unit quaternion to 3x3 rotation matrix.
%   R = rotation_converter.quaternion_to_rotation_matrix(Q)
%   Q: [w, x, y, z] unit quaternion.
%   Returns: 3x3 SO(3) rotation matrix.
%
%   Precondition:  Q is a unit quaternion.
%   Postcondition: R is in SO(3) (orthogonal, det=+1).

    q = q(:)';
    rotation_converter.internal.require(numel(q) == 4, ...
        'quaternion must have 4 elements');
    rotation_converter.internal.require_finite(q, 'quaternion');
    rotation_converter.internal.require(abs(norm(q) - 1.0) < 1e-6, ...
        'quaternion must be unit', norm(q));

    w = q(1); x = q(2); y = q(3); z = q(4);

    R = [ 1 - 2*(y^2 + z^2),   2*(x*y - w*z),     2*(x*z + w*y);
          2*(x*y + w*z),       1 - 2*(x^2 + z^2),   2*(y*z - w*x);
          2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x^2 + y^2) ];

    rotation_converter.internal.ensure(abs(det(R) - 1.0) < 1e-9, ...
        'result must be SO(3)');
end
