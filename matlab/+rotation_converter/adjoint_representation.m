function Ad = adjoint_representation(T)
%ADJOINT_REPRESENTATION Compute 6x6 adjoint representation of SE(3) matrix.
%   Ad = rotation_converter.adjoint_representation(T)
%   T: 4x4 SE(3) homogeneous transformation matrix.
%   Returns: 6x6 adjoint matrix [R 0; [p]xR R].

    rotation_converter.internal.require(all(size(T) == [4, 4]), ...
        'T must be 4x4');
    rotation_converter.internal.require( ...
        norm(T(4, :) - [0 0 0 1]) < 1e-9, 'bottom row must be [0,0,0,1]');

    R = T(1:3, 1:3);
    p = T(1:3, 4);

    rotation_converter.internal.validate_rotation_matrix(R);

    pK = rotation_converter.internal.skew_symmetric(p);

    Ad = zeros(6, 6);
    Ad(1:3, 1:3) = R;
    Ad(4:6, 4:6) = R;
    Ad(4:6, 1:3) = pK * R;

    rotation_converter.internal.ensure(all(size(Ad) == [6, 6]), ...
        'result must be 6x6');
end
