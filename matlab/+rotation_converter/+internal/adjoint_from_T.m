function Ad = adjoint_from_T(T)
%ADJOINT_FROM_T Internal helper: 6x6 adjoint of an SE(3) matrix.
%   Ad = adjoint_from_T(T) computes [R 0; [p]xR R].

    R = T(1:3, 1:3);
    p = T(1:3, 4);
    pK = rotation_converter.internal.skew_symmetric(p(:)');

    Ad = zeros(6, 6);
    Ad(1:3, 1:3) = R;
    Ad(4:6, 4:6) = R;
    Ad(4:6, 1:3) = pK * R;
end
