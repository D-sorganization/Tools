function T = FKinBody(M, Blist, thetalist)
%FKinBody Forward kinematics in the body frame (product of exponentials).
%   T = rotation_converter.FKinBody(M, BLIST, THETALIST)
%   M: 4x4 home configuration of end-effector.
%   BLIST: 6xn matrix of body-frame screw axes (columns).
%   THETALIST: n-element vector of joint angles.
%   Returns: 4x4 SE(3) end-effector configuration.
%
%   T = M * exp([B1]*theta1) * ... * exp([Bn]*thetan)
%
%   Reference: Lynch & Park, Modern Robotics, Eq. 4.17.

    rotation_converter.internal.require(all(size(M) == [4, 4]), 'M must be 4x4');
    rotation_converter.internal.require_finite(M, 'M');
    rotation_converter.internal.require(size(Blist, 1) == 6, ...
        'Blist must have 6 rows');
    rotation_converter.internal.require_finite(Blist, 'Blist');
    n = size(Blist, 2);
    thetalist = thetalist(:)';
    rotation_converter.internal.require(numel(thetalist) == n, ...
        sprintf('thetalist must have %d elements', n));
    rotation_converter.internal.require_finite(thetalist, 'thetalist');

    T = M;
    for i = 1:n
        se3 = rotation_converter.VecTose3(Blist(:, i)') * thetalist(i);
        T = T * rotation_converter.MatrixExp6(se3);
    end

    rotation_converter.internal.ensure(abs(det(T(1:3, 1:3)) - 1.0) < 1e-9, ...
        'result must be SE(3)');
end
