function T = FKinSpace(M, Slist, thetalist)
%FKinSpace Forward kinematics in the space frame (product of exponentials).
%   T = rotation_converter.FKinSpace(M, SLIST, THETALIST)
%   M: 4x4 home configuration of end-effector.
%   SLIST: 6xn matrix of space-frame screw axes (columns).
%   THETALIST: n-element vector of joint angles.
%   Returns: 4x4 SE(3) end-effector configuration.
%
%   T = exp([S1]*theta1) * ... * exp([Sn]*thetan) * M
%
%   Reference: Lynch & Park, Modern Robotics, Eq. 4.13.

    rotation_converter.internal.require(all(size(M) == [4, 4]), 'M must be 4x4');
    rotation_converter.internal.require_finite(M, 'M');
    rotation_converter.internal.require(size(Slist, 1) == 6, ...
        'Slist must have 6 rows');
    rotation_converter.internal.require_finite(Slist, 'Slist');
    n = size(Slist, 2);
    thetalist = thetalist(:)';
    rotation_converter.internal.require(numel(thetalist) == n, ...
        sprintf('thetalist must have %d elements', n));
    rotation_converter.internal.require_finite(thetalist, 'thetalist');

    T = eye(4);
    for i = 1:n
        se3 = rotation_converter.VecTose3(Slist(:, i)') * thetalist(i);
        T = T * rotation_converter.MatrixExp6(se3);
    end
    T = T * M;

    rotation_converter.internal.ensure(abs(det(T(1:3, 1:3)) - 1.0) < 1e-9, ...
        'result must be SE(3)');
end
