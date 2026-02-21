function [thetalist, success] = IKinBody(Blist, M, T_desired, thetalist0, eomg, ev, max_iter)
%IKinBody Iterative inverse kinematics using Newton-Raphson in body frame.
%   [THETALIST, SUCCESS] = rotation_converter.IKinBody(BLIST, M, T_DESIRED, THETALIST0)
%   [THETALIST, SUCCESS] = rotation_converter.IKinBody(..., EOMG, EV, MAX_ITER)
%
%   BLIST: 6xn body screw axes.
%   M: 4x4 home configuration.
%   T_DESIRED: 4x4 desired end-effector SE(3) pose.
%   THETALIST0: n-element initial guess for joint angles.
%   EOMG: Angular error tolerance (default 1e-4 rad).
%   EV: Linear error tolerance (default 1e-4).
%   MAX_ITER: Maximum iterations (default 100).
%
%   Returns: THETALIST (joint angles), SUCCESS (logical).
%
%   Reference: Lynch & Park, Modern Robotics, Algorithm.

    if nargin < 5; eomg = 1e-4; end
    if nargin < 6; ev = 1e-4; end
    if nargin < 7; max_iter = 100; end

    rotation_converter.internal.require(all(size(M) == [4, 4]), 'M must be 4x4');
    rotation_converter.internal.require_finite(M, 'M');
    rotation_converter.internal.require(all(size(T_desired) == [4, 4]), ...
        'T_desired must be 4x4');
    rotation_converter.internal.require_finite(T_desired, 'T_desired');
    rotation_converter.internal.require(ndims(Blist) == 2 && size(Blist, 1) == 6, ...
        'Blist must be 6xn');
    rotation_converter.internal.require_finite(Blist, 'Blist');
    thetalist = thetalist0(:)';
    rotation_converter.internal.require_finite(thetalist, 'thetalist0');
    rotation_converter.internal.require(eomg > 0, ...
        'angular tolerance must be positive', eomg);
    rotation_converter.internal.require(ev > 0, ...
        'linear tolerance must be positive', ev);
    rotation_converter.internal.require(max_iter > 0, ...
        'max_iter must be positive', max_iter);

    for iter = 1:max_iter
        T_current = rotation_converter.FKinBody(M, Blist, thetalist);
        T_error = rotation_converter.TransInv(T_current) * T_desired;
        Vb = rotation_converter.se3ToVec(rotation_converter.MatrixLog6(T_error));
        omega_err = norm(Vb(1:3));
        v_err = norm(Vb(4:6));

        if omega_err < eomg && v_err < ev
            success = true;
            return;
        end

        Jb = rotation_converter.JacobianBody(Blist, thetalist);
        % Damped least-squares (same as Python's lstsq)
        thetalist = thetalist + (Jb \ Vb(:))';
    end

    success = false;
end
