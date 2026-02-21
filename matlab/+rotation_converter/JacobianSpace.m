function Js = JacobianSpace(Slist, thetalist)
%JacobianSpace Compute the space Jacobian for a serial chain.
%   Js = rotation_converter.JacobianSpace(SLIST, THETALIST)
%   SLIST: 6xn space screw axes.
%   THETALIST: n-element joint angle vector.
%   Returns: 6xn space Jacobian matrix.
%
%   J_s(:,i) = Ad_{exp([S1]*t1)...exp([Si-1]*ti-1)} * Si
%
%   Reference: Lynch & Park, Modern Robotics, Eq. 5.11.

    rotation_converter.internal.require(ndims(Slist) == 2 && size(Slist, 1) == 6, ...
        'Slist must be 6xn');
    rotation_converter.internal.require_finite(Slist, 'Slist');
    n = size(Slist, 2);
    thetalist = thetalist(:)';
    rotation_converter.internal.require(numel(thetalist) == n, ...
        sprintf('thetalist must have %d elements', n));
    rotation_converter.internal.require_finite(thetalist, 'thetalist');

    Js = Slist;  % copy; first column stays the same
    T = eye(4);
    for i = 2:n
        se3 = rotation_converter.VecTose3(Slist(:, i-1)') * thetalist(i-1);
        T = T * rotation_converter.MatrixExp6(se3);
        Ad = rotation_converter.internal.adjoint_from_T(T);
        Js(:, i) = Ad * Slist(:, i);
    end

    rotation_converter.internal.ensure(all(size(Js) == [6, n]), ...
        'Jacobian must be 6xn');
end
