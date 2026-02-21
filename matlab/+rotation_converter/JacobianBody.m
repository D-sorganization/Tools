function Jb = JacobianBody(Blist, thetalist)
%JacobianBody Compute the body Jacobian for a serial chain.
%   Jb = rotation_converter.JacobianBody(BLIST, THETALIST)
%   BLIST: 6xn body screw axes.
%   THETALIST: n-element joint angle vector.
%   Returns: 6xn body Jacobian matrix.
%
%   J_b(:,i) = Ad_{exp(-[Bn]*tn)...exp(-[Bi+1]*ti+1)} * Bi
%
%   Reference: Lynch & Park, Modern Robotics, Eq. 5.15.

    rotation_converter.internal.require(ndims(Blist) == 2 && size(Blist, 1) == 6, ...
        'Blist must be 6xn');
    rotation_converter.internal.require_finite(Blist, 'Blist');
    n = size(Blist, 2);
    thetalist = thetalist(:)';
    rotation_converter.internal.require(numel(thetalist) == n, ...
        sprintf('thetalist must have %d elements', n));
    rotation_converter.internal.require_finite(thetalist, 'thetalist');

    Jb = Blist;  % copy; last column stays the same
    T = eye(4);
    for i = (n-1):-1:1
        se3 = rotation_converter.VecTose3(-Blist(:, i+1)') * thetalist(i+1);
        T = T * rotation_converter.MatrixExp6(se3);
        Ad = rotation_converter.internal.adjoint_from_T(T);
        Jb(:, i) = Ad * Blist(:, i);
    end

    rotation_converter.internal.ensure(all(size(Jb) == [6, n]), ...
        'Jacobian must be 6xn');
end
