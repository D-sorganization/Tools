function result = near_zero(val, tol)
%NEAR_ZERO Check if a scalar is effectively zero.
%   result = near_zero(VAL) checks with default tol=1e-12.
%   result = near_zero(VAL, TOL) uses the given tolerance.

    if nargin < 2
        tol = 1e-12;
    end
    result = abs(val) < tol;
end
