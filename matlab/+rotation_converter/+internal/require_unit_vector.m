function require_unit_vector(v, name)
%REQUIRE_UNIT_VECTOR Verify v is a unit vector (norm ≈ 1).
%   require_unit_vector(V, NAME) throws PreconditionError if norm(V) is not ~1.

    rotation_converter.internal.require_finite(v, name);
    n = norm(v);
    if abs(n - 1.0) > 1e-6
        error('rotation_converter:PreconditionError', ...
              'Precondition violated: %s must be unit vector (norm=%g)', name, n);
    end
end
