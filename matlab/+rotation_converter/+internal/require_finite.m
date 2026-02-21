function require_finite(M, name)
%REQUIRE_FINITE Verify all elements of M are finite (no NaN or Inf).
%   require_finite(M, NAME) throws PreconditionError if M contains NaN/Inf.

    if ~all(isfinite(M(:)))
        error('rotation_converter:PreconditionError', ...
              'Precondition violated: %s must be finite', name);
    end
end
