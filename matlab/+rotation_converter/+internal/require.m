function require(condition, msg, varargin)
%REQUIRE Design-by-Contract precondition check.
%   require(CONDITION, MSG) throws an error if CONDITION is false.
%   require(CONDITION, MSG, VALUE) includes VALUE in the error message.

    if ~condition
        if nargin > 2
            error('rotation_converter:PreconditionError', ...
                  'Precondition violated: %s (got %s)', msg, mat2str(varargin{1}));
        else
            error('rotation_converter:PreconditionError', ...
                  'Precondition violated: %s', msg);
        end
    end
end
