function ensure(condition, msg, varargin)
%ENSURE Design-by-Contract postcondition check.
%   ensure(CONDITION, MSG) throws an error if CONDITION is false.

    if ~condition
        if nargin > 2
            error('rotation_converter:PostconditionError', ...
                  'Postcondition violated: %s (got %s)', msg, mat2str(varargin{1}));
        else
            error('rotation_converter:PostconditionError', ...
                  'Postcondition violated: %s', msg);
        end
    end
end
