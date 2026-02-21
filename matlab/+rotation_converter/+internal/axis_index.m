function idx = axis_index(ch)
%AXIS_INDEX Convert axis character to 1-based index.
%   idx = axis_index('x') returns 1
%   idx = axis_index('y') returns 2
%   idx = axis_index('z') returns 3

    switch lower(ch)
        case 'x'
            idx = 1;
        case 'y'
            idx = 2;
        case 'z'
            idx = 3;
        otherwise
            error('rotation_converter:PreconditionError', ...
                  'Invalid axis character: %s', ch);
    end
end
