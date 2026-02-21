classdef FrameError
%FRAMEERROR Error descriptor for frame mismatch during composition.
%
%   Works in both MATLAB and Octave (does not extend MException for
%   Octave compatibility). Use raise() to throw the error.
%
%   Properties:
%     expected_frame - the frame that was expected
%     actual_frame   - the frame that was found
%     operation_desc - description of the failed operation

    properties
        expected_frame
        actual_frame
        operation_desc
    end

    methods
        function obj = FrameError(expected, actual, operation, detail)
            if nargin < 4; detail = ''; end
            obj.expected_frame = expected;
            obj.actual_frame = actual;
            obj.operation_desc = operation;
        end

        function raise(obj)
            %RAISE Throw this frame error as a MATLAB/Octave error.
            error('rotation_converter:FrameError', ...
                  'Frame mismatch in %s: expected "%s", got "%s"', ...
                  obj.operation_desc, obj.expected_frame, obj.actual_frame);
        end
    end
end
