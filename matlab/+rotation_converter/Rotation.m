classdef Rotation
%ROTATION Immutable rotation representation with multiple output formats.
%
%   Stores a rotation as a unit quaternion [w, x, y, z] and provides
%   conversion to/from all supported representations.
%
%   Factory methods (static):
%     rot = rotation_converter.Rotation.identity()
%     rot = rotation_converter.Rotation.from_quaternion(q)
%     rot = rotation_converter.Rotation.from_rotation_matrix(R)
%     rot = rotation_converter.Rotation.from_euler(a, b, c, convention)
%     rot = rotation_converter.Rotation.from_axis_angle(axis, angle)
%     rot = rotation_converter.Rotation.from_rodrigues(r)
%
%   Output methods:
%     q = rot.as_quaternion()
%     R = rot.as_rotation_matrix()
%     [a, b, c] = rot.as_euler(convention)
%     [axis, angle] = rot.as_axis_angle()
%     r = rot.as_rodrigues()
%
%   Composition:
%     rot3 = rot1.compose(rot2)
%     rot_inv = rot.inverse()
%
%   Compatible with MATLAB R2014b+ and Octave 5.0+.

    properties (Access = private)
        q_  % internal unit quaternion [w, x, y, z]
    end

    methods
        function obj = Rotation(q)
            %ROTATION Construct from raw unit quaternion (prefer factory methods).
            q = q(:)';
            rotation_converter.internal.require(numel(q) == 4, ...
                'quaternion must have 4 elements');
            obj.q_ = rotation_converter.normalize_quaternion(q);
        end

        % --- Output conversions ---

        function q = as_quaternion(obj)
            %AS_QUATERNION Return [w, x, y, z] unit quaternion (copy).
            q = obj.q_;
        end

        function R = as_rotation_matrix(obj)
            %AS_ROTATION_MATRIX Return 3x3 SO(3) rotation matrix.
            R = rotation_converter.quaternion_to_rotation_matrix(obj.q_);
        end

        function [a, b, c] = as_euler(obj, convention)
            %AS_EULER Return Euler angles for the given convention.
            [a, b, c] = rotation_converter.quaternion_to_euler(obj.q_, convention);
        end

        function [axis, angle] = as_axis_angle(obj)
            %AS_AXIS_ANGLE Return axis (unit vector) and angle (radians).
            [axis, angle] = rotation_converter.quaternion_to_axis_angle(obj.q_);
        end

        function r = as_rodrigues(obj)
            %AS_RODRIGUES Return 3-element Rodrigues vector (axis * angle).
            r = rotation_converter.quaternion_to_rodrigues(obj.q_);
        end

        % --- Composition ---

        function result = compose(obj, other)
            %COMPOSE Return the composition of this rotation with another.
            q_new = rotation_converter.quaternion_multiply(obj.q_, other.q_);
            result = rotation_converter.Rotation(q_new);
        end

        function result = inverse(obj)
            %INVERSE Return the inverse rotation.
            result = rotation_converter.Rotation( ...
                rotation_converter.quaternion_conjugate(obj.q_));
        end

        % --- Display ---

        function disp(obj)
            fprintf('Rotation(q=[%.6f, %.6f, %.6f, %.6f])\n', ...
                    obj.q_(1), obj.q_(2), obj.q_(3), obj.q_(4));
        end

        function result = eq(obj, other)
            %EQ Test equality (up to sign ambiguity).
            if ~isa(other, 'rotation_converter.Rotation')
                result = false;
                return;
            end
            d = abs(dot(obj.q_, other.q_));
            result = abs(d - 1.0) < 1e-10;
        end
    end

    methods (Static)
        function obj = identity()
            %IDENTITY Create identity rotation.
            obj = rotation_converter.Rotation([1, 0, 0, 0]);
        end

        function obj = from_quaternion(q)
            %FROM_QUATERNION Create Rotation from quaternion [w, x, y, z].
            obj = rotation_converter.Rotation(q);
        end

        function obj = from_rotation_matrix(R)
            %FROM_ROTATION_MATRIX Create Rotation from 3x3 SO(3) matrix.
            q = rotation_converter.rotation_matrix_to_quaternion(R);
            obj = rotation_converter.Rotation(q);
        end

        function obj = from_euler(a, b, c, convention)
            %FROM_EULER Create Rotation from Euler angles.
            q = rotation_converter.euler_to_quaternion(a, b, c, convention);
            obj = rotation_converter.Rotation(q);
        end

        function obj = from_axis_angle(axis, angle)
            %FROM_AXIS_ANGLE Create Rotation from axis-angle.
            q = rotation_converter.axis_angle_to_quaternion(axis, angle);
            obj = rotation_converter.Rotation(q);
        end

        function obj = from_rodrigues(r)
            %FROM_RODRIGUES Create Rotation from Rodrigues vector.
            q = rotation_converter.rodrigues_to_quaternion(r);
            obj = rotation_converter.Rotation(q);
        end
    end
end
