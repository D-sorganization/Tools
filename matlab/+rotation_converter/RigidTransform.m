classdef RigidTransform
%RIGIDTRANSFORM Frame-aware SE(3) rigid body transformation.
%
%   Stores a 4x4 SE(3) matrix with source and target frame labels.
%   All composition operations enforce frame consistency.
%
%   Factory methods (static):
%     T = rotation_converter.RigidTransform.identity(frame)
%     T = rotation_converter.RigidTransform.from_matrix(T, source, target)
%     T = rotation_converter.RigidTransform.from_rotation_translation(R, p, source, target)
%     T = rotation_converter.RigidTransform.from_quaternion_translation(q, p, source, target)
%     T = rotation_converter.RigidTransform.from_euler_translation(a,b,c,p,conv,source,target)
%     T = rotation_converter.RigidTransform.from_axis_angle_translation(ax,ang,p,source,target)
%     T = rotation_converter.RigidTransform.from_rodrigues_translation(r, p, source, target)
%     T = rotation_converter.RigidTransform.from_twist(xi, theta, source, target)
%     T = rotation_converter.RigidTransform.pure_translation(p, source, target)
%     T = rotation_converter.RigidTransform.pure_rotation(rot, source, target)
%
%   Compatible with MATLAB R2014b+ and Octave 5.0+.

    properties (Access = private)
        T_       % 4x4 SE(3) matrix
        source_  % source frame label (string)
        target_  % target frame label (string)
    end

    methods
        function obj = RigidTransform(T, source_frame, target_frame)
            %RIGIDTRANSFORM Construct from 4x4 SE(3) matrix and frame labels.
            rotation_converter.internal.require(all(size(T) == [4, 4]), ...
                'T must be 4x4');
            rotation_converter.internal.require( ...
                norm(T(4, :) - [0 0 0 1]) < 1e-9, ...
                'bottom row must be [0,0,0,1]');
            rotation_converter.internal.validate_rotation_matrix(T(1:3, 1:3));
            obj.T_ = T;
            obj.source_ = source_frame;
            obj.target_ = target_frame;
        end

        % --- Properties ---

        function f = source_frame(obj)
            f = obj.source_;
        end

        function f = target_frame(obj)
            f = obj.target_;
        end

        function p = translation(obj)
            %TRANSLATION Return 3-element translation vector (copy).
            p = obj.T_(1:3, 4)';
        end

        function R = rotation_matrix(obj)
            %ROTATION_MATRIX Return 3x3 SO(3) rotation matrix (copy).
            R = obj.T_(1:3, 1:3);
        end

        % --- Output conversions ---

        function T = as_matrix(obj)
            %AS_MATRIX Return 4x4 SE(3) matrix (copy).
            T = obj.T_;
        end

        function [R, p] = as_rotation_translation(obj)
            R = obj.T_(1:3, 1:3);
            p = obj.T_(1:3, 4)';
        end

        function rot = as_rotation(obj)
            %AS_ROTATION Return a Rotation object for the rotational part.
            rot = rotation_converter.Rotation.from_rotation_matrix(obj.T_(1:3, 1:3));
        end

        function [q, p] = as_quaternion_translation(obj)
            q = rotation_converter.rotation_matrix_to_quaternion(obj.T_(1:3, 1:3));
            p = obj.T_(1:3, 4)';
        end

        function [euler_angles, p] = as_euler_translation(obj, convention)
            [a, b, c] = rotation_converter.rotation_matrix_to_euler( ...
                obj.T_(1:3, 1:3), convention);
            euler_angles = [a, b, c];
            p = obj.T_(1:3, 4)';
        end

        function [axis, angle, p] = as_axis_angle_translation(obj)
            [axis, angle] = rotation_converter.rotation_matrix_to_axis_angle( ...
                obj.T_(1:3, 1:3));
            p = obj.T_(1:3, 4)';
        end

        function [r, p] = as_rodrigues_translation(obj)
            q = rotation_converter.rotation_matrix_to_quaternion(obj.T_(1:3, 1:3));
            r = rotation_converter.quaternion_to_rodrigues(q);
            p = obj.T_(1:3, 4)';
        end

        function [xi, theta] = as_twist(obj)
            [xi, theta] = rotation_converter.homogeneous_to_twist_angle(obj.T_);
        end

        function screw = as_screw(obj)
            [xi, ~] = rotation_converter.homogeneous_to_twist_angle(obj.T_);
            omega_norm = norm(xi(1:3));
            if omega_norm > 1e-12
                screw = rotation_converter.twist_to_screw(xi);
            else
                v_norm = norm(xi(4:6));
                if v_norm > 1e-12
                    screw = rotation_converter.twist_to_screw(xi);
                else
                    screw = struct('axis', [0 0 1], 'point', [0 0 0], 'pitch', 0);
                end
            end
        end

        % --- Predicates ---

        function result = is_identity(obj, tol)
            if nargin < 2; tol = 1e-9; end
            result = norm(obj.T_ - eye(4), 'fro') < tol;
        end

        function result = is_pure_translation(obj, tol)
            if nargin < 2; tol = 1e-9; end
            result = norm(obj.T_(1:3, 1:3) - eye(3), 'fro') < tol;
        end

        function result = is_pure_rotation(obj, tol)
            if nargin < 2; tol = 1e-9; end
            result = norm(obj.T_(1:3, 4)) < tol;
        end

        % --- Composition ---

        function result = compose(obj, other)
            %COMPOSE Compose two frame-consistent rigid transforms.
            if ~strcmp(obj.target_, other.source_)
                fe = rotation_converter.FrameError( ...
                    obj.target_, other.source_, 'compose');
                fe.raise();
            end
            T_new = obj.T_ * other.T_;
            result = rotation_converter.RigidTransform( ...
                T_new, obj.source_, other.target_);
        end

        function result = mtimes(obj, other)
            %MTIMES Allow T1 * T2 syntax.
            if isa(other, 'rotation_converter.RigidTransform')
                result = obj.compose(other);
            else
                error('rotation_converter:TypeError', ...
                      'Can only multiply RigidTransform with another RigidTransform');
            end
        end

        function result = inverse(obj)
            %INVERSE Return the inverse transform (with swapped frames).
            T_inv = rotation_converter.TransInv(obj.T_);
            result = rotation_converter.RigidTransform( ...
                T_inv, obj.target_, obj.source_);
        end

        % --- Point/vector transformations ---

        function p_out = apply_point(obj, point)
            %APPLY_POINT Transform a 3D point.
            point = point(:);
            p_out = (obj.T_(1:3, 1:3) * point + obj.T_(1:3, 4))';
        end

        function v_out = apply_vector(obj, vec)
            %APPLY_VECTOR Transform a 3D vector (rotation only, no translation).
            vec = vec(:);
            v_out = (obj.T_(1:3, 1:3) * vec)';
        end

        function P_out = apply_points(obj, P)
            %APPLY_POINTS Transform Nx3 matrix of points.
            R = obj.T_(1:3, 1:3);
            t = obj.T_(1:3, 4)';
            P_out = (R * P')' + repmat(t, size(P, 1), 1);
        end

        function V_out = apply_vectors(obj, V)
            %APPLY_VECTORS Transform Nx3 matrix of vectors.
            R = obj.T_(1:3, 1:3);
            V_out = (R * V')';
        end

        % --- Body/space twist conversions ---

        function Vb = body_twist(obj)
            %BODY_TWIST Compute body-frame twist (matrix logarithm).
            se3 = rotation_converter.MatrixLog6(obj.T_);
            Vb = rotation_converter.se3ToVec(se3);
        end

        function Vs = space_twist(obj)
            %SPACE_TWIST Compute space-frame twist.
            Vb = obj.body_twist();
            Ad = rotation_converter.adjoint_representation(obj.T_);
            Vs = (Ad * Vb(:))';
        end

        function Vs = body_to_space_twist(obj, Vb)
            %BODY_TO_SPACE_TWIST Convert body twist to space twist.
            Ad = rotation_converter.adjoint_representation(obj.T_);
            Vs = (Ad * Vb(:))';
        end

        function Vb = space_to_body_twist(obj, Vs)
            %SPACE_TO_BODY_TWIST Convert space twist to body twist.
            Ad_inv = rotation_converter.adjoint_representation( ...
                rotation_converter.TransInv(obj.T_));
            Vb = (Ad_inv * Vs(:))';
        end

        % --- Wrench transformations ---

        function Fs = body_to_space_wrench(obj, Fb)
            %BODY_TO_SPACE_WRENCH Convert body wrench to space wrench.
            Ad = rotation_converter.adjoint_representation( ...
                rotation_converter.TransInv(obj.T_));
            Fs = (Ad' * Fb(:))';
        end

        function Fb = space_to_body_wrench(obj, Fs)
            %SPACE_TO_BODY_WRENCH Convert space wrench to body wrench.
            Ad = rotation_converter.adjoint_representation(obj.T_);
            Fb = (Ad' * Fs(:))';
        end

        % --- Display ---

        function disp(obj)
            fprintf('RigidTransform(%s -> %s)\n', obj.source_, obj.target_);
            disp(obj.T_);
        end

        function result = eq(obj, other)
            if ~isa(other, 'rotation_converter.RigidTransform')
                result = false;
                return;
            end
            result = strcmp(obj.source_, other.source_) && ...
                     strcmp(obj.target_, other.target_) && ...
                     norm(obj.T_ - other.T_, 'fro') < 1e-10;
        end
    end

    methods (Static)
        function obj = identity(frame)
            obj = rotation_converter.RigidTransform(eye(4), frame, frame);
        end

        function obj = from_matrix(T, source, target)
            obj = rotation_converter.RigidTransform(T, source, target);
        end

        function obj = from_rotation_translation(R, p, source, target)
            T = rotation_converter.RpToTrans(R, p);
            obj = rotation_converter.RigidTransform(T, source, target);
        end

        function obj = from_quaternion_translation(q, p, source, target)
            R = rotation_converter.quaternion_to_rotation_matrix(q);
            T = rotation_converter.RpToTrans(R, p(:));
            obj = rotation_converter.RigidTransform(T, source, target);
        end

        function obj = from_euler_translation(a, b, c, p, convention, source, target)
            R = rotation_converter.euler_to_rotation_matrix(a, b, c, convention);
            T = rotation_converter.RpToTrans(R, p(:));
            obj = rotation_converter.RigidTransform(T, source, target);
        end

        function obj = from_axis_angle_translation(axis, angle, p, source, target)
            R = rotation_converter.axis_angle_to_rotation_matrix(axis, angle);
            T = rotation_converter.RpToTrans(R, p(:));
            obj = rotation_converter.RigidTransform(T, source, target);
        end

        function obj = from_rodrigues_translation(r, p, source, target)
            q = rotation_converter.rodrigues_to_quaternion(r);
            R = rotation_converter.quaternion_to_rotation_matrix(q);
            T = rotation_converter.RpToTrans(R, p(:));
            obj = rotation_converter.RigidTransform(T, source, target);
        end

        function obj = from_twist(xi, theta, source, target)
            T = rotation_converter.twist_angle_to_homogeneous(xi, theta);
            obj = rotation_converter.RigidTransform(T, source, target);
        end

        function obj = pure_translation(p, source, target)
            T = eye(4);
            T(1:3, 4) = p(:);
            obj = rotation_converter.RigidTransform(T, source, target);
        end

        function obj = pure_rotation(rot, source, target)
            R = rot.as_rotation_matrix();
            T = rotation_converter.RpToTrans(R, [0; 0; 0]);
            obj = rotation_converter.RigidTransform(T, source, target);
        end
    end
end
