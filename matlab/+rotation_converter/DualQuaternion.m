classdef DualQuaternion %
        DUALQUATERNION
            A Dual Quaternion representation of rigid body displacement.%
        % Combines rotation(real part) and
    translation(dual part) into a single % mathematical entity structured as
    : Q = q_r + eps *
                    q_d

                    properties(SetAccess = private) qr
                    % Real quaternion component(w, x, y, z) qd
                    % Dual quaternion component(w, x, y, z)
end

    methods function obj =
        DualQuaternion(qr, qd) % Initialize dual quaternion from real and
        dual quaternions.if size (qr, 2) ~=
            4 || size(qr, 1) ~=
                1 error('Real quaternion must be a 1x4 vector.');
end if size (qd, 2) ~= 4 || size(qd, 1) ~=
                           1 error('Dual quaternion must be a 1x4 vector.');
end obj.qr = qr;
obj.qd = qd;
end

    function new_obj =
        multiply(obj, other) % Multiply two dual quaternions.qr_new =
            rotation_converter.quaternion_multiply(obj.qr, other.qr);
qd_new = rotation_converter.quaternion_multiply(obj.qr, other.qd) +
         ... rotation_converter.quaternion_multiply(obj.qd, other.qr);
new_obj = rotation_converter.DualQuaternion(qr_new, qd_new);
end

    function t = extract_translation(obj) % Extract the translation 3 -
                 vector from the dual quaternion.qr_conj =
                     rotation_converter.quaternion_conjugate(obj.qr);
t_quat = 2.0 * rotation_converter.quaternion_multiply(obj.qd, qr_conj);
t = t_quat(2 : 4);
end end

methods(Static)
function obj = from_translation_rotation(translation, rotation_quaternion) %
                   Construct from a translation vector and
               rotation quaternion.if numel (translation) ~=
                   3 error('Translation must be a 3-vector.');
end translation = translation( :)'; qr =
    rotation_converter.normalize_quaternion(rotation_quaternion);

% Dual part = (1 / 2) *t *qr t_quat =
    [ 0.0, translation(1), translation(2), translation(3) ];
qd = 0.5 * rotation_converter.quaternion_multiply(t_quat, qr);

obj = rotation_converter.DualQuaternion(qr, qd);
end end end
