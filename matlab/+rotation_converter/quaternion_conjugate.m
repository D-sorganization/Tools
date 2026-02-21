function q_out = quaternion_conjugate(q)
%QUATERNION_CONJUGATE Compute the conjugate of a quaternion.
%   q_out = rotation_converter.quaternion_conjugate(Q)
%   Q: [w, x, y, z] quaternion.
%   Returns: [w, -x, -y, -z].

    q = q(:)';
    q_out = [q(1), -q(2), -q(3), -q(4)];
end
