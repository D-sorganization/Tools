function q_out = quaternion_multiply(q1, q2)
%QUATERNION_MULTIPLY Hamilton product of two quaternions.
%   q_out = rotation_converter.quaternion_multiply(Q1, Q2)
%   Q1, Q2: [w, x, y, z] quaternions.
%   Returns: Q1 * Q2 (Hamilton product).

    q1 = q1(:)';
    q2 = q2(:)';
    rotation_converter.internal.require(numel(q1) == 4, ...
        'q1 must have 4 elements');
    rotation_converter.internal.require(numel(q2) == 4, ...
        'q2 must have 4 elements');

    w1 = q1(1); x1 = q1(2); y1 = q1(3); z1 = q1(4);
    w2 = q2(1); x2 = q2(2); y2 = q2(3); z2 = q2(4);

    q_out = [ w1*w2 - x1*x2 - y1*y2 - z1*z2, ...
              w1*x2 + x1*w2 + y1*z2 - z1*y2, ...
              w1*y2 - x1*z2 + y1*w2 + z1*x2, ...
              w1*z2 + x1*y2 - y1*x2 + z1*w2 ];
end
